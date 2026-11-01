"""Train the motion tracker with brax PPO on MuJoCo Warp (GPU).

Run inside ``.runtime/world-model/mjx-venv`` with the host driver shim::

    LD_LIBRARY_PATH=$PWD/.runtime/warp-cuda-shim:$PWD/.runtime/nvidia-595.91.07 \\
    CUDA_VISIBLE_DEVICES=0 .runtime/world-model/mjx-venv/bin/python -m \\
        world_model.tracker.train_tracker --library .runtime/world-model/tracker/motion-library-v1 \\
        --output .runtime/world-model/tracker/runs/<name>
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np


def _jax_compat():
    """brax 0.14.2 still calls jax.device_put_replicated, removed in JAX 0.10."""
    try:
        jax.device_put_replicated  # noqa: B018 - present in older JAX
        return
    except AttributeError:
        pass

    def device_put_replicated(x, devices):
        return jax.tree.map(lambda a: jax.device_put(jp.stack([jp.asarray(a)] * len(devices))), x)

    jax.device_put_replicated = device_put_replicated


def network_kwargs(args):
    """PPO network settings shared by training and evaluation.

    "bounded_normal" (default): Gaussian whose mean is soft-sign bounded to
    (-1, 1) and whose state-independent std starts at ``init_noise_std``.
    brax's tanh_normal ignores ``init_noise_std`` (std starts at ~0.69), and
    an unbounded normal let v1's mean drift beyond the env's action clip.
    """
    distribution = args.get("distribution", "normal")
    kwargs = {"policy_hidden_layer_sizes": (1024, 512, 256),
              "value_hidden_layer_sizes": (1024, 512, 256)}
    if distribution == "bounded_normal":
        kwargs.update(distribution_type="normal", noise_std_type="log",
                      init_noise_std=args["init_noise_std"], mean_clip_scale=1.)
    elif distribution == "normal":     # v1 configuration
        kwargs.update(distribution_type="normal", noise_std_type="log",
                      init_noise_std=args["init_noise_std"])
    else:
        kwargs.update(distribution_type="tanh_normal")
    return kwargs


def _sources():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def smoke(env, num_envs=256, steps=50):
    from .mjx_tracking import wrap_for_training
    wrapped = wrap_for_training(env, episode_length=env._config.episode_length)
    reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
    state = reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
    zero = jp.zeros((num_envs, env.action_size))
    state = step(state, zero)
    jax.block_until_ready(state.obs)
    start, failures, truncations = time.time(), 0., 0.
    for _ in range(steps):
        state = step(state, zero)
        failures += float(jp.sum(state.done * (1 - state.info["truncation"])))
        truncations += float(jp.sum(state.info["truncation"]))
    jax.block_until_ready(state.obs)
    elapsed = time.time() - start
    return {"obsSize": int(state.obs.shape[-1]), "finiteObs": bool(jp.all(jp.isfinite(state.obs))),
            "zeroResidualStepsPerSecond": num_envs * steps / elapsed,
            "failuresPerEnvStep": failures / (num_envs * steps),
            "truncationsPerEnvStep": truncations / (num_envs * steps),
            "meanPositionError": float(jp.mean(state.metrics["position_error"]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timesteps", type=int, default=300_000_000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--evals", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    # Small exploration noise as in PHC: large PD-target noise on 63 joints
    # makes the body jitter (training power ~40 kW/step at std 0.7).
    parser.add_argument("--init-noise-std", type=float, default=.15)
    parser.add_argument("--entropy-cost", type=float, default=0.)
    # "normal" lets the mean drift outside the env's [-1, 1] clip where it has
    # no gradient (v1 diverged that way); tanh_normal bounds actions smoothly.
    parser.add_argument("--distribution", default="bounded_normal",
                        choices=("bounded_normal", "tanh_normal", "normal"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restore", default=None, help="params pickle from a previous run")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--include-seated", action="store_true")
    parser.add_argument("--include-terrain", action="store_true", help="stage-4 step boxes (terrain clips)")
    # Rounds of adaptive_train.sh: one run directory, cumulative step numbers.
    parser.add_argument("--clip-weights", default="", help="sweep_clips output; failed clips drawn more often")
    parser.add_argument("--step-offset", type=int, default=0, help="steps already trained in earlier rounds")
    parser.add_argument("--planned-timesteps", type=int, default=0, help="total steps over all rounds")
    parser.add_argument("--more-rounds", action="store_true", help="do not mark the run finished")
    args = parser.parse_args()

    _jax_compat()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from .mjx_tracking import TrackingEnv, default_config, wrap_for_training

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.restore is not None or args.smoke_only)
    config = default_config()
    # a seat box adds box-capsule contacts on top of the floor contacts
    config.naconmax = (128 if args.include_terrain else 96 if args.include_seated else 64) * args.num_envs
    config.include_seated = args.include_seated
    config.include_terrain = args.include_terrain
    config.clip_weights = args.clip_weights
    env = TrackingEnv(args.library, config)
    offset = args.step_offset
    info = {"args": vars(args), "config": config.to_dict(), "clips": len(env.clips),
            "stepOffset": offset, "plannedTimesteps": args.planned_timesteps or offset + args.timesteps,
            "library": env.library_summary, "sourceSha256": _sources(),
            "jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
    info["smoke"] = smoke(env, num_envs=256)
    print(json.dumps(info["smoke"]), flush=True)
    (output / "run.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    if args.smoke_only:
        return

    network_factory = functools.partial(ppo_networks.make_ppo_networks, **network_kwargs(vars(args)))
    log = (output / "progress.jsonl").open("a")
    start = time.time()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    def progress(step, metrics):
        row = {"step": int(step) + offset, "elapsedSeconds": time.time() - start,
               **{k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        keys = [k for k in row if k.startswith("eval/episode_") or k in ("step", "elapsedSeconds")]
        print(json.dumps({k: round(row[k], 4) for k in keys}), flush=True)

    def save(step, make_policy, params):
        del make_policy
        with (checkpoints / f"params_{int(step) + offset:012d}.pkl").open("wb") as f:
            pickle.dump(jax.device_get(params), f)

    restore = None
    if args.restore:
        with open(args.restore, "rb") as f:
            restore = pickle.load(f)
    make_inference, params, metrics = ppo.train(
        environment=env, num_timesteps=args.timesteps, num_envs=args.num_envs,
        episode_length=config.episode_length, wrap_env_fn=wrap_for_training,
        learning_rate=args.learning_rate, entropy_cost=args.entropy_cost, discounting=.99, gae_lambda=.95,
        unroll_length=32, batch_size=1024, num_minibatches=32, num_updates_per_batch=4,
        clipping_epsilon=.2, max_grad_norm=1., normalize_observations=True,
        # Some reference features are constant in the mocap (e.g. knee twist = 0);
        # a variance floor keeps unseen non-zero values (T2M output) bounded.
        normalize_observations_std_eps=1e-4,
        reward_scaling=1., network_factory=network_factory, seed=args.seed,
        num_evals=args.evals, num_eval_envs=256, deterministic_eval=True,
        progress_fn=progress, policy_params_fn=save, restore_params=restore,
        log_training_metrics=True)
    with (output / "final_params.pkl").open("wb") as f:
        pickle.dump(jax.device_get(params), f)
    if not args.more_rounds:
        (output / "done.json").write_text(json.dumps({"elapsedSeconds": time.time() - start,
                                                      "timesteps": offset + args.timesteps}))


if __name__ == "__main__":
    main()
