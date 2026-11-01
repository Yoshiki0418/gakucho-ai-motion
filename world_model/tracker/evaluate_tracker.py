"""Evaluate a trained tracker in CPU MuJoCo, independently of the GPU trainer.

The same MJCF and solver settings are simulated with native MuJoCo (the engine
used by the existing ``world_model`` evaluations). Observations are produced
by the trainer's own observation function on CPU arrays, so the policy input
contract is identical. Every clip runs from its first frame to its end with
no resets; a clip succeeds when the mean body position error never exceeds
the termination distance (the PHC success criterion).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _policy(run_dir, params_path):
    import functools
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from .mjx_tracking import TrackingEnv
    from brax.training.acme import running_statistics
    from .train_tracker import network_kwargs
    run = json.loads((Path(run_dir) / "run.json").read_text())
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    obs_size = run["smoke"]["obsSize"]
    networks = ppo_networks.make_ppo_networks(
        obs_size, 63, preprocess_observations_fn=running_statistics.normalize,
        **network_kwargs(run["args"]))
    make = ppo_networks.make_inference_fn(networks)
    policy = jax.jit(make(params[:2], deterministic=True))
    return run, policy


SEATED = os.environ.get("TRACKER_EVAL_SEATED") == "1"


def _evaluate_clip(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from .mjx_tracking import TrackingEnv, default_config, load_clips
    global _CACHE
    run_dir, params_path, library, clip_path, max_seconds, save_dir = args
    key = (run_dir, params_path, library)
    if "_CACHE" not in globals():
        _CACHE = {}
    if key not in _CACHE:
        run, policy = _policy(run_dir, params_path)
        config = default_config()
        for name, value in run["config"].items():
            if name in config and not isinstance(value, dict):
                config[name] = tuple(value) if isinstance(value, list) else value
        config.impl = "jax"
        config.split = "all"
        config.include_seated = SEATED
        config.clip_weights = ""   # training-time clip sampling only
        env = TrackingEnv(library, config)
        model = env.mj_model          # same MJCF and solver settings as training
        _, _, ref = load_clips(library, split="all", include_seated=SEATED,
                               keys=("qpos", "qvel", "xpos"))
        tables = (env._qpos, env._starts, env._lengths)

        def observe(view, info, table, starts, lengths):
            # the library is an argument, not a captured constant (slow CPU compiles)
            saved = env._qpos, env._starts, env._lengths
            env._qpos, env._starts, env._lengths = table, starts, lengths
            try:
                return env._observation(SimpleNamespace(**view), info)
            finally:   # never leave traced arrays on the env
                env._qpos, env._starts, env._lengths = saved

        compiled = jax.jit(observe)
        obs_fn = lambda view, info: compiled(view, info, *tables)
        _CACHE[key] = (run, policy, config, env, model, ref, obs_fn)
    run, policy, config, env, model, ref, obs_fn = _CACHE[key]
    index = next(i for i, c in enumerate(env.clips) if c["path"] == clip_path)
    clip = env.clips[index]
    start, length = clip["start"], clip["length"]
    data = mujoco.MjData(model)
    data.qpos[:] = ref["qpos"][start]
    data.qvel[:] = ref["qvel"][start]
    if SEATED:
        from .mjx_tracking import seat_pose
        data.mocap_pos[0], data.mocap_quat[0] = seat_pose(clip["seat"])
    mujoco.mj_forward(model, data)
    nb = len(B.SEGMENTS)
    info = {"clip": jp.int32(index), "frame": jp.int32(0), "last_act": jp.zeros(63),
            "linvel": env._ref_linvel(jp.int32(index), jp.int32(0))}
    errors, rng = [], jax.random.PRNGKey(0)
    trajectory = [data.qpos.copy()]
    steps = min(length - 1, int(max_seconds * 50))
    failed_at = None
    for frame in range(1, steps + 1):
        view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
        obs = obs_fn(view, info)
        action, _ = policy(obs, rng)
        action = np.clip(np.asarray(action), -1., 1.)
        row = start + frame
        data.ctrl[:] = ref["qpos"][row, 7:] + config.action_scale * action
        prev = data.xpos[1:nb + 1].copy()
        for _ in range(B.CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
        err = np.linalg.norm(data.xpos[1:nb + 1] - ref["xpos"][row], axis=-1)
        errors.append(err)
        trajectory.append(data.qpos.copy())
        info = {"clip": jp.int32(index), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
        if err.mean() > config.termination_distance or not np.isfinite(data.qpos).all():
            failed_at = frame
            break
    errors = np.asarray(errors)
    if save_dir:
        name = clip_path.replace("/", "_").replace("#", "_")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(save_dir) / f"{name}.npz", qpos=np.asarray(trajectory),
                            reference=ref["qpos"][start:start + len(trajectory)])
    return {"path": clip_path, "label": clip["label"], "dataset": clip["dataset"], "text": clip["text"],
            "needsSeat": clip["needsSeat"], "frames": int(steps), "success": failed_at is None,
            "failedAtSeconds": None if failed_at is None else failed_at / 50.,
            "meanBodyErrorMeters": float(errors.mean()) if len(errors) else None,
            "maxMeanBodyErrorMeters": float(errors.mean(axis=1).max()) if len(errors) else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-seated", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-trajectories", default="", help="directory for simulated qpos per clip")
    parser.add_argument("--paths-from", default="",
                        help="evaluate exactly the clips of an earlier result JSON (e.g. after relabeling seats)")
    args = parser.parse_args()
    from .mjx_tracking import load_clips
    _, clips, _ = load_clips(args.library, split=args.split, include_seated=args.include_seated)
    if args.paths_from:
        wanted = [c["path"] for c in json.loads(Path(args.paths_from).read_text())["clips"]]
        by_path = {c["path"]: c for c in clips}
        missing = [p for p in wanted if p not in by_path]
        if missing:
            raise SystemExit(f"{len(missing)} clips are not in {args.library} ({args.split}): {missing[:3]}")
        clips = [by_path[p] for p in wanted]
    elif args.limit:
        clips = clips[::max(1, len(clips) // args.limit)][:args.limit]
    # workers read this at import time (spawned processes inherit the environment)
    os.environ["TRACKER_EVAL_SEATED"] = "1" if args.include_seated else "0"
    start = time.time()
    jobs = [(args.run, args.params, args.library, c["path"], args.max_seconds, args.save_trajectories)
            for c in clips]
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_evaluate_clip, jobs, chunksize=1)
    success = [r["success"] for r in results]
    summary = {"run": args.run, "params": args.params,
               "paramsSha256": hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
               "split": args.split, "includeSeated": args.include_seated, "clips": len(results),
               "successes": int(sum(success)), "successRate": float(np.mean(success)),
               "meanBodyErrorMeters": float(np.mean([r["meanBodyErrorMeters"] for r in results
                                                     if r["meanBodyErrorMeters"] is not None])),
               "engine": "native MuJoCo CPU", "elapsedSeconds": time.time() - start}
    by_label = {}
    for r in results:
        by_label.setdefault(f'{r["dataset"]}/{r["label"]} {r["text"][:24]}', []).append(r["success"])
    summary["worstLabels"] = sorted(((k, sum(v), len(v)) for k, v in by_label.items()),
                                    key=lambda x: x[1] / x[2])[:20]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "clips": results},
                                            ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
