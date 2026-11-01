"""Play every training clip once on the GPU and turn failures into sampling weights.

Each clip starts at its first frame without initial noise and runs with the
deterministic policy for its whole length (capped at ``--max-seconds``). A
clip fails when the mean body position error exceeds the termination
distance, the same criterion as ``evaluate_tracker``. Finished or failed
environments restart from their first frame so their bodies keep producing an
ordinary number of contacts (the Warp contact pool is shared by the batch).

The result only chooses what the next round practises (PHC's hard-negative
mining); reported numbers always come from the CPU evaluator.

    p(clip) = (1 - hard_share) / n + hard_share * failed(clip) / n_failed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CHUNK = 50  # steps per jitted scan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory whose run.json describes the network")
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--include-seated", action="store_true")
    parser.add_argument("--include-terrain", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--hard-share", type=float, default=.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from .train_tracker import _jax_compat
    _jax_compat()
    import jax
    import jax.numpy as jp
    from .evaluate_tracker import _policy
    from .mjx_tracking import TrackingEnv, default_config, load_clips

    _, policy = _policy(args.run, args.params)
    config = default_config()
    config.include_seated = args.include_seated
    config.include_terrain = args.include_terrain
    config.split = args.split
    n = len(load_clips(args.library, split=args.split, include_seated=args.include_seated,
                       include_terrain=args.include_terrain, keys=())[1])
    # the whole batch shares one contact pool, as in training (contacts per environment)
    config.naconmax = (128 if args.include_terrain else 96 if args.include_seated else 64) * n
    env = TrackingEnv(args.library, config)
    lengths = np.asarray([c["length"] for c in env.clips])
    horizon = int(min(lengths.max() - 1, round(args.max_seconds * 50)))
    last = jp.asarray(np.minimum(lengths - 1, horizon))

    clips = jp.arange(n)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    first = jax.jit(jax.vmap(lambda k, c: env.reset_at(k, c, 0, init_noise=False)))(keys, clips)

    def where(mask, new, old):
        if old.ndim == 0 or old.shape[0] != n:
            return old   # Warp buffers shared across the batch
        return jp.where(jp.reshape(mask, (n,) + (1,) * (old.ndim - 1)), new, old)

    step = jax.vmap(env.step)

    @jax.jit
    def advance(carry, t0):
        def body(carry, t):
            state, active, failed_at, err_sum, err_max = carry
            action, _ = policy(state.obs, jax.random.PRNGKey(0))
            state = step(state, action)
            error = state.metrics["position_error"]
            failed = (state.done > 0) & active
            failed_at = jp.where(failed & (failed_at < 0), t + 1, failed_at)
            err_sum = err_sum + jp.where(active, error, 0.)
            err_max = jp.maximum(err_max, jp.where(active, error, 0.))
            active = active & ~failed & (t + 1 < last)
            state = jax.tree.map(lambda a, b: where(active, a, b), state, first)
            return (state, active, failed_at, err_sum, err_max), None
        carry, _ = jax.lax.scan(body, carry, t0 + jp.arange(CHUNK))
        return carry

    carry = (first, jp.ones(n, bool), -jp.ones(n, jp.int32), jp.zeros(n), jp.zeros(n))
    start = time.time()
    for t0 in range(0, horizon, CHUNK):
        carry = advance(carry, t0)
        if not bool(jp.any(carry[1])):
            break
    _, _, failed_at, err_sum, err_max = jax.device_get(carry)
    elapsed = time.time() - start

    steps = np.where(failed_at >= 0, failed_at, np.minimum(lengths - 1, horizon))
    failed = failed_at >= 0
    n_failed = int(failed.sum())
    share = args.hard_share if n_failed else 0.
    prob = (1 - share) / n + share * failed / max(n_failed, 1)
    clips_out = []
    for i, clip in enumerate(env.clips):
        clips_out.append({"path": clip["path"], "text": clip["text"], "needsSeat": clip["needsSeat"],
                          "frames": int(steps[i]), "failed": bool(failed[i]),
                          "failedAtSeconds": round(float(failed_at[i]) / 50, 2) if failed[i] else None,
                          "meanBodyErrorMeters": round(float(err_sum[i] / max(steps[i], 1)), 4),
                          "maxBodyErrorMeters": round(float(err_max[i]), 4)})
    by_text = {}
    for c in clips_out:
        by_text.setdefault(c["text"][:30], []).append(c["failed"])
    summary = {"run": args.run, "params": args.params, "library": args.library, "split": args.split,
               "includeSeated": args.include_seated, "clips": n, "failed": n_failed,
               "successRate": round(1 - n_failed / n, 4), "hardShare": share,
               "seatedFailed": int(sum(c["failed"] for c in clips_out if c["needsSeat"])),
               "seatedClips": int(sum(c["needsSeat"] for c in clips_out)),
               "horizonSteps": horizon, "elapsedSeconds": round(elapsed, 1), "engine": "MuJoCo Warp GPU",
               "worstTexts": sorted(((k, int(sum(v)), len(v)) for k, v in by_text.items() if any(v)),
                                    key=lambda x: (-x[1] / x[2], -x[2]))[:25]}
    weights = {c["path"]: round(float(p * n), 4) for c, p in zip(clips_out, prob)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "weights": weights, "clips": clips_out},
                                            ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
