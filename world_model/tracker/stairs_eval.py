"""G6 stage 4: walk up a low step or a short flight of stairs and stop on top.

Scenes are ``world_model.scenes.generate_scene(4, seed, difficulty)``: difficulty < 1/3 is one
step of 6-12 cm in front of the spawn, 1/3-2/3 three steps of 9.5-11.5 cm (38 cm treads) up to
a landing, above 2/3 sixteen steps of 16-18 cm to a second floor. The reference is a captured
walk bent onto the straight route and adapted to the terrain (``terrain_reference.adapt``).
Gates: no fall, no contact with walls or railings (the step boxes are floor), and a 0.5 s stop
on the raised goal (within 0.25 m, root speed < 0.25 m/s, upright, pelvis at least 0.75 m above
the surface under it).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HOLD = 2.


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from ..scenes import generate_scene
    from .navigation_eval import _load, scene_elements
    from .path_reference import follow
    from .terrain_reference import TERRAIN, adapt, terrain_height
    run_dir, params, library, seed, difficulty, index, output = args
    run, policy, config, env, observe, bank = _load(run_dir, params, library)
    scene = generate_scene(4, seed, difficulty)
    goal = scene["goals"][-1] if scene["metadata"]["terrain"] != "stairs-and-second-floor" else \
        next(g for g in scene["goals"] if g["id"] == "stair-landing")
    height = terrain_height(scene)
    walk_name, walk = bank[index % len(bank)]
    start, target = np.asarray(scene["spawn"][:2], float), np.asarray(goal["position"][:2], float)
    flat, _ = follow(walk, np.stack([start, target]), start_direction=float(np.arctan2(*(target - start)[::-1])))
    plain = B.build_model()
    reference = adapt(plain, flat, height)["qpos"].astype(np.float64)
    reference = np.concatenate([reference, np.repeat(reference[-1:], int(HOLD * 50), axis=0)])
    length = len(reference)
    table = np.concatenate([reference, np.repeat(reference[-1:], 20, axis=0)]).astype(np.float32)

    model = B.build_model(floor=None, extra_world=scene_elements(scene))
    model.opt.iterations, model.opt.ls_iterations = config.solver_iterations, config.solver_ls_iterations
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    mujoco.mj_forward(model, data)
    nb = len(B.SEGMENTS)
    body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
    blocked = {model.geom(g["id"]).id: g["id"] for g in scene["geoms"]
               if g.get("semantic") not in TERRAIN + ("floor",)}
    info = {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(63), "linvel": jp.zeros((nb, 3))}
    jtable, jlength = jp.asarray(table), jp.int32(len(table))
    rng = jax.random.PRNGKey(0)
    contacts, fell, stopped, success_time = [], None, 0., None
    trajectory = [data.qpos.copy()]
    for frame in range(1, length):
        view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
        action = np.clip(np.asarray(policy(observe(view, info, jtable, jlength), rng)[0]), -1., 1.)
        data.ctrl[:] = table[frame, 7:] + config.action_scale * action
        prev = data.xpos[1:nb + 1].copy()
        for _ in range(B.CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:
                pair = (c.geom1, c.geom2)
                hit = [g for g in pair if g in blocked]
                if hit and any(g in body_geoms for g in pair):
                    contacts.append({"time": float(data.time), "obstacle": blocked[hit[0]],
                                     "body": model.geom(pair[0] if pair[0] in body_geoms else pair[1]).name})
        trajectory.append(data.qpos.copy())
        info = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
        up = data.xmat[1].reshape(3, 3)[2, 2]
        ground = height(data.qpos[:2])[0]
        if up < .5 or data.qpos[2] - ground < .55:
            fell = float(data.time)
            break
        if contacts:
            break
        still = (np.linalg.norm(data.qpos[:2] - target) < .25 and np.linalg.norm(data.qvel[:2]) < .25
                 and up > .8 and data.qpos[2] - ground > .75 and ground > goal["position"][2] - .02)
        stopped = stopped + .02 if still else 0.
        if stopped >= .5 and success_time is None:
            success_time = float(data.time)
    result = {"seed": seed, "difficulty": difficulty, "terrain": scene["metadata"]["terrain"],
              "rise": scene["metadata"].get("rise"), "steps": scene["metadata"].get("steps"), "walk": walk_name,
              "success": success_time is not None and not contacts and fell is None, "fellAt": fell,
              "firstContact": contacts[0] if contacts else None, "finalHeight": float(height(data.qpos[:2])[0]),
              "finalGoalDistance": float(np.linalg.norm(data.qpos[:2] - target)), "seconds": float(data.time)}
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"stairs-{seed}.npz", qpos=np.asarray(trajectory), reference=reference)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=2 / 3 - 1e-3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    difficulties = np.linspace(args.difficulty_min, args.difficulty_max, args.episodes)
    jobs = [(args.run, args.params, args.library, args.seed_base + i, float(difficulties[i]), i,
             str(Path(args.output) / "trajectories")) for i in range(args.episodes)]
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    by_terrain = {}
    for r in results:
        by_terrain.setdefault(r["terrain"], [0, 0])
        by_terrain[r["terrain"]][0] += r["success"]
        by_terrain[r["terrain"]][1] += 1
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results), "byTerrain": by_terrain,
               "falls": sum(r["fellAt"] is not None for r in results),
               "contacts": sum(r["firstContact"] is not None for r in results),
               "reference": "captured walk adapted to the terrain by IK (terrain_reference)",
               "engine": "native MuJoCo CPU", "params": args.params, "elapsedSeconds": time.time() - start}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results},
                                                               ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
