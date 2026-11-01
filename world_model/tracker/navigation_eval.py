"""Phase-2 furniture avoidance with planner + bent-walk reference + tracker.

Scenes come from the same generator as the existing phase-2 evaluation
(``navigation_scenes.generate_navigation_scene``). This evaluation is a
development pipeline, kept separate from the earlier R4/5aa results:

1. Route: privileged visibility route around the inflated furniture
   (``visual_teacher.oracle_route``; full scene geometry, no vision yet).
2. Reference: a captured walk, cycle-extended and bent onto the route,
   starting from the spawn heading (``path_reference.follow``).
3. Control: the trained tracker in native MuJoCo (CPU) with the scene
   geometry. Root state is set only at reset.

Gates (per episode): no body contact with walls or furniture at any 5 ms
physics step, no fall, and at the end a sustained 0.5 s stop within 0.25 m
of the goal (root speed < 0.25 m/s, both feet loaded >= 20 N, upright).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np

MAX_FRAMES = 2000          # padded reference length (40 s at 50 Hz)
HOLD_SECONDS = 2.          # keep the final reference pose after the walk


def scene_elements(scene):
    from .body import WORLD_CONAFFINITY, WORLD_CONTYPE
    elements = []
    for g in scene["geoms"]:
        if g.get("type", "box") != "box":
            raise ValueError("only box scene geoms are supported")
        attrs = {"name": g["id"], "type": "box", "size": " ".join(f"{v / 2:.6g}" for v in g["size"]),
                 "pos": " ".join(f"{v:.6g}" for v in g["position"]),
                 "euler": f"0 0 {float(g.get('yaw', 0.)):.6g}",
                 "contype": str(WORLD_CONTYPE), "conaffinity": str(WORLD_CONAFFINITY),
                 "rgba": "0.55 0.5 0.45 1"}
        if "friction" in g:
            f = g["friction"]
            f = [f, .005, .0001] if isinstance(f, (int, float)) else f
            attrs["friction"] = " ".join(f"{v:.6g}" for v in f)
        elements.append(ET.Element("geom", **attrs))
    return elements


def _walk_bank(library):
    """Captured forward walks from the training split (never held-out takes)."""
    manifest = json.loads((Path(library) / "manifest.json").read_text())
    qpos = np.load(Path(library) / "library.npz")["qpos"]
    clips = [c for c in manifest["clips"] if c["status"] == "ok" and c["split"] == "train"
             and c["dataset"] == "0001" and c["label"] == "001" and "augmentation" not in c]
    return [(c["path"], qpos[c["start"]:c["start"] + c["length"]].astype(np.float64)) for c in clips]


def _load(run_dir, params, library):
    """Policy, observation function and walk bank, built once per worker process."""
    global _CACHE
    if "_CACHE" not in globals():
        import jax
        import jax.numpy as jp
        from .evaluate_tracker import _policy
        from .mjx_tracking import TrackingEnv, default_config
        run, policy = _policy(run_dir, params)
        config = default_config()
        for key, value in run["config"].items():
            if key in config and not isinstance(value, dict):
                config[key] = tuple(value) if isinstance(value, list) else value
        config.impl, config.split = "jax", "all"
        config.clip_weights = ""   # training-time clip sampling only
        env = TrackingEnv(library, config)

        def observe(view, info, table, length):
            saved = env._qpos, env._starts, env._lengths
            env._qpos, env._starts, env._lengths = table, jp.zeros(1, jp.int32), length[None]
            try:
                return env._observation(SimpleNamespace(**view), info)
            finally:   # never leave traced arrays on the env
                env._qpos, env._starts, env._lengths = saved

        _CACHE = (run, policy, config, env, jax.jit(observe), _walk_bank(library))
    return _CACHE


def walk_route(cache, scene, route, walk, start_direction, goal, zero=False, reference=None):
    """Bend ``walk`` onto ``route``, track it in ``scene`` (native MuJoCo) and apply the gates.

    ``reference`` (qpos) replaces the bent walk, e.g. a turn on the spot followed by it.
    Returns the gate result and the simulated trajectory with its reference.
    """
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from .path_reference import follow
    run, policy, config, env, observe, bank = cache
    stride_scale = float("nan")
    if reference is None:
        reference, stride_scale = follow(walk, route, start_direction=start_direction)
    walk_frames = len(reference)
    hold = np.repeat(reference[-1:], int(HOLD_SECONDS * 50), axis=0)
    reference = np.concatenate([reference, hold])
    length = len(reference)
    if length > MAX_FRAMES:
        raise ValueError("reference longer than the padded table")
    table = np.zeros((MAX_FRAMES, reference.shape[1]), dtype=np.float32)
    table[:length], table[length:] = reference, reference[-1]

    model = B.build_model(floor=None, extra_world=scene_elements(scene))
    model.opt.iterations, model.opt.ls_iterations = config.solver_iterations, config.solver_ls_iterations
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    nb = len(B.SEGMENTS)
    body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
    obstacle_ids = {model.geom(g["id"]).id: g["id"] for g in scene["geoms"] if g.get("semantic") != "floor"}
    floor_id = model.geom(next(g["id"] for g in scene["geoms"] if g.get("semantic") == "floor")).id
    feet = {side: {model.geom(f"{side}_foot").id, model.geom(f"{side}_toe").id} for side in ("left", "right")}
    goal = np.asarray(goal[:2])

    info = {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(63), "linvel": jp.zeros((nb, 3))}
    jtable, jlength = jp.asarray(table), jp.int32(length)
    rng = jax.random.PRNGKey(0)
    contacts, fell, stop_time, success_time = [], None, 0., None
    trajectory, force = [data.qpos.copy()], np.zeros(6)
    for frame in range(1, length):
        view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
        action = (np.zeros(63, dtype=np.float32) if zero else
                  np.clip(np.asarray(policy(observe(view, info, jtable, jlength), rng)[0]), -1., 1.))
        data.ctrl[:] = table[frame, 7:] + config.action_scale * action
        prev = data.xpos[1:nb + 1].copy()
        for _ in range(B.CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:
                pair = (c.geom1, c.geom2)
                hit = [g for g in pair if g in obstacle_ids]
                if hit and any(g in body_geoms for g in pair):
                    contacts.append({"time": float(data.time), "obstacle": obstacle_ids[hit[0]],
                                     "body": model.geom(pair[0] if pair[0] in body_geoms else pair[1]).name})
        trajectory.append(data.qpos.copy())
        info = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
        up = data.xmat[1].reshape(3, 3)[2, 2]
        if data.qpos[2] < .55 or up < .5:
            fell = float(data.time)
            break
        if contacts:
            break
        # foot support from floor contacts
        load = {"left": 0., "right": 0.}
        for i, c in enumerate(data.contact[:data.ncon]):
            for side, ids in feet.items():
                if (c.geom1 == floor_id and c.geom2 in ids) or (c.geom2 == floor_id and c.geom1 in ids):
                    mujoco.mj_contactForce(model, data, i, force)
                    load[side] += abs(force[0])
        stopped = (np.linalg.norm(data.qpos[:2] - goal) < .25 and np.linalg.norm(data.qvel[:2]) < .25
                   and up > .8 and min(load.values()) >= 20.)
        stop_time = stop_time + .02 if stopped else 0.
        if stop_time >= .5 and success_time is None:
            success_time = float(data.time)
    result = {"routeLength": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
              "strideScale": float(stride_scale), "referenceSeconds": walk_frames / 50.,
              "success": success_time is not None and not contacts and fell is None,
              "successTime": success_time, "fellAt": fell,
              "firstContact": contacts[0] if contacts else None,
              "finalGoalDistance": float(np.linalg.norm(data.qpos[:2] - goal)), "seconds": float(data.time)}
    return result, np.asarray(trajectory), reference


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import mujoco
    from . import body as B
    from ..navigation_scenes import generate_navigation_scene
    from .path_reference import follow
    from ..visual_teacher import oracle_route
    run_dir, params, library, scene_seed, difficulty, walk_index, margin, output, zero, planner = args
    cache = _load(run_dir, params, library)
    bank = cache[5]

    scene = generate_navigation_scene(scene_seed, mode="obstacle", difficulty=difficulty)
    walk_name, walk = bank[walk_index % len(bank)]
    heading = float(scene.get("spawnHeading", 0.))
    start_direction = math.atan2(-math.cos(heading), math.sin(heading))
    perception = None
    if planner == "depth":
        # G5: the route comes from one head depth image taken in the starting pose (depth_map.Grid:
        # hits 5 cm-2 m above the floor become occupied cells, unseen space is free); no scene geometry is read.
        from ..scenes import point_box_distance
        from .depth_map import Grid, observe as look_at, plan
        start_xy = np.asarray(scene["spawn"][:2], dtype=np.float64)
        goal_xy = np.asarray(scene["goals"][0]["position"][:2], dtype=np.float64)
        start_pose = follow(walk, np.stack([start_xy, goal_xy]), start_direction=start_direction)[0][0]
        look = B.build_model(floor=None, extra_world=scene_elements(scene))
        look_data = mujoco.MjData(look)
        look_data.qpos[:] = start_pose
        mujoco.mj_forward(look, look_data)
        grid = Grid(start_xy, goal_xy)
        hits = look_at(look, look_data, grid)
        try:
            route = np.asarray(plan(grid, start_xy, goal_xy, margin)[0], dtype=np.float64)
        except ValueError:
            route = None
        # evaluation only: how close the planned route comes to the true furniture (never used for planning)
        truth = next(g for g in scene["geoms"] if g.get("entityId") == "navigation-furniture")
        clearance = None
        if route is not None:
            samples = [a + (b - a) * t for a, b in zip(route[:-1], route[1:]) for t in np.linspace(0, 1, 50)]
            clearance = float(min(point_box_distance((*p, 0.), truth) for p in samples))
        perception = {"obstacleHits": hits, "routePlanned": route is not None,
                      "routeWaypoints": None if route is None else len(route),
                      "routeClearanceToTrueFurnitureMeters": clearance}
        if route is None:
            return {"sceneSeed": scene_seed, "difficulty": difficulty, "walk": walk_name, "planner": planner,
                    "perception": perception, "success": False, "planFailed": True, "successTime": None,
                    "fellAt": None, "firstContact": None, "finalGoalDistance": None, "seconds": 0.,
                    "passSide": "none"}
    else:
        route = np.asarray(oracle_route(scene, margin), dtype=np.float64)
    goal = scene["goals"][0]["position"]
    gates, trajectory, reference = walk_route(cache, scene, route, walk, start_direction, goal, zero)
    result = {"sceneSeed": scene_seed, "difficulty": difficulty, "walk": walk_name, "planner": planner,
              "perception": perception, "routeMargin": margin, **gates,
              "passSide": "left(+X)" if route[1][0] > [g for g in scene["geoms"]
                                                      if g.get("entityId") == "navigation-furniture"][0]["position"][0]
                          else "right(-X)"}
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"scene-{scene_seed}.npz", qpos=trajectory, reference=reference, route=route)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--scene-seed-base", type=int, default=1_300_000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.5)
    parser.add_argument("--margin", type=float, default=.5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", default="",
                        help="JSON list of {sceneSeed, difficulty} (e.g. the Codex new-32 cohort)")
    parser.add_argument("--planner", default="oracle", choices=("oracle", "depth"),
                        help="oracle: true furniture geometry; depth: furniture estimated from the head depth image")
    parser.add_argument("--zero-policy", action="store_true",
                        help="baseline: zero residual, i.e. PD tracking of the reference only")
    args = parser.parse_args()
    if args.scenes:
        cohort = [(e["sceneSeed"], e["difficulty"]) for e in json.loads(Path(args.scenes).read_text())]
    else:
        rng = np.random.default_rng(args.scene_seed_base)
        difficulties = rng.uniform(args.difficulty_min, args.difficulty_max, args.episodes)
        cohort = [(args.scene_seed_base + i, float(difficulties[i])) for i in range(args.episodes)]
    jobs = [(args.run, args.params, args.library, seed, difficulty, i, args.margin,
             str(Path(args.output) / "trajectories"), args.zero_policy, args.planner)
            for i, (seed, difficulty) in enumerate(cohort)]
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results),
               "contacts": sum(r["firstContact"] is not None for r in results),
               "falls": sum(r["fellAt"] is not None for r in results),
               "bySide": {side: [sum(r["success"] for r in results if r["passSide"] == side),
                                 sum(r["passSide"] == side for r in results)]
                          for side in ("left(+X)", "right(-X)")},
               "sceneSeeds": [cohort[0][0], cohort[-1][0]], "cohortFile": args.scenes or None,
               "difficultyRange": [args.difficulty_min, args.difficulty_max],
               "planner": ("A* on an occupancy grid from one head depth image at the spawn"
                           " (no scene geometry read)" if args.planner == "depth"
                           else "privileged visibility route (full geometry)"),
               "engine": "native MuJoCo CPU",
               "controller": "zero residual (reference PD only)" if args.zero_policy else "trained tracker",
               "params": args.params, "elapsedSeconds": time.time() - start}
    seen = [r["perception"] for r in results if r.get("perception")]
    if seen:
        planned = [p for p in seen if p["routePlanned"]]
        clearances = [p["routeClearanceToTrueFurnitureMeters"] for p in planned]
        summary["perception"] = {
            "routePlanned": [len(planned), len(seen)],
            "furnitureSeen": sum(p["obstacleHits"] > 0 for p in seen),
            "clearanceMinMeters": min(clearances) if clearances else None,
            "clearanceMedianMeters": float(np.median(clearances)) if clearances else None}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results},
                                                               ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
