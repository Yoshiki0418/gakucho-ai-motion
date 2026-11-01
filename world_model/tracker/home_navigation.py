"""G6 (first part): walk to named places in the displayed home with the tracker.

The home is ``world_model.home_scene``: the house of the home screen (10 m
room, kitchen, dining table with four chairs, sofa; 70 contact boxes). Routes
run from the spawn to each walk goal (キッチン・ダイニング・リビング) and between
the walk goals, each with several captured walks.

Planner ``oracle``: A* (``depth_map.plan``) on an occupancy grid rasterised
from the true boxes between 5 cm and 2 m above the floor (a box reaching into
that band blocks its footprint, so legs, tabletops and wall cabinets count).
The route is then tracked exactly as in ``navigation_eval.walk_route``, with
the same gates (no contact with any box, no fall, a 0.5 s stop within 0.25 m
of the goal).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .depth_map import CELL, OBSTACLE_Z, Grid, plan


def rasterise(grid, scene):
    """Mark every cell whose centre lies in a box footprint (grown by half a cell)."""
    centres = grid.centre(np.stack(np.meshgrid(np.arange(grid.occupied.shape[0]),
                                               np.arange(grid.occupied.shape[1]), indexing="ij"), -1))
    for g in scene["geoms"]:
        if g.get("semantic") == "floor":
            continue
        bottom, top = g["position"][2] - g["size"][2] / 2, g["position"][2] + g["size"][2] / 2
        if top < OBSTACLE_Z[0] or bottom > OBSTACLE_Z[1]:
            continue
        c, s = math.cos(g.get("yaw", 0.)), math.sin(g.get("yaw", 0.))
        d = centres - np.asarray(g["position"][:2])
        local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)
        half = np.asarray(g["size"][:2]) / 2 + CELL / 2
        grid.occupied |= np.all(np.abs(local) <= half, axis=-1)


# The kitchen and dining places stand 0.35 m from the counter / table edge, facing it; a hand
# then touches the top while stopping or turning. The avatar stops (and later starts) this much
# further back along the place's facing; arrival is still judged at the place (0.25 m).
STOP_BACK = .12


def stand_point(goal):
    heading = float(goal.get("heading", 0.))
    return np.asarray(goal["position"][:2], float) - STOP_BACK * np.asarray([math.sin(heading), -math.cos(heading)])


def routes(scene):
    """(name, start xy, start scene-heading, goal) for spawn->goal and goal->goal."""
    walks = [g for g in scene["goals"] if g["kind"] == "walk"]
    starts = [("spawn", scene["spawn"][:2], float(scene.get("spawnHeading", 0.)))]
    starts += [(g["id"], stand_point(g), float(g["heading"])) for g in walks]
    return [(f"{name}->{goal['id']}", np.asarray(xy, float), heading, goal)
            for name, xy, heading in starts for goal in walks if goal["id"] != name]


TURN_CLIP = "0008/02/001/001/005.bvh"   # 「前から後ろに振り向く」: 149 degrees right on the spot, 6 cm drift


BACK_CLIP = "0003/01/001/001/015.bvh"   # 「後ろ向きに歩く」: 0.5 m back in 1.5 s
STEP_BACK = .45                          # m; leave the counter or the table before turning


def _clip(library, path, key="qpos"):
    manifest = json.loads((Path(library) / "manifest.json").read_text())
    clip = next(c for c in manifest["clips"] if c["path"] == path)
    data = np.load(Path(library) / "library.npz")[key]
    return data[clip["start"]:clip["start"] + clip["length"]].astype(np.float64)


def _turn_clip(library):
    return _clip(library, TURN_CLIP)


def _step_back(library, start, heading):
    """The first steps of a captured backward walk, placed at ``start`` facing ``heading``,
    cut at the first double-support frame once ``STEP_BACK`` metres are covered."""
    from .path_reference import _align, _yaw_quat, root_yaw
    qpos, xpos = _clip(library, BACK_CLIP), _clip(library, BACK_CLIP, "xpos")
    forward = lambda yaw: np.asarray([np.sin(yaw), -np.cos(yaw)])
    back = (qpos[:, :2] - qpos[0, :2]) @ forward(root_yaw(qpos[:1])[0])
    feet = np.maximum(xpos[:, 16, 2], xpos[:, 20, 2])          # higher of the two feet
    after = np.nonzero(back <= -STEP_BACK)[0]
    cut = next((t for t in after if feet[t] < feet[0] + .03), after[0] if len(after) else len(qpos) - 1)
    anchor = qpos[:1].copy()
    anchor[0, :2] = start
    anchor[0, 3:7] = _yaw_quat(np.asarray(heading))
    return _align(qpos[:cut + 1], anchor[0])


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from ..home_scene import create_home_scene
    from .navigation_eval import _load, walk_route
    run_dir, params, library, route_index, walk_index, margin, output, facing = args
    cache = _load(run_dir, params, library)
    bank = cache[5]
    scene = create_home_scene()
    name, start, heading, goal = routes(scene)[route_index]
    scene = dict(scene, spawn=[*start, 0.], spawnHeading=heading, goals=[goal])
    walk_name, walk = bank[walk_index % len(bank)]
    goal_xy = stand_point(goal)
    grid = Grid(start, goal_xy)
    rasterise(grid, scene)
    try:
        # the goals stand next to furniture: keep a margin-sized disc around start and goal free
        route = np.asarray(plan(grid, start, goal_xy, margin, clear=int(np.ceil(margin / CELL)))[0],
                           dtype=np.float64)
    except ValueError:
        return {"route": name, "goal": goal["label"], "walk": walk_name, "success": False, "planFailed": True}
    reference = None
    if facing == "back-turn":
        # step back from the furniture, turn on the spot, then walk a route planned from there
        from . import body as B
        from .path_reference import follow, join, root_yaw, turn_then
        back = _step_back(library, start, heading)
        try:
            route = np.asarray(plan(grid, back[-1, :2], goal_xy, margin, clear=int(np.ceil(margin / CELL)))[0],
                               dtype=np.float64)
        except ValueError:
            return {"route": name, "goal": goal["label"], "walk": walk_name, "success": False, "planFailed": True}
        start_direction = math.atan2(*(route[1] - route[0])[::-1])
        walk_reference, _ = follow(walk, route, start_direction=start_direction)
        turned, _ = turn_then(_turn_clip(library), root_yaw(back[-1:])[0], walk_reference, B.build_model())
        reference = join(back, turned)
    elif facing == "route":
        # already turned towards the first leg
        start_direction = math.atan2(*(route[1] - route[0])[::-1])
    elif facing == "turn":
        # stand at the place's heading, turn on the spot (a captured turn, scaled), then walk the route
        from . import body as B
        from .path_reference import follow, turn_then
        start_direction = math.atan2(*(route[1] - route[0])[::-1])
        walk_reference, _ = follow(walk, route, start_direction=start_direction)
        reference, _ = turn_then(_turn_clip(library), heading, walk_reference, B.build_model())
    else:
        start_direction = math.atan2(-math.cos(heading), math.sin(heading))
    gates, trajectory, reference = walk_route(cache, scene, route, walk, start_direction, goal["position"],
                                              reference=reference)
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name.replace('>', '')}-{walk_index}.npz", qpos=trajectory,
                            reference=reference, route=route)
    return {"route": name, "goal": goal["label"], "walk": walk_name, "routeMargin": margin, "facing": facing,
            "waypoints": len(route), **gates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--walks", type=int, default=4, help="captured walks per route")
    parser.add_argument("--margin", type=float, default=.4)
    parser.add_argument("--facing", default="scene", choices=("scene", "route", "turn", "back-turn"),
                        help="start heading: the place's heading, or already facing the first leg of the route")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from ..home_scene import create_home_scene
    names = [r[0] for r in routes(create_home_scene())]
    jobs = [(args.run, args.params, args.library, r, r * args.walks + w, args.margin,
             str(Path(args.output) / "trajectories"), args.facing)
            for r in range(len(names)) for w in range(args.walks)]
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    by_route = {n: [sum(r["success"] for r in results if r["route"] == n),
                    sum(r["route"] == n for r in results)] for n in names}
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results),
               "contacts": sum(r.get("firstContact") is not None for r in results),
               "falls": sum(r.get("fellAt") is not None for r in results),
               "planFailed": sum(bool(r.get("planFailed")) for r in results), "byRoute": by_route,
               "planner": "A* on an occupancy grid of the true home boxes (privileged)",
               "margin": args.margin, "facing": args.facing, "engine": "native MuJoCo CPU", "params": args.params,
               "elapsedSeconds": time.time() - start}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results},
                                                               ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
