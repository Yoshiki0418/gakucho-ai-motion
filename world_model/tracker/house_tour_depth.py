"""G5 x G6: the stage-3 house tour with a map built only from the head depth camera.

Same tour and gates as ``house_tour`` (rooms in the task order, then sit on the bedroom chair),
but the planner reads no scene geometry: only the goal poses are given (the instruction's
targets). The avatar works in stop-and-look segments, closed loop in native MuJoCo:

1. standing, take a depth image (``depth_perception`` camera: 64x48 rays, 90 degrees, 5 m) and
   add it to the map: hits 5 cm-2 m above the floor are obstacles, the floor cells the rays
   crossed are seen free (``Map``)
2. plan on that map (A*, unseen cells count as free, the widest clearance 0.6-0.4 m that works)
3. turn on the spot towards the route (captured turn), then walk along it at most 2.5 m and
   never into cells not yet seen free (0.4 m short of them); stop
4. repeat; when the next part is unseen, turn towards it to look. At the goal: stand 1.5 s.
   The sit leg ends like ``house_tour``: approach head-on, turn round, step back, sit.

Each new segment is composed from the reference's last pose and appended to the reference the
tracker is following; depth images are taken from the simulated body, so mapping errors and
tracking errors act on each other as they would on a robot.
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

import numpy as np

SEGMENT = 2.5       # m walked at most between looks
UNSEEN_BUFFER = .4  # m kept short of the first unseen cell on the route
OWN_SPACE = .6      # m around where the avatar stands counts as seen free (the head camera looks ahead)
FINISH = .5         # m: closer than this the rest of the leg is walked without another look
ARRIVED = .15       # m: already there, just stand
MAX_SEGMENTS = 60   # per tour
MAX_SECONDS = 600.


class Map:
    """Occupancy from depth hits plus the floor cells the rays have crossed (seen free)."""

    def __init__(self, low, high):
        from .depth_map import CELL, Grid
        self.grid = Grid(np.asarray(low, float), np.asarray(high, float), pad=4.)
        self.seen = np.zeros_like(self.grid.occupied)
        self.cell = CELL
        self.frame_index = None   # set by the episode: control step of the current pose (for the viewer records)
        self.shots = []           # every depth image the planner used: (frame, metres or -1 when nothing hit, obstacle pixels)

    def observe(self, model, data):
        self.stand(data.qpos[:2])
        from ..depth_perception import DepthCameraConfig, _camera_pose, _cast, _filtered_model, _pixel_rays
        import mujoco
        from .depth_map import OBSTACLE_Z
        config = DepthCameraConfig()
        _, origin, basis, root = _camera_pose(SimpleNamespace(model=model, data=data), config)
        sensor_model, query, _ = _filtered_model(model, root)
        mujoco.mj_copyData(query, model, data)
        rays = np.ascontiguousarray(_pixel_rays(config) @ basis)
        ranges, ids = _cast(sensor_model, query, origin, rays, config.far_meters, "multi_ray")
        hit = (ids >= 0) & (ranges >= 0)
        reach = np.where(hit, ranges, config.far_meters)
        points = origin + rays * reach[:, None]
        obstacle = hit & (points[:, 2] > OBSTACLE_Z[0]) & (points[:, 2] < OBSTACLE_Z[1])
        self.grid.mark(points[obstacle, :2])
        self.shots.append((self.frame_index() if self.frame_index else -1,
                           np.where(hit, ranges, -1.).reshape(config.height, config.width),
                           obstacle.reshape(config.height, config.width)))
        # every ray crosses free space up to its end (floor hits included); stop 1 cell short of a hit
        for end in points[:, :2]:
            length = np.linalg.norm(end - origin[:2])
            steps = max(int(length / self.cell), 1)
            line = origin[:2] + (end - origin[:2]) * (np.arange(steps) / steps)[:, None]
            ij = self.grid.index(line)
            ok = np.all((ij >= 0) & (ij < self.seen.shape), axis=1)
            self.seen[ij[ok, 0], ij[ok, 1]] = True
        return int(obstacle.sum())

    def plan(self, start, goal, margin):
        from .depth_map import plan
        from .house_tour import MARGINS
        for m in [m for m in MARGINS if m >= margin] or [margin]:
            try:
                return np.asarray(plan(self.grid, start, goal, m, clear=int(math.ceil(m / self.cell)))[0]), m
            except ValueError:
                continue
        raise ValueError("no route on the depth map")

    def stand(self, xy):
        """The floor around where the avatar stands is free (it is standing there)."""
        ij = self.grid.index(xy)
        r = int(OWN_SPACE / self.cell)
        ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
        keep = ii ** 2 + jj ** 2 <= r * r
        ii, jj = ii[keep] + ij[0], jj[keep] + ij[1]
        ok = (ii >= 0) & (jj >= 0) & (ii < self.seen.shape[0]) & (jj < self.seen.shape[1])
        self.seen[ii[ok], jj[ok]] = True

    def seen_until(self, route):
        """Path length along ``route`` before its first unseen cell (whole length if all seen)."""
        travelled = 0.
        for a, b in zip(route[:-1], route[1:]):
            length = float(np.linalg.norm(b - a))
            for s in np.arange(0., length, self.cell):
                ij = self.grid.index(a + (b - a) * (s / max(length, 1e-9)))
                if not self.seen[ij[0], ij[1]]:
                    return travelled + s
            travelled += length
        return travelled


def truncate(route, distance):
    """The first ``distance`` metres of a polyline."""
    out, left = [route[0]], distance
    for a, b in zip(route[:-1], route[1:]):
        length = float(np.linalg.norm(b - a))
        if length >= left:
            out.append(a + (b - a) * (left / max(length, 1e-9)))
            return np.asarray(out)
        out.append(b)
        left -= length
    return np.asarray(out)


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from ..scenes import generate_scene
    from .home_navigation import _turn_clip
    from .house_tour import (APPROACH, HOLD, SEAT_LOAD, SEATED_HOLD, CHAIR_NEAR, _forward, _sit_clips,
                             back_steps, blend_join, align_end, place_sit)
    from .navigation_eval import _load, scene_elements
    from .path_reference import follow, join, root_yaw, scale_turn, turn_then
    run_dir, params, library, seat_library, seed, difficulty, index, margin, output, sit_paths = args
    run, policy, config, env, observe, bank = _load(run_dir, params, library)
    scene = generate_scene(3, seed, difficulty)
    goals = {g["id"]: g for g in scene["goals"]}
    order = scene["metadata"]["taskSequence"]
    points = np.asarray([scene["spawn"][:2]] + [g["position"][:2] for g in scene["goals"]], float)
    world = Map(points.min(axis=0), points.max(axis=0))
    walk_name, walk = bank[index % len(bank)]
    sits = [c for c in _sit_clips(seat_library) if not sit_paths or c[0] in sit_paths]
    sit = sits[index % len(sits)]
    turn, backward, plain = _turn_clip(library), back_steps(library), B.build_model()
    model = B.build_model(floor=None, extra_world=scene_elements(scene))
    model.opt.iterations, model.opt.ls_iterations = config.solver_iterations, config.solver_ls_iterations
    nb = len(B.SEGMENTS)
    body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
    names = {model.geom(g["id"]).id: g["id"] for g in scene["geoms"] if g.get("semantic") != "floor"}
    chair_ids = {i for i, n in names.items() if n.startswith("bedroom-chair/")}
    seat_id = model.geom("bedroom-chair/seat").id
    floor_id = model.geom(next(g["id"] for g in scene["geoms"] if g.get("semantic") == "floor")).id
    feet = {s: {model.geom(f"{s}_foot").id, model.geom(f"{s}_toe").id} for s in ("left", "right")}
    weight = float(model.body_subtreemass[1]) * 9.81
    seat_xy = np.asarray(next(g for g in scene["goals"] if g["kind"] == "sit")["position"][:2])

    table_size = int(MAX_SECONDS * 50) + 100
    table = np.zeros((table_size, 70), np.float32)
    # standing at the spawn facing its heading: the first frame of the turn clip placed there
    from .path_reference import _align, _yaw_quat
    anchor = turn[0].copy()
    anchor[:2] = scene["spawn"][:2]
    anchor[3:7] = _yaw_quat(np.asarray(float(scene.get("spawnHeading", 0.))))
    reference = _align(turn[:1].copy(), anchor)
    table[:20] = reference[0]
    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    mujoco.mj_forward(model, data)
    state = {"frame": 0, "contacts": [], "fell": None, "near": False, "seat_load": 0., "seated": 0.,
             "info": {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(63), "linvel": jp.zeros((nb, 3))}}
    rng, force = jax.random.PRNGKey(0), np.zeros(6)
    trajectory = [data.qpos.copy()]   # simulated pose per control step (for the viewer records)
    world.frame_index = lambda: len(trajectory) - 1

    def append(segment, blend=True):
        nonlocal reference
        reference = blend_join(reference, segment) if blend else np.concatenate([reference, segment])
        if len(reference) + 20 > table_size:
            raise ValueError("tour too long")
        table[:len(reference)] = reference
        table[len(reference):len(reference) + 20] = reference[-1]

    def hold(frames):
        append(np.repeat(reference[-1:], frames, axis=0), blend=False)

    def simulate(goal=None, sitting=False):
        """Track the reference up to its current end; returns seconds stopped at ``goal``."""
        jtable, jlength = jp.asarray(table), jp.int32(len(reference) + 20)
        stopped, hold_from = 0., len(reference) - int(HOLD * 50)
        while state["frame"] < len(reference) - 1:
            frame = state["frame"] + 1
            view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                    "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
            action = np.clip(np.asarray(policy(observe(view, state["info"], jtable, jlength), rng)[0]), -1., 1.)
            data.ctrl[:] = table[frame, 7:] + config.action_scale * action
            prev = data.xpos[1:nb + 1].copy()
            for _ in range(B.CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
                load = 0.
                for i, c in enumerate(data.contact[:data.ncon]):
                    pair = (c.geom1, c.geom2)
                    hit = [g for g in pair if g in names]
                    if not hit or not any(g in body_geoms for g in pair):
                        continue
                    if hit[0] in chair_ids and state["near"]:
                        if hit[0] == seat_id:
                            mujoco.mj_contactForce(model, data, i, force)
                            load += abs(force[0])
                        continue
                    state["contacts"].append({"time": float(data.time), "obstacle": names[hit[0]],
                                              "body": model.geom(pair[0] if pair[0] in body_geoms else pair[1]).name})
                state["seat_load"] = load
            state["frame"] = frame
            trajectory.append(data.qpos.copy())
            state["info"] = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                             "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
            state["near"] = state["near"] or (sitting and np.linalg.norm(data.qpos[:2] - seat_xy) < CHAIR_NEAR)
            up = data.xmat[1].reshape(3, 3)[2, 2]
            if up < .5 or (not sitting and data.qpos[2] < .55):
                state["fell"] = float(data.time)
                return 0.
            if state["contacts"]:
                return 0.
            if goal is not None and frame > hold_from:
                load = {"left": 0., "right": 0.}
                for i, c in enumerate(data.contact[:data.ncon]):
                    for side, ids in feet.items():
                        if (c.geom1 == floor_id and c.geom2 in ids) or (c.geom2 == floor_id and c.geom1 in ids):
                            mujoco.mj_contactForce(model, data, i, force)
                            load[side] += abs(force[0])
                still = (np.linalg.norm(data.qpos[:2] - goal) < .25 and np.linalg.norm(data.qvel[:2]) < .25
                         and up > .8 and min(load.values()) >= 20.)
                stopped = stopped + .02 if still else 0.
            if sitting and frame > len(reference) - 1 - int(SEATED_HOLD * 50):
                on_seat = (np.linalg.norm(data.qpos[:2] - reference[frame, :2]) < .15
                           and abs(data.qpos[2] - reference[frame, 2]) < .12 and state["seat_load"] >= SEAT_LOAD * weight)
                state["seated"] = state["seated"] + .02 if on_seat else 0.
        return stopped

    def turn_towards(yaw):
        """Turn on the spot from the current pose to body yaw ``yaw``; False if already facing it."""
        angle = (yaw - root_yaw(reference[-1:])[0] + math.pi) % (2 * math.pi) - math.pi
        if abs(angle) < math.radians(20.):
            return False
        append(_align(scale_turn(turn, angle, plain), reference[-1]))
        return True

    reached, segments, looks, result_extra, trace = [], 0, 0, {}, []
    try:
        world.observe(model, data)
        looks += 1
        for goal_id in order:
            goal = goals[goal_id]
            sitting = goal["kind"] == "sit"
            if sitting:
                placed = place_sit(sit[1], sit[2], goal["position"][:2],
                                   float(next(g for g in scene["geoms"] if g["id"] == f"{goal['entityId']}/seat").get("yaw", 0.)))
                placed = join(align_end(backward, placed[0]), placed)
                facing = root_yaw(placed[:1])[0] + math.pi
                turned_sit, _ = turn_then(turn, facing, placed, plain)
                target = turned_sit[0, :2] - APPROACH * _forward(facing)
            else:
                target = np.asarray(goal["position"][:2], float)
            while True:
                segments += 1
                if segments > MAX_SEGMENTS or state["fell"] or state["contacts"]:
                    raise StopIteration
                here = reference[-1, :2]
                route, _ = world.plan(here, target, margin)
                total = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
                direction = math.atan2(*(route[1] - route[0])[::-1]) if len(route) > 1 else root_yaw(reference[-1:])[0]
                go = min(SEGMENT, world.seen_until(route) - UNSEEN_BUFFER)
                trace.append({"goal": goal_id, "here": [round(float(v), 2) for v in here],
                              "target": [round(float(v), 2) for v in target], "route": len(route),
                              "total": round(total, 2), "go": round(go, 2),
                              "yaw": round(float(root_yaw(reference[-1:])[0]), 2), "dir": round(direction, 2)})
                if total < ARRIVED:   # already there: stand and be checked
                    if sitting:
                        raise ValueError("at the turn point without a route to it")
                    hold(int(HOLD * 50))
                    if simulate(np.asarray(goal["position"][:2])) >= .5:
                        reached.append(goal_id)
                    if state["fell"] or state["contacts"]:
                        raise StopIteration
                    break
                if total <= max(go, FINISH) + 1e-6:      # the rest of the way is seen (or short): finish the leg
                    if sitting:   # walk on head-on to where the turn round starts
                        route = np.vstack([route, turned_sit[0, :2]])
                    walked, _ = follow(walk, route, start_direction=direction)
                    leg, _ = turn_then(turn, root_yaw(reference[-1:])[0], walked, plain)
                    if sitting:
                        leg = blend_join(leg, turn_then(turn, root_yaw(leg[-1:])[0], placed, plain)[0])
                        leg = np.concatenate([leg, np.repeat(leg[-1:], int(SEATED_HOLD * 50), axis=0)])
                    else:
                        leg = np.concatenate([leg, np.repeat(leg[-1:], int(HOLD * 50), axis=0)])
                    append(leg)
                    stopped = simulate(None if sitting else np.asarray(goal["position"][:2]), sitting)
                    if state["fell"] or state["contacts"]:
                        raise StopIteration
                    if not sitting and stopped >= .5:
                        reached.append(goal_id)
                    world.observe(model, data)
                    looks += 1
                    break
                if go < .3:   # the way on is unseen: turn towards the route and look
                    if turn_towards(direction + math.pi / 2):   # body yaw facing the travel direction
                        hold(25)
                        simulate()
                        world.observe(model, data)
                        looks += 1
                        continue
                    go = .5   # already facing it and still unseen: step forward a little
                part = truncate(route, go)
                walked, _ = follow(walk, part, start_direction=direction)
                leg, _ = turn_then(turn, root_yaw(reference[-1:])[0], walked, plain)
                append(leg)
                hold(25)
                simulate()
                if state["fell"] or state["contacts"]:
                    raise StopIteration
                world.observe(model, data)
                looks += 1
    except (StopIteration, ValueError) as stop:
        result_extra["stoppedBy"] = str(stop) or None
    walk_goals = [g for g in order if goals[g]["kind"] == "walk"]
    has_sit = any(goals[g]["kind"] == "sit" for g in order)
    seated = state["seated"] >= 1. if has_sit else None
    result = {"seed": seed, "difficulty": difficulty, "walk": walk_name, "sitClip": sit[0],
              "roomsReached": len(reached), "rooms": len(walk_goals), "seated": seated,
              "success": len(reached) == len(walk_goals) and seated is not False and not state["contacts"]
              and state["fell"] is None,
              "fellAt": state["fell"], "firstContact": state["contacts"][0] if state["contacts"] else None,
              "segments": segments, "looks": looks, "seconds": float(data.time),
              "mapSeenCells": int(world.seen.sum()), "mapObstacleCells": int(world.grid.occupied.sum()),
              "trace": trace, **result_extra}
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"tour-{seed}.npz", qpos=np.asarray(trajectory), reference=reference,
                            occupied=world.grid.occupied, seen=world.seen, low=world.grid.low,
                            shot_frame=np.asarray([s[0] for s in world.shots], np.int32).reshape(-1),
                            shot_depth=np.asarray([s[1] for s in world.shots], np.float16).reshape(-1, 48, 64),
                            shot_obstacle=np.asarray([s[2] for s in world.shots], bool).reshape(-1, 48, 64))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--seat-library", required=True)
    parser.add_argument("--seed-base", type=int, default=1_000_100)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=1.)
    parser.add_argument("--margin", type=float, default=.4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sit-clips", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sit_paths = tuple(p for p in args.sit_clips.split(",") if p)
    rng = np.random.default_rng(args.seed_base)
    difficulties = rng.uniform(args.difficulty_min, args.difficulty_max, args.episodes)
    jobs = [(args.run, args.params, args.library, args.seat_library, args.seed_base + i, float(difficulties[i]), i,
             args.margin, str(Path(args.output) / "trajectories"), sit_paths) for i in range(args.episodes)]
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results),
               "roomsReached": [sum(r["roomsReached"] for r in results), sum(r["rooms"] for r in results)],
               "seated": [sum(bool(r["seated"]) for r in results), sum(r["seated"] is not None for r in results)],
               "contacts": sum(r["firstContact"] is not None for r in results),
               "falls": sum(r["fellAt"] is not None for r in results),
               "stoppedBy": [r.get("stoppedBy") for r in results if r.get("stoppedBy")],
               "planner": "A* on a map from head depth images taken at each stop (no scene geometry read)",
               "seeds": [args.seed_base, args.seed_base + args.episodes - 1], "sitClips": list(sit_paths),
               "engine": "native MuJoCo CPU", "params": args.params, "elapsedSeconds": time.time() - start}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results},
                                                               ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
