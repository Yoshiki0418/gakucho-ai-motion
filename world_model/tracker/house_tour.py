"""G6: the stage-3 house — walk from room to room through the doors, then sit on the chair.

Scenes are ``world_model.scenes.generate_scene(3, seed, difficulty)``: four rooms off a 2 m
corridor, doors 1.05-1.6 m wide, a fixture in each room and a free-standing chair in the
bed room (held-out seeds start at 1,000,000). One episode is the scene's task sequence,
work-room -> kitchen -> bed-room -> living-room -> sit on the bedroom chair, tracked as ONE
continuous reference in native MuJoCo (no reset between legs):

* walk legs: A* on the true boxes (``home_navigation.rasterise``, privileged map), a captured
  turn on the spot towards the route, a captured walk bent onto it, then 1.5 s standing.
* the sit leg: a captured 「立っている状態から、椅子に座る。」 is rotated and moved so its
  estimated seat (``seats.json``) lands on the chair seat; before it come a captured backward
  walk (0.45 m) and a captured turn (scaled to the needed angle), so the body turns round clear
  of the chair legs and steps back to the chair; the walk approaches the turn's start head-on.

Gates: no body contact with walls, partitions, fixtures or the chair (the target chair may be
touched once the pelvis is within 1 m of its seat on the last leg, as when people sit down), no fall; after each walk leg a 0.5 s stop within 0.25 m of the
room goal; at the end the pelvis rests on the seat for 1 s (within 0.15 m / 0.12 m of the
reference pelvis, seat contact carrying at least 30% of the body weight).
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

SIT_TEXT = "立っている状態から、椅子に座る。"
HOLD = 1.5          # s standing at each room goal
SEATED_HOLD = 2.    # s seated at the end
APPROACH = .6       # m walked straight towards the chair before turning round
MARGINS = (.6, .55, .5, .45, .4)   # each leg takes the widest clearance that still has a route
CHAIR_NEAR = 1.     # m; the target chair may be touched once the pelvis is this close to its seat
SEAT_LOAD = .3      # share of body weight carried by the seat
SIT_START = .4      # m from the seat centre at least: closer starts put the feet among the chair legs


def _sit_clips(library):
    """(path, qpos, seat) of captured sits whose seat height matches a 0.46 m chair."""
    manifest = json.loads((Path(library) / "manifest.json").read_text())
    seats = json.loads((Path(library) / "seats.json").read_text())["seats"]
    qpos = np.load(Path(library) / "library.npz")["qpos"]
    start = lambda c: np.linalg.norm(qpos[c["start"], :2] - [seats[c["path"]]["x"], seats[c["path"]]["y"]])
    clips = [c for c in manifest["clips"] if c["status"] == "ok" and c["text"] == SIT_TEXT
             and c["path"] in seats and abs(seats[c["path"]]["top"] - .46) < .02 and start(c) >= SIT_START]
    clips.sort(key=lambda c: (c["split"] != "test", c["path"]))   # held-out takes first
    return [(c["path"], qpos[c["start"]:c["start"] + c["length"]].astype(np.float64), seats[c["path"]])
            for c in clips]


def place_sit(qpos, seat, chair_xy, chair_yaw):
    """Rotate/move a sit clip so its estimated seat centre and seated heading match the chair."""
    from .path_reference import _quat_mul, _yaw_quat
    delta = chair_yaw - seat["yaw"]
    c, s = math.cos(delta), math.sin(delta)
    rel = qpos[:, :2] - np.asarray([seat["x"], seat["y"]])
    out = qpos.copy()
    out[:, :2] = np.asarray(chair_xy) + np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
    out[:, 3:7] = _quat_mul(_yaw_quat(np.full(len(out), delta)), qpos[:, 3:7])
    return out


def align_end(segment, target):
    """Rotate/move ``segment`` (qpos) so its LAST root pose equals ``target``'s in position and yaw."""
    from .path_reference import _quat_mul, _yaw_quat, root_yaw
    delta = root_yaw(target[None])[0] - root_yaw(segment[-1:])[0]
    c, s = math.cos(delta), math.sin(delta)
    rel = segment[:, :2] - segment[-1, :2]
    out = segment.copy()
    out[:, :2] = target[:2] + np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
    out[:, 3:7] = _quat_mul(_yaw_quat(np.full(len(out), delta)), segment[:, 3:7])
    return out


def back_steps(library):
    """A captured backward walk cut at its first double-support frame after ``STEP_BACK`` metres."""
    from .home_navigation import BACK_CLIP, STEP_BACK, _clip
    from .path_reference import root_yaw
    qpos, xpos = _clip(library, BACK_CLIP), _clip(library, BACK_CLIP, "xpos")
    back = (qpos[:, :2] - qpos[0, :2]) @ _forward(root_yaw(qpos[:1])[0])
    feet = np.maximum(xpos[:, 16, 2], xpos[:, 20, 2])
    after = np.nonzero(back <= -STEP_BACK)[0]
    cut = next((t for t in after if feet[t] < feet[0] + .03), after[0])
    return qpos[:cut + 1]


def blend_join(first, second, blend=20):
    """``second`` after ``first`` WITHOUT moving it: the first ``blend`` frames fade from ``first``'s
    last pose (root position, rotation and joints), so world-fixed parts (the sit on the chair,
    the planned routes) stay where they were placed. Use when the two already nearly meet."""
    from scipy.spatial.transform import Rotation, Slerp
    w = np.clip(np.arange(1, len(second) + 1) / blend, 0., 1.)
    w = w * w * (3 - 2 * w)
    out = second.copy()
    out[:, :3] = (1 - w[:, None]) * first[-1, :3] + w[:, None] * second[:, :3]
    out[:, 7:] = (1 - w[:, None]) * first[-1, 7:] + w[:, None] * second[:, 7:]
    n = min(blend, len(second))
    for t in range(n):
        pair = Rotation.from_quat([first[-1, [4, 5, 6, 3]], second[t, [4, 5, 6, 3]]])
        q = Slerp([0., 1.], pair)([w[t]]).as_quat()[0]
        out[t, 3:7] = q[[3, 0, 1, 2]]
    return np.concatenate([first, out[1:]])


def _forward(yaw):
    return np.asarray([math.sin(yaw), -math.cos(yaw)])


def build_tour(scene, walk, turn, sit, grid, margin, model, backward=None):
    """One continuous reference for the whole tour and the frame windows of each leg."""
    from .depth_map import CELL, plan
    from .path_reference import follow, join, root_yaw, turn_then
    hold = int(HOLD * 50)
    goals = {g["id"]: g for g in scene["goals"]}
    order = scene["metadata"]["taskSequence"]
    position = np.asarray(scene["spawn"][:2], float)
    reference, legs = None, []
    heading = float(scene.get("spawnHeading", 0.))
    def widest(a, b):
        # at the widest feasible clearance the route runs through the middle of a narrow door
        for m in [m for m in MARGINS if m >= margin] or [margin]:
            try:
                return np.asarray(plan(grid, a, b, m, clear=int(math.ceil(m / CELL)))[0]), m
            except ValueError:
                continue
        raise ValueError("no route")

    for goal_id in order:
        goal = goals[goal_id]
        if goal["kind"] == "walk":
            target = np.asarray(goal["position"][:2], float)
            route, used = widest(position, target)
        else:   # sit: walk head-on to where the turn round starts
            sit_path, sit_qpos, seat = sit
            chair = next(g for g in scene["geoms"] if g["id"] == f"{goal['entityId']}/seat")
            placed = place_sit(sit_qpos, seat, goal["position"][:2], float(chair.get("yaw", 0.)))
            facing_chair = root_yaw(placed[:1])[0] + math.pi
            if backward is not None:
                # turn round clear of the chair legs, then step back to where the sit starts
                placed = join(align_end(backward, placed[0]), placed)
            turned_sit, _ = turn_then(turn, facing_chair, placed, model)
            start_turn = turned_sit[0, :2]
            before = start_turn - APPROACH * _forward(facing_chair)
            route, used = widest(position, before)
            route = np.vstack([route, start_turn])
        direction = math.atan2(*(route[1] - route[0])[::-1])
        walked, _ = follow(walk, route, start_direction=direction)
        # the tour starts standing at the spawn facing its heading; later legs where the last one ended
        before_yaw = heading if reference is None else root_yaw(reference[-1:])[0]
        leg, _ = turn_then(turn, before_yaw, walked, model)
        if goal["kind"] != "walk":
            # the walk's own sway leaves its end heading a few degrees off: turn from that heading,
            # so the sit is not rotated away from the chair
            turned_sit, _ = turn_then(turn, root_yaw(leg[-1:])[0], placed, model)
            leg = blend_join(leg, turned_sit)
            leg = np.concatenate([leg, np.repeat(leg[-1:], int(SEATED_HOLD * 50), axis=0)])
        else:
            leg = np.concatenate([leg, np.repeat(leg[-1:], hold, axis=0)])
        start = 0 if reference is None else len(reference) - 1
        reference = leg if reference is None else blend_join(reference, leg)
        legs.append({"goal": goal_id, "kind": goal["kind"], "start": start, "end": len(reference) - 1,
                     "routeLength": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()), "margin": used})
        if goal["kind"] != "walk":
            legs[-1].update(sitClip=sit_path, sitFrames=len(placed) + int(SEATED_HOLD * 50))
        position = reference[-1, :2].copy()
    return reference, legs


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from ..scenes import generate_scene
    from .depth_map import Grid
    from .home_navigation import _turn_clip, rasterise
    from .navigation_eval import _load, scene_elements
    run_dir, params, library, seat_library, seed, difficulty, index, margin, output, sit_paths, task = args
    run, policy, config, env, observe, bank = _load(run_dir, params, library)
    scene = generate_scene(3, seed, difficulty)
    if task:   # an instruction's steps (instruction_plan) replace the scene's task sequence
        scene["metadata"] = dict(scene["metadata"], taskSequence=[target for _, target in task["steps"]])
    bounds = scene["metadata"]["floorBounds"]
    grid = Grid(np.asarray(bounds[:2], float), np.asarray(bounds[2:], float), pad=.2)
    rasterise(grid, scene)
    walk_name, walk = bank[index % len(bank)]
    sits = [c for c in _sit_clips(seat_library) if not sit_paths or c[0] in sit_paths]
    sit = sits[index % len(sits)]
    model = B.build_model(floor=None, extra_world=scene_elements(scene))
    model.opt.iterations, model.opt.ls_iterations = config.solver_iterations, config.solver_ls_iterations
    try:
        reference, legs = build_tour(scene, walk, _turn_clip(library), sit, grid, margin, B.build_model(),
                                     back_steps(library))
    except ValueError as error:
        return {"seed": seed, "difficulty": difficulty, "success": False, "planFailed": str(error)}
    length = len(reference)
    table = np.concatenate([reference, np.repeat(reference[-1:], 20, axis=0)]).astype(np.float32)

    data = mujoco.MjData(model)
    data.qpos[:] = reference[0]
    mujoco.mj_forward(model, data)
    nb = len(B.SEGMENTS)
    body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
    names = {model.geom(g["id"]).id: g["id"] for g in scene["geoms"] if g.get("semantic") != "floor"}
    chair_ids = {i for i, name in names.items() if name.startswith("bedroom-chair/")}
    seat_id = model.geom("bedroom-chair/seat").id
    floor_id = model.geom(next(g["id"] for g in scene["geoms"] if g.get("semantic") == "floor")).id
    feet = {side: {model.geom(f"{side}_foot").id, model.geom(f"{side}_toe").id} for side in ("left", "right")}
    has_sit = legs[-1]["kind"] == "sit"
    sit_leg = legs[-1] if has_sit else {"start": length, "end": length, "sitFrames": 0}
    sit_from = sit_leg["end"] - sit_leg["sitFrames"]
    seat_xy = np.asarray(next(g for g in scene["goals"] if g["kind"] == "sit")["position"][:2])
    near_chair = False
    weight = float(model.body_subtreemass[1]) * 9.81

    info = {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(63), "linvel": jp.zeros((nb, 3))}
    jtable, jlength = jp.asarray(table), jp.int32(len(table))
    rng = jax.random.PRNGKey(0)
    contacts, fell, force = [], None, np.zeros(6)
    stops = {leg["goal"]: 0. for leg in legs if leg["kind"] == "walk"}
    stopped_ok, seated_time = set(), 0.
    goals = {g["id"]: np.asarray(g["position"][:2]) for g in scene["goals"]}
    trajectory = [data.qpos.copy()]
    for frame in range(1, length):
        view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
        action = np.clip(np.asarray(policy(observe(view, info, jtable, jlength), rng)[0]), -1., 1.)
        data.ctrl[:] = table[frame, 7:] + config.action_scale * action
        prev = data.xpos[1:nb + 1].copy()
        for _ in range(B.CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
            seat_load = 0.   # summed over the seat contacts of the last substep
            for i, c in enumerate(data.contact[:data.ncon]):
                pair = (c.geom1, c.geom2)
                hit = [g for g in pair if g in names]
                if not hit or not any(g in body_geoms for g in pair):
                    continue
                if hit[0] in chair_ids and near_chair:
                    if hit[0] == seat_id:
                        mujoco.mj_contactForce(model, data, i, force)
                        seat_load += abs(force[0])
                    continue
                contacts.append({"time": float(data.time), "obstacle": names[hit[0]],
                                 "body": model.geom(pair[0] if pair[0] in body_geoms else pair[1]).name})
        trajectory.append(data.qpos.copy())
        near_chair = near_chair or (frame >= sit_leg["start"] and np.linalg.norm(data.qpos[:2] - seat_xy) < CHAIR_NEAR)
        info = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
        up = data.xmat[1].reshape(3, 3)[2, 2]
        if up < .5 or (frame < sit_from and data.qpos[2] < .55):
            fell = float(data.time)
            break
        if contacts:
            break
        leg = next(l for l in legs if l["start"] <= frame <= l["end"])
        if leg["kind"] == "walk" and frame > leg["end"] - int(HOLD * 50):
            load = {"left": 0., "right": 0.}
            for i, c in enumerate(data.contact[:data.ncon]):
                for side, ids in feet.items():
                    if (c.geom1 == floor_id and c.geom2 in ids) or (c.geom2 == floor_id and c.geom1 in ids):
                        mujoco.mj_contactForce(model, data, i, force)
                        load[side] += abs(force[0])
            still = (np.linalg.norm(data.qpos[:2] - goals[leg["goal"]]) < .25 and np.linalg.norm(data.qvel[:2]) < .25
                     and up > .8 and min(load.values()) >= 20.)
            stops[leg["goal"]] = stops[leg["goal"]] + .02 if still else 0.
            if stops[leg["goal"]] >= .5:
                stopped_ok.add(leg["goal"])
        if has_sit and frame > length - 1 - int(SEATED_HOLD * 50):
            on_seat = (np.linalg.norm(data.qpos[:2] - reference[frame, :2]) < .15
                       and abs(data.qpos[2] - reference[frame, 2]) < .12 and seat_load >= SEAT_LOAD * weight)
            seated_time = seated_time + .02 if on_seat else 0.
    walk_goals = [l["goal"] for l in legs if l["kind"] == "walk"]
    reached = [g for g in walk_goals if g in stopped_ok]
    seated = seated_time >= 1. if has_sit else None
    result = {"seed": seed, "difficulty": difficulty, "walk": walk_name, "sitClip": sit[0] if has_sit else None,
              "instruction": task and task["instruction"], "steps": task and task["steps"],
              "doorWidth": scene["metadata"].get("portals") and scene["metadata"]["portals"][0]["width"],
              "legs": [{k: v for k, v in l.items() if k in ("goal", "kind", "routeLength", "margin")} for l in legs],
              "roomsReached": len(reached), "rooms": len(walk_goals), "seated": seated,
              "success": len(reached) == len(walk_goals) and seated is not False and not contacts and fell is None,
              "fellAt": fell, "firstContact": contacts[0] if contacts else None,
              "referenceSeconds": length / 50., "seconds": float(data.time)}
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"tour-{seed}.npz", qpos=np.asarray(trajectory), reference=reference)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--library", required=True, help="walk bank and turn clip (motion-library-v1)")
    parser.add_argument("--seat-library", required=True, help="sit clips with seats.json (motion-library-v3)")
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=1.)
    parser.add_argument("--margin", type=float, default=.4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sit-clips", default="", help="comma-separated sit clip paths (chosen on other seeds)")
    parser.add_argument("--instructions", default="", help="instruction_plan output: one episode per planned instruction")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sit_paths = tuple(p for p in args.sit_clips.split(",") if p)
    tasks = [None] * args.episodes
    if args.instructions:
        tasks = [{"instruction": c["instruction"], "steps": c["steps"]}
                 for c in json.loads(Path(args.instructions).read_text())["cases"] if c["steps"]]
        args.episodes = len(tasks)
    rng = np.random.default_rng(args.seed_base)
    difficulties = rng.uniform(args.difficulty_min, args.difficulty_max, args.episodes)
    jobs = [(args.run, args.params, args.library, args.seat_library, args.seed_base + i, float(difficulties[i]), i,
             args.margin, str(Path(args.output) / "trajectories"), sit_paths, tasks[i]) for i in range(args.episodes)]
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_episode, jobs, chunksize=1)
    rooms = sum(r.get("roomsReached", 0) for r in results), sum(r.get("rooms", 0) for r in results)
    summary = {"episodes": len(results), "successes": sum(r["success"] for r in results),
               "instructions": args.instructions or None,
               "roomsReached": list(rooms), "seated": [sum(bool(r.get("seated")) for r in results),
                                                       sum(r.get("seated") is not None for r in results)],
               "contacts": sum(r.get("firstContact") is not None for r in results),
               "falls": sum(r.get("fellAt") is not None for r in results),
               "planFailed": sum(bool(r.get("planFailed")) for r in results),
               "seeds": [args.seed_base, args.seed_base + args.episodes - 1],
               "planner": "A* on the true boxes (privileged)", "margin": args.margin, "sitClips": list(sit_paths),
               "engine": "native MuJoCo CPU", "params": args.params, "elapsedSeconds": time.time() - start}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps({"summary": summary, "episodes": results},
                                                               ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
