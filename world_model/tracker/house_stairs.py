"""The house staircase (house2f: 16 risers of 0.19 m, 0.25 m treads, 0.9 m wide, straight, +X up).

    python -m world_model.tracker.house_stairs references --library <walk bank> --clips 240 --output <dir>
    python -m world_model.tracker.house_stairs evaluate --run <run> --params <pkl> --library <walk bank> --output <dir>

The captured walks step about 0.65 m; a 0.25 m tread takes one step each, so the stair gait is built
from a walk, not adapted from it:

* the walk is bent onto the straight route and its travel along the stair is scaled so one step is
  one tread (``follow`` repeats the gait, the scale is the walk's own step length over the tread)
* every stance is put on the stair grid, one tread further than the stance before: going up, the
  toes stop 2 cm short of the next riser (the 34 cm foot overhangs the tread behind); going down, the
  heel stays 2 cm off the riser behind (the toes overhang the nosing), as people do on steep stairs
* a swing going up lifts first and moves forward late, one going down moves forward first and drops
  late, each with a small extra lift; the foot keeps the walk's own orientation and swing height
* the pelvis follows the lower foot: going up the lower stance plus a little of the height difference,
  going down the lower of the two foot targets as they move (the supporting knee bends while the leading
  foot reaches down); damped least squares IK
  on the hip, knee and ankle hinges reaches the foot and toe targets (as ``terrain_reference.adapt``)

Training clips record the stair as boxes in the training environment's slots
(``mjx_tracking.TERRAIN_BOXES``: 15 ``house-tread`` boxes and the ``landing`` as the 2F floor).
A clip is kept only when IK reached the foot targets (95% of frames within 3 cm, all within 20 cm) and
no body part sinks more than 6 cm into the stair; otherwise another walk is tried.
The evaluation walks the real house (``house2f`` with its walls): up from the 1F hall onto the 2F
hall, and down; gates as ``stairs_eval``: no fall, no contact with walls or railings, a 0.5 s stop
on the goal floor.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..house2f import FLOOR_2F, RISE, RISERS, STAIR_X0, STAIR_X1, STAIR_Y, TREAD, WALL

CENTRE_Y = sum(STAIR_Y) / 2
WIDTH = STAIR_Y[1] - STAIR_Y[0] - WALL       # 0.9 m between the stair walls
TOE_TIP, HEEL = .267, .075                   # m from the ankle (foot body) along the foot
GAP = .02                                    # m kept from a riser
END_WINDOW = {True: (3.3, 3.85), False: (-1.3, -.95)}   # ankle x where a walk up / down may end (2F hall, 1F hall)
BOTTOM, TOP = (-1.15, 3.6)                   # x of the 1F hall (washroom wall at -1.6) and the 2F hall (wall at 4.15)
HOLD = 1.5                                   # s standing at the end of a training clip
PELVIS_SHARE = .3                            # pelvis above the lower foot: this share of the feet's height difference
DOWN_DROP = .02                              # m the pelvis is kept lower going down (reach for the leading foot)


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3. - 2. * u)


def level(k, rise=RISE):
    """Height of the surface a stance on stair index ``k`` stands on (0 = 1F floor, 16+ = 2F floor)."""
    return rise * np.clip(k, 0, RISERS)


def stair_index(x):
    """0 on the 1F floor, 1..15 on a tread, 16 on the 2F floor."""
    return int(np.clip(np.floor((x - STAIR_X0) / TREAD) + 1, 0, RISERS))


def surface(x, rise=RISE):
    return level(stair_index(x), rise)


def ankle_x(k, up):
    """Ankle x of a stance on index ``k`` (k = 1..15 treads; the grid goes on over both floors)."""
    front = STAIR_X0 + TREAD * k          # the riser in front of tread k going up (the back one going down)
    return front - GAP - TOE_TIP if up else front - GAP - HEEL


def boxes(rise=RISE):
    """The stair in the training environment's terrain slots."""
    out = [{"kind": "house-tread", "x": STAIR_X0 + TREAD * (k - .5), "y": CENTRE_Y, "top": rise * k, "size": [TREAD, WIDTH]}
           for k in range(1, RISERS)]
    out.append({"kind": "landing", "x": STAIR_X1 + 1.6, "y": CENTRE_Y, "top": rise * RISERS, "size": [3.2, 1.6]})
    return out


def _stances(z, rest):
    on = z < rest + .03
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = ([0] if on[0] else []) + [e + 1 for e in edges if not on[e]]
    ends = [e for e in edges if on[e]] + ([len(z) - 1] if on[-1] else [])
    return list(zip(starts, ends))


def _stance_list(model, names, qpos):
    """Body positions and every stance of both feet in time order: (start, end, side, mean ankle x)."""
    from .path_reference import kinematics
    fk = kinematics(model, qpos)["xpos"].astype(np.float64)
    rest = float(min(fk[0, names.index("left_foot"), 2], fk[0, names.index("right_foot"), 2]))
    stances = []
    for side in ("left", "right"):
        foot = names.index(f"{side}_foot")
        stances += [(s, e, side, float(fk[s:e + 1, foot, 0].mean())) for s, e in _stances(fk[:, foot, 2], rest)]
    return fk, sorted(stances)


def stair_walk(model, walk, up, start_x, end_x, lateral=0., rise=RISE):
    """A reference (``kinematics`` dict) walking the stair from ``start_x`` to about ``end_x`` (a whole number
    of treads) on the centre line (+ ``lateral``), standing ``HOLD`` s at the end."""
    import mujoco
    from . import body as B
    from .path_reference import follow, kinematics
    names = [getattr(s, "name", s) for s in B.SEGMENTS]
    sign = 1. if up else -1.
    y = CENTRE_Y + lateral
    direction = 0. if up else np.pi
    k_start = int(round((start_x + GAP + (TOE_TIP if up else HEEL) - STAIR_X0) / TREAD))
    want = int(round(abs(end_x - start_x) / TREAD))
    lo, hi = END_WINDOW[up]
    # the walk's own step length, then a flat walk whose number of steps ends the stair walk inside the end
    # window (``follow`` repeats whole gait cycles, so each walk only makes every other number of steps)
    _, probe = _stance_list(model, names, follow(walk, np.asarray([[0., 0.], [6., 0.]]), start_direction=0.)[0])
    step = float(np.median(np.abs(np.diff([x for *_, x in probe]))[1:-1]))
    found = None
    for steps in (want, want + 1, want - 1, want + 2, want - 2):
        flat, _ = follow(walk, np.asarray([[start_x, y], [start_x + sign * steps * step, y]]), start_direction=direction)
        fk, stances = _stance_list(model, names, flat)
        moving = sum(abs(b[3] - a[3]) > .4 * step for a, b in zip(stances, stances[1:]))
        if lo <= ankle_x(k_start + int(sign) * moving, up) <= hi:
            found = (flat, fk, stances, moving)
            break
    if found is None:
        raise ValueError("no number of steps of this walk ends inside the end window")
    flat, fk, stances, moving = found
    steps = moving
    # every stance on the stair grid, one index further than the stance before (side by side: the same)
    placed, k = [], k_start
    for i, (s, e, side, x) in enumerate(stances):
        if i and abs(x - stances[i - 1][3]) > .4 * step:
            k += int(sign)
        placed.append((s, e, side, x, k))
    # the flat walk's travel is warped so each stance lands on its grid place: the pelvis goes along
    pairs = {}
    for *_, x, k in placed:
        pairs.setdefault(k, []).append(x)
    xs = np.asarray([np.mean(v) for v in pairs.values()])
    gs = np.asarray([ankle_x(k, up) for k in pairs])
    order = np.argsort(xs)
    xs, gs = xs[order], gs[order]
    slope = TREAD / step

    def warp(x):
        out = np.interp(x, xs, gs)
        out = np.where(x < xs[0], gs[0] + (x - xs[0]) * slope, out)
        return np.where(x > xs[-1], gs[-1] + (x - xs[-1]) * slope, out)
    hold = int(HOLD * 50)
    flat = np.concatenate([flat, np.repeat(flat[-1:], hold, axis=0)])
    fk = np.concatenate([fk, np.repeat(fk[-1:], hold, axis=0)])
    targets, levels = {}, {}
    for side in ("left", "right"):
        foot, toe = names.index(f"{side}_foot"), names.index(f"{side}_toe")
        mine = [p for p in placed if p[2] == side]
        ax = np.zeros(len(flat))          # ankle x
        lift_z = np.zeros(len(flat))      # added to the walk's own foot height
        stance_level = np.zeros(len(flat))
        for i, (s, e, _, x, k) in enumerate(mine):
            ax[s:e + 1] = ankle_x(k, up)
            lift_z[s:e + 1] = stance_level[s:e + 1] = level(k, rise)
            if i + 1 < len(mine):
                s2, _, _, _, k2 = mine[i + 1]
                n = s2 - e - 1
                if n <= 0:
                    continue
                phase = np.arange(1, n + 1) / (n + 1)
                h, h2 = level(k, rise), level(k2, rise)
                if h2 > h + 1e-6:           # up: lift first, forward late
                    v, f, lift = smooth(phase / .35), smooth((phase - .2) / .7), .05
                elif h2 < h - 1e-6:         # down: forward first, drop late
                    v, f, lift = smooth((phase - .45) / .55), smooth(phase / .6), .03
                else:
                    v, f, lift = np.zeros(n), smooth(phase), 0.
                x0, x1 = ankle_x(k, up), ankle_x(k2, up)
                ax[e + 1:s2] = x0 + (x1 - x0) * f
                lift_z[e + 1:s2] = h + (h2 - h) * v + lift * np.sin(np.pi * phase)
                stance_level[e + 1:s2] = h
        first, last = mine[0], mine[-1]
        ax[:first[0]], ax[last[1] + 1:] = ankle_x(first[4], up), ankle_x(last[4], up)
        lift_z[:first[0]] = stance_level[:first[0]] = level(first[4], rise)
        lift_z[last[1] + 1:] = stance_level[last[1] + 1:] = level(last[4], rise)
        foot_target = fk[:, foot].copy()
        foot_target[:, 0] = ax
        foot_target[:, 2] += lift_z
        levels[side] = stance_level if up else lift_z
        targets[side] = (foot, toe, foot_target, foot_target + (fk[:, toe] - fk[:, foot]))
    low = np.minimum(levels["left"], levels["right"])
    q = flat.astype(np.float64).copy()
    q[:, 0] = warp(q[:, 0])
    share, drop = (PELVIS_SHARE, 0.) if up else (0., DOWN_DROP)
    q[:, 2] += gaussian_filter1d(low + share * np.abs(levels["left"] - levels["right"]), 8, mode="nearest") - drop
    scale = slope
    data = mujoco.MjData(model)
    jac_foot, jac_toe = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    missed = np.zeros(len(q))          # how far IK stayed from the foot targets (m)
    for side, (foot, toe, foot_target, toe_target) in targets.items():
        joints = [f"{side}_{part}_{axis}" for part in ("thigh", "shin", "foot") for axis in "xyz"]
        qadr = np.asarray([model.jnt_qposadr[model.joint(j).id] for j in joints])
        dadr = np.asarray([model.jnt_dofadr[model.joint(j).id] for j in joints])
        foot_body, toe_body = foot + 1, toe + 1
        for t in range(len(q)):
            data.qpos[:] = q[t]
            for _ in range(20):
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                error = np.concatenate([foot_target[t] - data.xpos[foot_body], toe_target[t] - data.xpos[toe_body]])
                if np.abs(error).max() < 2e-3:
                    break
                mujoco.mj_jacBody(model, data, jac_foot, None, foot_body)
                mujoco.mj_jacBody(model, data, jac_toe, None, toe_body)
                jac = np.concatenate([jac_foot[:, dadr], jac_toe[:, dadr]])
                data.qpos[qadr] += jac.T @ np.linalg.solve(jac @ jac.T + .02 ** 2 * np.eye(6), error)
            q[t] = data.qpos
            missed[t] = max(missed[t], float(np.abs(error).max()))
    stair_walk.debug = {"missed": missed, "targets": targets}
    out = kinematics(model, q)
    out["qvel"][-hold:] = 0.
    out["linvel"][-hold:] = 0.
    out["angvel"][-hold:] = 0.
    return out, {"scale": round(scale, 3), "step": round(step, 3), "stances": len(placed), "steps": steps, "moving": int(moving),
                 "endX": round(float(q[-1, 0]), 3),
                 "ikMissMax": round(float(missed.max()), 3), "ikMiss95": round(float(np.percentile(missed, 95)), 3)}


def stair_scene(rise=RISE):
    """Only the stair (and the floors) as a scene, for checking references kinematically."""
    geoms = [{"id": "floor", "type": "box", "position": [0., 0., -.05], "size": [20., 20., .1], "semantic": "floor"}]
    for b in boxes(rise):
        sx, sy = b["size"]
        geoms.append({"id": f"{b['kind']}-{len(geoms)}", "type": "box", "position": [b["x"], b["y"], b["top"] / 2],
                      "size": [sx, sy, b["top"]], "semantic": "stair"})
    return {"geoms": geoms}


def penetration(model_scene, qpos):
    """Deepest overlap (m) of the body with the scene, per frame (kinematic check of a reference)."""
    import mujoco
    data = mujoco.MjData(model_scene)
    out = np.zeros(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model_scene, data)
        out[t] = max([-c.dist for c in data.contact[:data.ncon]] + [0.])
    return out


# ---------------------------------------------------------------------------------------------- references
def _clip(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from . import body as B
    from .navigation_eval import _walk_bank
    index, library, walks = args
    rng = random.Random(index)
    bank = _walk_bank(library)
    up = index % 2 == 0
    if walks:     # only the walks given for this direction
        allowed = set(json.loads(Path(walks).read_text())["up" if up else "down"])
        bank = [b for b in bank if b[0] in allowed]
    rise = RISE if rng.random() < .7 else rng.uniform(.17, RISE)
    lateral = rng.uniform(-.08, .08)
    start = rng.uniform(-1.25, -1.0) if up else rng.uniform(3.4, 3.75)
    end = rng.uniform(3.4, 3.75) if up else rng.uniform(-1.25, -1.0)
    from .navigation_eval import scene_elements
    check = B.build_model(floor=None, extra_world=scene_elements(stair_scene(rise)))
    while True:
        walk_name, walk = bank[rng.randrange(len(bank))]
        try:
            data, info = stair_walk(B.build_model(), walk, up, start, end, lateral, rise)
        except ValueError:
            continue
        # quality gate: the legs reached the foot targets and nothing sinks into the stair
        info["penetrationMax"] = round(float(penetration(check, data["qpos"]).max()), 3)
        if info["ikMiss95"] <= .03 and info["ikMissMax"] <= .2 and info["penetrationMax"] <= .06:
            break
    clip = {"path": f"house-stairs/{index:04d}-{'up' if up else 'down'}", "dataset": "house-stairs", "label": "up" if up else "down",
            "text": f"家の階段を{'上る' if up else '下りる'}", "split": "train", "needsSeat": False, "status": "ok",
            "walk": walk_name, "terrain": boxes(rise), "rise": round(rise, 4), "lateral": round(lateral, 3), **info}
    return clip, data


def references(args):
    from .motion_library import ARRAYS
    with get_context("spawn").Pool(args.workers) as pool:
        made = pool.map(_clip, [(i, args.library, args.walks) for i in range(args.clips)], chunksize=1)
    manifest, arrays, offset = [], {k: [] for k in ARRAYS}, 0
    for clip, data in made:
        clip["start"], clip["length"] = offset, int(len(data["qpos"]))
        offset += clip["length"]
        for key in ARRAYS:
            arrays[key].append(data[key].astype(np.float32))
        manifest.append(clip)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    summary = {"version": 1, "source": "house_stairs.stair_walk (captured walks, one tread per step)", "walks": args.walks, "hz": 50,
               "clips": len(manifest), "usable": len(manifest), "frames": offset,
               "byDirection": {d: sum(c["label"] == d for c in manifest) for d in ("up", "down")}}
    (out / "manifest.json").write_text(json.dumps({"summary": summary, "clips": manifest}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------------- evaluation
def _touches(model, data, qpos, blocked, body_geoms):
    """True when the reference itself (kinematically, every other frame) puts a body part into a wall or a railing."""
    import mujoco
    for q in qpos[::2]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for c in data.contact[:data.ncon]:
            if c.dist < 0 and ((c.geom1 in blocked and c.geom2 in body_geoms) or (c.geom2 in blocked and c.geom1 in body_geoms)):
                return True
    return False


def _episode(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jp
    import mujoco
    from . import body as B
    from ..house2f import create_house
    from .navigation_eval import _load, scene_elements
    run_dir, params, library, index, output, walks, check_walls = args
    run, policy, config, env, observe, bank = _load(run_dir, params, library)
    rng = random.Random(1_000_000 + index)
    up = index % 2 == 0
    if walks:     # only the walks given for this direction (a stair gait the tracker follows)
        allowed = set(json.loads(Path(walks).read_text())["up" if up else "down"])
        bank = [b for b in bank if b[0] in allowed]
    scene = create_house()
    model = B.build_model(floor=None, extra_world=scene_elements(scene))
    model.opt.iterations, model.opt.ls_iterations = config.solver_iterations, config.solver_ls_iterations
    data = mujoco.MjData(model)
    nb = len(B.SEGMENTS)
    body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
    blocked = {model.geom(g["id"]).id: g["id"] for g in scene["geoms"] if g.get("semantic") not in ("floor", "stair")}
    walk_name, walk = bank[rng.randrange(len(bank))]
    start, end = (BOTTOM, TOP) if up else (TOP, BOTTOM)
    for _ in range(4 * len(bank)):      # (a walk whose steps do not end in the end window raises ValueError)
        try:
            reference = stair_walk(B.build_model(), walk, up, start, end, rng.uniform(-.05, .05))[0]["qpos"].astype(np.float64)
            if check_walls and _touches(model, data, reference, blocked, body_geoms):
                raise ValueError("the reference itself touches a wall")      # a gait nobody would choose here
            break
        except ValueError:
            walk_name, walk = bank[rng.randrange(len(bank))]
    else:
        raise ValueError(f"no walk fits the stair ({len(bank)} walks)")
    end = float(reference[-1, 0])
    table = np.concatenate([reference, np.repeat(reference[-1:], 20, axis=0)]).astype(np.float32)
    data.qpos[:] = reference[0]
    mujoco.mj_forward(model, data)
    info = {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(model.nu), "linvel": jp.zeros((nb, 3))}
    jtable, jlength = jp.asarray(table), jp.int32(len(table))
    key = jax.random.PRNGKey(index)
    trajectory, contacts, fell, stopped = [data.qpos.copy()], [], None, 0.
    goal_z = FLOOR_2F if up else 0.
    for frame in range(1, len(reference)):
        view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
        action = np.clip(np.asarray(policy(observe(view, info, jtable, jlength), key)[0]), -1., 1.)
        data.ctrl[:] = table[frame, 7:] + config.action_scale * action
        prev = data.xpos[1:nb + 1].copy()
        for _ in range(B.CONTROL_SUBSTEPS):
            mujoco.mj_step(model, data)
            for c in data.contact[:data.ncon]:
                hit = [g for g in (c.geom1, c.geom2) if g in blocked]
                if hit and any(g in body_geoms for g in (c.geom1, c.geom2)):
                    contacts.append({"time": round(float(data.time), 2), "obstacle": blocked[hit[0]]})
        info = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
        trajectory.append(data.qpos.copy())
        if data.xmat[1].reshape(3, 3)[2, 2] < .5 or data.qpos[2] - surface(data.qpos[0]) < .55:
            fell = round(frame * .02, 2)
            break
        near = abs(data.qpos[0] - end) < .25 and abs(data.qpos[1] - CENTRE_Y) < .25
        slow = np.linalg.norm(data.qvel[:2]) < .25
        on_floor = abs(data.qpos[2] - reference[-1, 2]) < .15
        stopped = stopped + .02 if near and slow and on_floor else 0.
    reached = stopped >= .5
    k = stair_index(data.qpos[0])
    result = {"index": index, "direction": "up" if up else "down", "walk": walk_name, "success": bool(reached and not fell and not contacts),
              "reached": bool(reached), "fell": fell, "contacts": contacts[:3], "risersPassed": int(k if up else RISERS - k),
              "final": [round(float(v), 3) for v in data.qpos[:3]], "seconds": round(len(trajectory) * .02, 2)}
    np.savez_compressed(Path(output) / f"stairs-{index:03d}.npz", qpos=np.asarray(trajectory), reference=reference)
    return result


def evaluate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [(args.run, args.params, args.library, i, str(out), args.walks, args.check_walls) for i in range(args.episodes)]
    with get_context("spawn").Pool(args.workers) as pool:
        episodes = pool.map(_episode, jobs, chunksize=1)
    by = {d: [sum(e["success"] for e in episodes if e["direction"] == d), sum(e["direction"] == d for e in episodes)] for d in ("up", "down")}
    summary = {"run": Path(args.run).name, "episodes": len(episodes), "success": sum(e["success"] for e in episodes),
               "byDirection": by, "walks": args.walks, "checkWalls": args.check_walls, "elapsedSeconds": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "episodes": episodes}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("references")
    r.add_argument("--library", required=True, help="walk bank (motion-library-v1)")
    r.add_argument("--clips", type=int, default=240)
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--walks", help="JSON {\"up\": [walk, ...], \"down\": [...]}: draw only these walks (default: the whole bank)")
    r.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--params", required=True)
    e.add_argument("--library", required=True)
    e.add_argument("--episodes", type=int, default=16)
    e.add_argument("--workers", type=int, default=16)
    e.add_argument("--walks", help="JSON {\"up\": [walk, ...], \"down\": [...]}: draw only these walks (default: the whole bank)")
    e.add_argument("--check-walls", action="store_true", help="draw another walk when the reference itself touches a wall or a railing")
    e.add_argument("--output", required=True)
    args = parser.parse_args()
    references(args) if args.command == "references" else evaluate(args)


if __name__ == "__main__":
    main()
