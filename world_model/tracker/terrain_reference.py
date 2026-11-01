"""G6 stage 4: a captured flat walk adapted to steps and stairs (kinematic reference only).

No captured stair climbing exists. A walk bent onto the route is adapted per frame:

* stance phases of each foot come from the flat walk (foot within 3 cm of its standing height);
  each stance is moved along the walking direction off a step edge onto the middle of the tread
  when it would land within ``EDGE`` of an edge
* a foot's height offset is the terrain height of its stance; during swing it moves to the next
  stance's height early when stepping up (first 40% of the swing, plus a small extra lift) and
  late when stepping down
* the pelvis rises by the smoothed mean of the two feet's offsets
* the legs reach the shifted foot and toe positions by damped least squares IK on the hip,
  knee and ankle hinges (the toes keep their angles)

The tracker then has to make this physical on the real step boxes.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

TERRAIN = ("step", "stair", "landing", "upper-floor")
FOOT_REST = .121
STANCE = .03
EDGE = .10          # m: a stance closer than this to a height change moves to the tread middle
LIFT = .04          # m extra swing lift when stepping up
IK_STEPS = 12


def terrain_height(scene):
    """h(xy): top of the highest terrain box under each point (0 = floor). Boxes are axis aligned."""
    boxes = [g for g in scene["geoms"] if g.get("semantic") in TERRAIN]
    for g in boxes:
        if abs(g.get("yaw", 0.)) > 1e-6:
            raise ValueError("rotated terrain boxes are not supported")
    low = np.asarray([np.asarray(g["position"][:2]) - np.asarray(g["size"][:2]) / 2 for g in boxes])
    high = np.asarray([np.asarray(g["position"][:2]) + np.asarray(g["size"][:2]) / 2 for g in boxes])
    top = np.asarray([g["position"][2] + g["size"][2] / 2 for g in boxes])

    def height(xy):
        xy = np.atleast_2d(xy)
        inside = np.all((xy[:, None] >= low[None]) & (xy[:, None] <= high[None]), axis=-1)
        return np.max(np.where(inside, top[None], 0.), axis=1)
    return height


def _stances(z):
    on = z < FOOT_REST + STANCE
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = [0] if on[0] else []
    starts += [e + 1 for e in edges if not on[e]]
    ends = [e for e in edges if on[e]] + ([len(z) - 1] if on[-1] else [])
    return list(zip(starts, ends))


def _tread_middle(xy, direction, height, step=.01, reach=.3):
    """Move ``xy`` along ``direction`` (either way, at most ``reach``) to the nearest point at least
    ``EDGE`` from any height change."""
    def clear(p):
        around = [p + direction * d for d in (-EDGE, EDGE)]
        return abs(height(around[0])[0] - height(p)[0]) < 1e-3 and abs(height(around[1])[0] - height(p)[0]) < 1e-3
    for d in np.arange(0., reach, step):
        for sign in (1., -1.):
            p = xy + direction * sign * d
            if clear(p):
                return p
    return xy


def adapt(model, qpos, height):
    """The walk ``qpos`` (T, nq) adapted to the terrain ``height`` function."""
    import mujoco
    from . import body as B
    from .path_reference import kinematics
    names = [getattr(s, "name", s) for s in B.SEGMENTS]
    q = qpos.astype(np.float64).copy()
    flat = kinematics(model, q)["xpos"]
    travel = np.gradient(gaussian_filter1d(q[:, :2], 10, axis=0), axis=0)
    travel /= np.maximum(np.linalg.norm(travel, axis=1, keepdims=True), 1e-9)
    targets, offsets = {}, {}
    for side in ("left", "right"):
        foot, toe = names.index(f"{side}_foot"), names.index(f"{side}_toe")
        stances = _stances(flat[:, foot, 2])
        shift = np.zeros((len(q), 2))
        level = np.zeros(len(q))
        placed = []
        for s, e in stances:
            xy = flat[s:e + 1, foot, :2].mean(axis=0)
            moved = _tread_middle(xy, travel[(s + e) // 2], height)
            placed.append((s, e, moved - xy, height(moved)[0]))
        for k, (s, e, d, h) in enumerate(placed):
            shift[s:e + 1], level[s:e + 1] = d, h
            if k + 1 < len(placed):   # swing to the next stance
                s2, _, d2, h2 = placed[k + 1]
                n = s2 - e - 1
                if n > 0:
                    phase = (np.arange(1, n + 1) / (n + 1))
                    if h2 >= h:
                        w = np.clip(phase / .4, 0., 1.)
                        lift = LIFT * np.sin(np.pi * phase) if h2 > h + 1e-3 else 0.
                    else:
                        w = np.clip((phase - .6) / .4, 0., 1.)
                        lift = 0.
                    w = w * w * (3 - 2 * w)
                    shift[e + 1:s2] = d + (d2 - d) * phase[:, None]
                    level[e + 1:s2] = h + (h2 - h) * w + lift
        if placed:
            level[:placed[0][0]] = placed[0][3]
            level[placed[-1][1] + 1:] = placed[-1][3]
            shift[placed[-1][1] + 1:] = placed[-1][2]
        offsets[side] = level
        delta = np.concatenate([shift, level[:, None]], axis=1)
        targets[side] = (foot, toe, flat[:, foot] + delta, flat[:, toe] + delta)
    q[:, 2] += gaussian_filter1d(.5 * (offsets["left"] + offsets["right"]), 8, mode="nearest")
    data = mujoco.MjData(model)
    jac_foot, jac_toe = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for side, (foot, toe, foot_target, toe_target) in targets.items():
        joints = [f"{side}_{part}_{axis}" for part in ("thigh", "shin", "foot") for axis in "xyz"]
        qadr = np.asarray([model.jnt_qposadr[model.joint(j).id] for j in joints])
        dadr = np.asarray([model.jnt_dofadr[model.joint(j).id] for j in joints])
        foot_body, toe_body = foot + 1, toe + 1     # body ids: world is 0
        for t in range(len(q)):
            data.qpos[:] = q[t]
            for _ in range(IK_STEPS):
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                error = np.concatenate([foot_target[t] - data.xpos[foot_body], toe_target[t] - data.xpos[toe_body]])
                if np.abs(error).max() < 2e-3:
                    break
                mujoco.mj_jacBody(model, data, jac_foot, None, foot_body)
                mujoco.mj_jacBody(model, data, jac_toe, None, toe_body)
                jac = np.concatenate([jac_foot[:, dadr], jac_toe[:, dadr]])
                step = jac.T @ np.linalg.solve(jac @ jac.T + .02 ** 2 * np.eye(6), error)
                data.qpos[qadr] += step
            q[t] = data.qpos
    return kinematics(model, q)
