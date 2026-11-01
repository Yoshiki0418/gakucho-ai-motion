"""Bend a captured walk onto a planar path (kinematic reference only).

A walking clip is re-timed onto a target path by arc length: the root's
horizontal position follows the new path, the root yaw is rotated by the
difference between the new and the original path direction, and the
original sideways sway is kept in the rotated frame. Joint angles are not
changed, so a bent walk has some stance-foot skating that the physical
tracker must absorb. ``random_bend`` creates training variants; ``follow``
aligns a clip with a planned route.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

HZ = 50
BLEND = 10  # frames over which a gait-cycle seam's joint mismatch is faded


def _yaw_quat(angle):
    return np.stack([np.cos(angle / 2), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2)], -1)


def _quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def walk_frame(qpos, smooth_seconds=.6):
    """Smoothed travel path of a clip: arc length, direction and sideways sway."""
    xy = qpos[:, :2].astype(np.float64)
    smooth = np.stack([gaussian_filter1d(xy[:, k], smooth_seconds * HZ, mode="nearest") for k in range(2)], 1)
    step = np.diff(smooth, axis=0, prepend=smooth[:1])
    arc = np.cumsum(np.linalg.norm(step, axis=1))
    direction = np.arctan2(step[:, 1], step[:, 0])
    # direction is undefined while standing; hold the nearest moving value
    moving = np.linalg.norm(step, axis=1) > 1e-4
    if moving.any():
        idx = np.where(moving)[0]
        direction = np.interp(np.arange(len(xy)), idx, np.unwrap(direction[idx]))
    else:
        direction = np.zeros(len(xy))
    normal = np.stack([-np.sin(direction), np.cos(direction)], 1)
    sway = np.sum((xy - smooth) * normal, axis=1)
    along = np.sum((xy - smooth) * np.stack([np.cos(direction), np.sin(direction)], 1), axis=1)
    return arc, direction, sway, along


def bend(qpos, heading_change):
    """Rotate the walk progressively by ``heading_change`` (radians per frame,
    cumulative). Returns a new qpos array with the same joint angles."""
    arc, direction, sway, along = walk_frame(qpos)
    new_direction = direction + heading_change
    step = np.diff(arc, prepend=0.)
    path = qpos[0, :2] + np.cumsum(step[:, None] * np.stack([np.cos(new_direction), np.sin(new_direction)], 1), 0)
    tangent = np.stack([np.cos(new_direction), np.sin(new_direction)], 1)
    normal = np.stack([-np.sin(new_direction), np.cos(new_direction)], 1)
    out = qpos.copy()
    out[:, :2] = path + sway[:, None] * normal + along[:, None] * tangent
    out[:, 3:7] = _quat_mul(_yaw_quat(heading_change), qpos[:, 3:7])
    return out


def random_bend(qpos, rng, *, max_curvature=1.2, segments=3):
    """Random smooth curvature profile over the clip's travelled distance."""
    arc = walk_frame(qpos)[0]
    total = max(arc[-1], 1e-6)
    knots = np.linspace(0., total, segments + 2)
    values = rng.uniform(-max_curvature, max_curvature, len(knots)) * (rng.random(len(knots)) < .8)
    curvature = np.interp(arc, knots, values)
    change = np.concatenate([[0.], np.cumsum(curvature[1:] * np.diff(arc))])
    return bend(qpos, change), float(np.abs(curvature).max())


def mirror(qpos, model):
    """Left-right mirror of a clip (reflection x -> -x; +X is the body's left).

    The hinge axes are the world axes of the T-pose, so a mirrored joint keeps
    its x angle and negates its y and z angles, and left and right swap.
    """
    from .body import JOINT_NAMES, qpos_layout
    out = qpos.copy()
    out[:, 0] = -qpos[:, 0]
    out[:, 3:7] = qpos[:, 3:7] * np.asarray([1., 1., -1., -1.])
    columns = dict(zip(JOINT_NAMES, qpos_layout(model)[0]))
    for name, column in columns.items():
        partner = name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
        out[:, columns[partner]] = qpos[:, column] * (1. if name.endswith("_x") else -1.)
    return out


def root_yaw(qpos):
    w, x, y, z = qpos[..., 3], qpos[..., 4], qpos[..., 5], qpos[..., 6]
    return np.unwrap(np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))


def scale_turn(qpos, angle, model):
    """A turn-on-the-spot clip whose root turns by ``angle`` (radians, + = left).

    The captured turns are all to the right (90-170 degrees); a left turn is the
    mirrored clip. The root yaw change is scaled; joint angles, and so the
    stepping, are kept (the tracker absorbs the small foot slip).
    """
    yaw = root_yaw(qpos)
    total = yaw[-1] - yaw[0]
    if np.sign(total) != np.sign(angle):
        qpos = mirror(qpos, model)
        yaw, total = -yaw, -total
    out = qpos.copy()
    out[:, 3:7] = _quat_mul(_yaw_quat((angle / total - 1.) * (yaw - yaw[0])), qpos[:, 3:7])
    return out


def join(first, second, blend=20):
    """``second`` continued from the last frame of ``first`` (root aligned, joints and height faded)."""
    second = _align(second, first[-1])
    w = np.clip(np.arange(1, len(second) + 1) / blend, 0., 1.)[:, None]
    w = w * w * (3 - 2 * w)
    second = second.copy()
    second[:, 2:3] = (1 - w) * first[-1, 2] + w * second[:, 2:3]
    second[:, 7:] = (1 - w) * first[-1, 7:] + w * second[:, 7:]
    return np.concatenate([first, second[1:]])


def turn_then(turn, start_heading, reference, model, min_angle=np.radians(20.)):
    """Prefix ``reference`` with ``turn`` scaled so the body first turns from
    ``start_heading`` (root yaw, radians) to the reference's first heading on the spot."""
    angle = (root_yaw(reference[:1])[0] - start_heading + np.pi) % (2 * np.pi) - np.pi
    if abs(angle) < min_angle:
        return reference, 0.
    turned = scale_turn(turn, angle, model)
    # place the turn so that it ends where the reference starts
    end = turned[-1:].copy()
    delta = root_yaw(reference[:1])[0] - root_yaw(end)[0]
    rel = turned[:, :2] - end[0, :2]
    c, s = np.cos(delta), np.sin(delta)
    placed = turned.copy()
    placed[:, :2] = reference[0, :2] + np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
    placed[:, 3:7] = _quat_mul(_yaw_quat(np.full(len(placed), delta)), turned[:, 3:7])
    return join(placed, reference), float(angle)


def _align(segment, previous_last):
    """Move ``segment`` (qpos) so its first root pose equals ``previous_last``
    in horizontal position and yaw; height and joint angles are unchanged."""
    def yaw(q):
        w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
        return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    delta = yaw(previous_last) - yaw(segment[0])
    c, s_ = np.cos(delta), np.sin(delta)
    rel = segment[:, :2] - segment[0, :2]
    out = segment.copy()
    out[:, :2] = previous_last[:2] + np.stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]], 1)
    out[:, 3:7] = _quat_mul(_yaw_quat(np.full(len(out), delta)), segment[:, 3:7])
    return out


def walking_cycle(qpos, joint_column=7 + 3 * 13):
    """(start, period) of one gait cycle in the steady part of a walk.

    The steady part is where the smoothed speed is above 80% of its median
    while moving; the period maximises the autocorrelation of the left hip
    flexion angle between 0.7 and 1.6 s.
    """
    arc = walk_frame(qpos)[0]
    speed = np.gradient(arc) * HZ
    moving = speed > .1
    if moving.sum() < 2 * HZ:
        raise ValueError("clip has too little steady walking")
    steady = np.where(speed > .8 * np.median(speed[moving]))[0]
    a, b = steady[0], steady[-1]
    signal = qpos[a:b, joint_column] - qpos[a:b, joint_column].mean()
    best, period = -np.inf, None
    for lag in range(int(.7 * HZ), min(int(1.6 * HZ), len(signal) - 5)):
        score = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
        if score > best:
            best, period = score, lag
    if period is None or b - a < period + 2:
        raise ValueError("no full gait cycle in the steady part")
    return int(a), int(period), float(best)


def extend_walk(qpos, distance):
    """Repeat the steady gait cycle until the walk covers ``distance`` metres,
    keeping the clip's own start (from standing) and end (stopping)."""
    start, period, _ = walking_cycle(qpos)
    arc = walk_frame(qpos)[0]
    per_cycle = arc[start + period] - arc[start]
    # number of extra cycles whose total length is closest to ``distance``
    repeats = max(0, int(np.round((distance - arc[-1]) / max(per_cycle, 1e-6))))
    # Each piece starts with the frame its predecessor ended on, so aligning
    # that frame onto the predecessor's last frame keeps the root continuous.
    # The joint-angle mismatch at a seam fades out over BLEND frames.
    cycle = qpos[start:start + period + 1]
    pieces = [qpos[:start + period + 1]]
    for segment in [cycle] * repeats + [qpos[start + period:]]:
        aligned = _align(segment, pieces[-1][-1])
        offset = pieces[-1][-1, 7:] - aligned[0, 7:]
        fade = np.clip(1. - np.arange(1, len(aligned)) / BLEND, 0., 1.)[:, None]
        aligned[1:, 7:] += offset * fade
        pieces.append(aligned[1:])
    return np.concatenate(pieces)


def follow(qpos, route, *, start_direction=None, ramp=.6):
    """Bend a walk so it travels along ``route`` (polyline, metres) and ends
    at its last point. The gait cycle is repeated to match the route length;
    any remaining mismatch (< one cycle) is absorbed by a uniform stride scale.

    ``start_direction`` (travel angle, radians, atan2(dy, dx)) keeps the body
    facing that way while standing and turns it onto the route over the
    first ``ramp`` metres, so the reference can start from a given spawn pose.
    """
    route = np.asarray(route, dtype=np.float64)
    seg = np.diff(route, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate([[0.], np.cumsum(seg_len)])
    walk = extend_walk(qpos, cumulative[-1])
    walk = walk.copy()
    walk[:, :2] += route[0] - walk[0, :2]
    arc, direction, _, _ = walk_frame(walk)
    scale = cumulative[-1] / max(arc[-1], 1e-6)
    s = arc * scale
    index = np.clip(np.searchsorted(cumulative, s, side="right") - 1, 0, len(seg) - 1)
    target = gaussian_filter1d(np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))[index], .4 * HZ, mode="nearest")
    if start_direction is not None:
        weight = np.clip(s / ramp, 0., 1.)
        weight = weight * weight * (3. - 2. * weight)
        offset = np.arctan2(np.sin(target - start_direction), np.cos(target - start_direction))
        target = start_direction + weight * offset
    # the walk's own start direction is aligned with the route start
    bent = bend(walk, target - direction)
    bent[:, :2] = route[0] + (bent[:, :2] - route[0]) * scale
    # The pelvis stops slightly off the smoothed travel path; spread the
    # remaining end error over the walk so the reference ends on the goal.
    weight = np.clip(s / max(s[-1], 1e-6), 0., 1.)
    weight = weight * weight * (3. - 2. * weight)
    bent[:, :2] += (route[-1] - bent[-1, :2]) * weight[:, None]
    return bent, scale


def kinematics(model, qpos, hz=HZ):
    """qvel, body positions/quaternions and angular velocities for a qpos clip."""
    from .retarget import _angular_velocity
    T, data = len(qpos), mujoco.MjData(model)
    qvel = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, qvel[t], 1. / hz, qpos[t], qpos[t + 1])
    qvel[-1] = qvel[-2]
    nb = model.body("right_toe").id
    xpos, xquat = np.zeros((T, nb, 3)), np.zeros((T, nb, 4))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        xpos[t], xquat[t] = data.xpos[1:nb + 1], data.xquat[1:nb + 1]
    return {"qpos": qpos.astype(np.float32), "qvel": qvel.astype(np.float32),
            "xpos": xpos.astype(np.float32), "xquat": xquat.astype(np.float32),
            "linvel": np.gradient(xpos, 1. / hz, axis=0).astype(np.float32),
            "angvel": _angular_velocity(xquat, 1. / hz).astype(np.float32)}
