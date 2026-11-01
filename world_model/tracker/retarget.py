"""Retarget T2M/mocap BVH clips onto the tracking humanoid in ``body.py``.

The BVH rest pose and the gakucho rest pose are both T-poses facing forward,
and every tracking body frame is world-aligned at rest, so the joint rotation
is the BVH local rotation after a fixed basis change. Bone lengths differ per
actor; the root translation is scaled by the leg-length ratio and the clip is
shifted vertically so its typical lowest point rests on the floor. Nothing
here runs physics: the result is a kinematic reference with diagnostics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..references import _parse_bvh
from .body import ACTUATED, DISPLAY_TO_PHYSICS, SEGMENT_NAMES, SEGMENTS, rest_joint_positions

CONTROL_HZ = 50
C = DISPLAY_TO_PHYSICS
# Wrist/forearm markers glitch often; their rotation speed is limited instead of
# discarding the clip. Larger body segments are never altered.
RATE_LIMITED = {"left_forearm", "left_hand", "right_forearm", "right_hand"}
RATE_LIMIT = 20.  # rad/s


def _leg_length(rest, names, upper, lower, foot):
    index = {name: i for i, name in enumerate(names)}
    return (np.linalg.norm(rest[index[lower]] - rest[index[upper]])
            + np.linalg.norm(rest[index[foot]] - rest[index[lower]]))


def _geom_bottoms(model, data):
    """Lowest world z of every body geom (boxes, capsules, spheres)."""
    bottoms = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        z, size, kind = data.geom_xpos[g, 2], model.geom_size[g], model.geom_type[g]
        R = data.geom_xmat[g].reshape(3, 3)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            bottoms.append(z - np.abs(R[2]).dot(size))
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            bottoms.append(z - abs(R[2, 2]) * size[1] - size[0])
        else:
            bottoms.append(z - size[0])
    return np.asarray(bottoms)


def _angular_velocity(quat_wxyz, dt):
    """World angular velocity of every body from consecutive quaternions (T, B, 4)."""
    T, B = quat_wxyz.shape[:2]
    rot = Rotation.from_quat(quat_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
    delta = rot[1:] @ np.swapaxes(rot[:-1], -1, -2)
    rotvec = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(T - 1, B, 3) / dt
    return np.concatenate([rotvec, rotvec[-1:]], axis=0)


def rate_limit(rotation, max_step):
    """Follow ``rotation`` but never rotate more than ``max_step`` rad per frame."""
    quats = rotation.as_quat()
    out = [Rotation.from_quat(quats[0])]
    limited = 0
    for t in range(1, len(quats)):
        target = Rotation.from_quat(quats[t])
        delta = target * out[-1].inv()
        angle = delta.magnitude()
        if angle > max_step:
            limited += 1
            target = Rotation.from_rotvec(delta.as_rotvec() * (max_step / angle)) * out[-1]
        out.append(target)
    return Rotation.from_quat(np.stack([r.as_quat() for r in out])), limited


def continuous_euler(rotation, order):
    """Intrinsic Euler angles that never jump between frames.

    Every rotation has two Tait-Bryan solutions, (a, b, c) and
    (a + pi, pi - b, c + pi). Choosing per frame the solution and 2*pi
    branch nearest to the previous frame removes flips that plain unwrapping
    cannot; hinge joints here are unlimited, so |b| > pi/2 is allowed.
    """
    euler = rotation.as_euler(order)
    alternative = np.stack([euler[:, 0] + np.pi, np.pi - euler[:, 1], euler[:, 2] + np.pi], axis=1)
    result = np.empty_like(euler)
    result[0] = euler[0]
    for t in range(1, len(euler)):
        previous = result[t - 1]
        best, best_cost = None, np.inf
        for candidate in (euler[t], alternative[t]):
            candidate = previous + (candidate - previous + np.pi) % (2 * np.pi) - np.pi
            cost = float(np.sum((candidate - previous) ** 2))
            if cost < best_cost:
                best, best_cost = candidate, cost
        result[t] = best
    return result


def retarget_bvh(text, model, profile, *, hz=CONTROL_HZ):
    """Return a dict of float32 arrays and per-clip diagnostics."""
    bvh = _parse_bvh(text, target_height=1.8)
    names = [bone.name for bone in bvh.bones]
    index = {name: i for i, name in enumerate(names)}
    missing = {j for seg in SEGMENTS for j in seg.bvh} - set(names)
    if missing:
        raise ValueError(f"BVH lacks joints {sorted(missing)}")
    count = len(bvh.frames)
    source_t = np.arange(count) * bvh.frame_time
    target_t = np.arange(0., source_t[-1] + 1e-9, 1. / hz)
    T = len(target_t)

    # Segment rotations in physics coordinates, resampled with slerp.
    local = {}
    for seg in SEGMENTS:
        R = np.eye(3)[None].repeat(count, 0)
        for joint in seg.bvh:
            R = R @ bvh.local_rotations[:, index[joint]]
        R = C @ R @ C.T
        local[seg.name] = Slerp(source_t, Rotation.from_matrix(R))(target_t)
    limited_frames = 0
    for name in RATE_LIMITED:
        local[name], count = rate_limit(local[name], RATE_LIMIT / hz)
        limited_frames += count

    rest = rest_joint_positions(profile)
    target_leg = (np.linalg.norm(rest["left_shin"] - rest["left_thigh"])
                  + np.linalg.norm(rest["left_foot"] - rest["left_shin"]))
    source_leg = _leg_length(bvh.rest_positions, names, "LeftUpLeg", "LeftLeg", "LeftFoot")
    scale = float(target_leg / source_leg)
    root = (C @ bvh.world_positions[:, index["Hips"]].T).T * scale
    root = np.stack([np.interp(target_t, source_t, root[:, k]) for k in range(3)], axis=1)

    qpos = np.zeros((T, model.nq))
    qpos[:, :3] = root
    qpos[:, 3:7] = local["pelvis"].as_quat()[:, [3, 0, 1, 2]]
    angles, geodesic = [], []
    for seg in ACTUATED:
        rotation = local[seg.name]
        angles.append(continuous_euler(rotation, seg.order))
        geodesic.append((rotation[1:] * rotation[:-1].inv()).magnitude() * hz if T > 1 else np.zeros(1))
    qpos[:, 7:] = np.concatenate(angles, axis=1)
    # keep the root quaternion sign continuous
    flips = np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1) < 0
    sign = np.cumprod(np.concatenate([[1.], np.where(flips, -1., 1.)]))
    qpos[:, 3:7] *= sign[:, None]

    data = mujoco.MjData(model)
    lowest = np.zeros(T)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        lowest[t] = _geom_bottoms(model, data).min()
    ground_shift = -float(np.median(lowest))
    qpos[:, 2] += ground_shift
    lowest += ground_shift

    dt = 1. / hz
    qvel = np.zeros((T, model.nv))
    for t in range(T - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t], qpos[t + 1])
    qvel[-1] = qvel[-2] if T > 1 else 0.
    xpos = np.zeros((T, model.nbody - 1, 3))
    xquat = np.zeros((T, model.nbody - 1, 4))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        xpos[t], xquat[t] = data.xpos[1:], data.xquat[1:]
    linvel = np.gradient(xpos, dt, axis=0) if T > 1 else np.zeros_like(xpos)
    angvel = _angular_velocity(xquat, dt) if T > 1 else np.zeros_like(xpos)
    hinge_speed = np.abs(qvel[:, 6:]).max(axis=0) if T > 1 else np.zeros(model.nv - 6)
    segment_speed = np.asarray([g.max() for g in geodesic])
    return {
        "qpos": qpos.astype(np.float32), "qvel": qvel.astype(np.float32),
        "xpos": xpos.astype(np.float32), "xquat": xquat.astype(np.float32),
        "linvel": linvel.astype(np.float32), "angvel": angvel.astype(np.float32),
        "meta": {"frames": int(T), "sourceFrames": int(count), "sourceFrameTime": float(bvh.frame_time),
                 "hz": hz, "rootScale": scale, "groundShift": ground_shift,
                 "lowestPointP05": float(np.percentile(lowest, 5)),
                 "lowestPointP95": float(np.percentile(lowest, 95)),
                 "maxHingeSpeed": float(hinge_speed.max()),
                 "maxHingeSpeedJoint": int(hinge_speed.argmax()),
                 "maxSegmentAngularSpeed": float(segment_speed.max()),
                 "rateLimitedHandForearmFrames": int(limited_frames),
                 "maxSegmentAngularSpeedSegment": ACTUATED[int(segment_speed.argmax())].name,
                 "maxRootSpeed": float(np.linalg.norm(qvel[:, :3], axis=1).max()),
                 "segments": list(SEGMENT_NAMES)},
    }


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_bvh(path):
    return Path(path).read_text(encoding="utf-8", errors="strict")
