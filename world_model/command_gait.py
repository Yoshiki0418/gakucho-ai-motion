"""A physical command gait with walking pace separate from sole clearance.

This is a reference-target adapter for a short-horizon command motor, not a
motion policy or a body-state setter. The established final-goal motor fades
its entire gait near a destination. Applying that fade to a continuously
moving 0.35--0.5 m command also suppressed foot unloading. Here command
distance scales longitudinal stride while walking retains the native T2M
cycle, swing lift and lateral balance. The caller still applies bounded
joint torque PD and a frozen or explicitly retrained residual policy.
"""
from __future__ import annotations

import math

import numpy as np

from .humanoid import JOINT_NAMES
from .locomotion import rotation_x, rotation_y, solve_leg
from .steering import rotation_z, solve_leg_with_yaw, wrap_angle


def command_gait_pose(env, command_goal):
    """Return 20 joint targets; leave all simulator and environment state intact.

    ``command_goal`` is a finite world-space target used only to choose stride
    and heading. Ground height and segment dimensions come from the current
    physical environment. No actual task goal, pose, velocity, force, control,
    contact state or gait-counter mutation occurs in this function.
    """
    goal = np.asarray(command_goal, dtype=np.float64)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("command_goal must contain three finite metre coordinates")
    sample = env.reference.sample(float(env.data.time) + env.phase_offset)
    root_rotation = env.data.xmat[env.pelvis_body_id].reshape(3, 3)
    heading = math.atan2(root_rotation[1, 0], root_rotation[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    horizontal = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    distance = float(np.linalg.norm(goal[:2] - env.data.qpos[:2]))
    ramp = float(np.clip(env.data.time / .8, 0., 1.))
    ramp = ramp * ramp * (3. - 2. * ramp)
    walk_gate = float(np.clip(distance / .15, 0., 1.))
    walking_weight = ramp * walk_gate
    pace = float(np.clip(distance / .7, 0., 1.))
    longitudinal_weight = walking_weight * pace
    step_length = float(np.linalg.norm(env.reference.stride)) * .46

    # Only torso/arm reference excursions use pace. Leg trajectories and the
    # ankle's roll-shifting feedback have their own independent walk gate.
    pose = env.standing_pose + longitudinal_weight * (
        sample.joint_targets - env.standing_pose)
    pose[JOINT_NAMES.index("torso_pitch")] = .05 * longitudinal_weight
    pose[JOINT_NAMES.index("torso_roll")] = 0.
    virtual_pelvis = env.data.qpos[:3].copy()
    virtual_pelvis[2] = env.floor_height + (
        .93 - .03 * walking_weight if env.avatar_profile is None
        else env.standing_root_height - .03 * env._body_scale * walking_weight)
    leg_targets = {}
    for side, sign, phase in (("left", 1, sample.phase),
                              ("right", -1, (sample.phase + .5) % 1.)):
        if phase < .4:
            fraction = phase / .4
            blend = fraction * fraction * (3. - 2. * fraction)
            longitudinal = step_length * (.5 - blend)
            # A smooth 0.075 m plateau keeps the sole unloaded long enough
            # at reduced stride. The original sine's brief peak was tracked
            # asymmetrically by the transferred stopping motor. The 0.4-cycle
            # swing and native T2M period remain unchanged.
            rise = float(np.clip(fraction / .30, 0., 1.))
            fall = float(np.clip((1. - fraction) / .30, 0., 1.))
            rise = rise * rise * (3. - 2. * rise)
            fall = fall * fall * (3. - 2. * fall)
            lift = .075 * min(rise, fall)
        else:
            longitudinal = step_length * (-.5 + (phase - .4) / .6)
            lift = 0.
        if env.avatar_profile is None:
            foot_center = np.asarray([sign * .10,
                                      -.065 + longitudinal * longitudinal_weight,
                                      env.floor_height + .08 + lift * walking_weight])
            world_ankle = horizontal @ [foot_center[0], foot_center[1] + .065, 0.]
            world_ankle[:2] += env.data.qpos[:2]
            world_ankle[2] = foot_center[2]
            hip_offset = np.asarray([sign * .10, 0., -.02])
            upper = lower = .415
        else:
            hip_offset = env.model.body_pos[env.model.body(f"{side}_thigh").id]
            upper = float(np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_shank").id]))
            lower = float(np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_foot").id]))
            foot_id = env.model.geom(f"humanoid_{side}_foot").id
            foot_offset = env.model.geom_pos[foot_id]
            foot_center = horizontal @ [hip_offset[0] + foot_offset[0],
                                        hip_offset[1] + foot_offset[1]
                                        + longitudinal * longitudinal_weight, 0.]
            foot_center[:2] += env.data.qpos[:2]
            foot_center[2] = (env.floor_height + env.model.geom_size[foot_id, 2]
                              + lift * walking_weight)
            world_ankle = foot_center - horizontal @ foot_offset
        vector = root_rotation.T @ (world_ankle - virtual_pelvis) - hip_offset
        pitch, roll, knee, ankle_pitch, ankle_roll = solve_leg(
            vector, root_rotation, horizontal, upper=upper, lower=lower)
        names = tuple(f"{side}_{suffix}" for suffix in
                      ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
        indices = np.asarray([JOINT_NAMES.index(name) for name in names])
        targets = np.asarray([pitch, roll, 0., knee, ankle_pitch, ankle_roll])
        pose[indices] = env.standing_pose[indices] + walking_weight * (
            targets - env.standing_pose[indices])
        leg_targets[side] = (indices, upper, lower)

    # Preserve the original physical balance feedback; short stride does not
    # turn off weight transfer needed to unload the other foot.
    local_up = root_rotation.T @ [0., 0., 1.]
    pitch = math.atan2(local_up[1], local_up[2])
    roll = math.atan2(-local_up[0], local_up[2])
    desired_roll = -.065 * math.sin(2. * math.pi * sample.phase) * walking_weight
    for side in ("left", "right"):
        pose[JOINT_NAMES.index(f"{side}_ankle_pitch")] += (
            2. * (pitch - .03 * longitudinal_weight) + .3 * env.data.qvel[3])
        pose[JOINT_NAMES.index(f"{side}_ankle_roll")] += (
            2. * (roll - desired_roll) + .3 * env.data.qvel[4])

    # Same bounded hip-yaw steering and analytic preservation of the ankle
    # origin/sole normal as SteeringHumanoidEnv, applied only to targets.
    command = 0.
    if env.steering_enabled and distance > .18:
        difference = goal[:2] - env.data.qpos[:2]
        desired_heading = math.atan2(difference[0], -difference[1])
        error = wrap_angle(desired_heading - heading)
        near = float(np.clip((distance - .18) / .5, 0., 1.))
        command = float(np.clip(env.steering_gain * error
                                - env.steering_damping * float(env.data.qvel[5]),
                                -env.max_turn_radians, env.max_turn_radians)) * ramp * near
    if command:
        for side, phase in (("left", sample.phase),
                            ("right", (sample.phase + .5) % 1.)):
            swing = math.sin(math.pi * phase / .4) ** 2 if phase < .4 else 0.
            indices, upper, lower = leg_targets[side]
            pitch, roll, previous_yaw, knee, ankle_pitch, ankle_roll = pose[indices]
            requested_yaw = previous_yaw + command * (2. * swing - 1.)
            hip_rotation = rotation_x(pitch) @ rotation_y(roll) @ rotation_z(previous_yaw)
            ankle_origin = hip_rotation @ (np.asarray([0., 0., -upper])
                                           + rotation_x(knee) @ [0., 0., -lower])
            foot_rotation = (root_rotation @ hip_rotation @ rotation_x(knee)
                             @ rotation_x(ankle_pitch) @ rotation_y(ankle_roll))
            pose[indices] = solve_leg_with_yaw(
                ankle_origin, root_rotation, foot_rotation[:, 2], requested_yaw,
                upper=upper, lower=lower)
    return np.asarray(pose, dtype=np.float64)
