"""Small-angle walking steering through the existing actuated hip joints.

This prior changes joint targets only. The floating pelvis is rotated by real
foot contact reactions; it never receives commanded pose, velocity, or force.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .environment import HumanoidRoomEnv
from .humanoid import (CONTROL_SUBSTEPS, JOINT_LIMITS, JOINT_NAMES, PD_KD, PD_KP,
                       RESIDUAL_LIMITS, TORQUE_LIMITS)
from .locomotion import FootstepHumanoidEnv, rotation_x, rotation_y


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def solve_leg_with_yaw(vector, root_rotation, foot_up, yaw, *, upper=.415, lower=.415):
    """Analytic Rx/Ry/Rz hip, Rx knee and Rx/Ry ankle inverse kinematics.

    ``vector`` is the desired ankle origin relative to the hip, expressed in
    the pelvis frame. ``foot_up`` is the desired world-space sole normal.
    Prescribing the third hip axis rotates the knee's bending plane, so hip
    pitch/roll and both ankle angles must be solved again. This returns six
    joint targets and never modifies a MuJoCo state.
    """
    vector = np.asarray(vector, dtype=np.float64)
    root_rotation = np.asarray(root_rotation, dtype=np.float64)
    foot_up = np.asarray(foot_up, dtype=np.float64)
    if (vector.shape != (3,) or root_rotation.shape != (3, 3) or foot_up.shape != (3,)
            or not np.isfinite(vector).all() or not np.isfinite(root_rotation).all()
            or not np.isfinite(foot_up).all() or not math.isfinite(yaw)):
        raise ValueError("Leg targets must contain finite vectors, rotation and yaw")
    if not math.isfinite(upper) or not math.isfinite(lower) or min(upper, lower) <= .002:
        raise ValueError("Leg segments must have finite positive metre lengths")
    length = float(np.linalg.norm(vector))
    if length < abs(upper-lower)-1e-9 or length > upper+lower+1e-9:
        raise ValueError("The ankle target is outside the physical leg reach")
    knee = math.acos(float(np.clip((length*length-upper*upper-lower*lower)/(2*upper*lower), -1., 1.)))
    down = upper+lower*math.cos(knee)
    bent = lower*math.sin(knee)
    yaw_bend = bent*math.sin(yaw)
    lateral_reach = math.hypot(down, yaw_bend)
    if lateral_reach < 1e-10 or abs(vector[0]) > lateral_reach+1e-9:
        raise ValueError("The requested hip yaw cannot reach this lateral ankle target")
    roll = math.asin(float(np.clip(-vector[0]/lateral_reach, -1., 1.)))-math.atan2(yaw_bend, down)
    inner_y = bent*math.cos(yaw)
    inner_z = yaw_bend*math.sin(roll)-down*math.cos(roll)
    pitch = math.atan2(vector[1], -vector[2])-math.atan2(inner_y, -inner_z)
    parent = root_rotation @ rotation_x(pitch) @ rotation_y(roll) @ rotation_z(yaw) @ rotation_x(knee)
    normal_length = float(np.linalg.norm(foot_up))
    if normal_length < 1e-10:
        raise ValueError("The sole normal cannot be zero")
    local_up = parent.T @ (foot_up/normal_length)
    ankle_pitch = math.atan2(-local_up[1], local_up[2])
    ankle_roll = math.atan2(local_up[0], math.hypot(local_up[1], local_up[2]))
    return pitch, roll, float(yaw), knee, ankle_pitch, ankle_roll


class _ObservedPhysicsEnv(HumanoidRoomEnv):
    """The original physics step with an optional read-only substep observer.

    Contacts in ``data`` immediately after mj_step are the contacts used by
    that real 2 ms solver step. A control-frame-only observer can miss a brief
    furniture collision. The callback receives (model, data) without an extra
    forward or altered integration. It must only measure and record state.
    """

    def step(self, action):
        if self._episode_done:
            raise RuntimeError("reset() is required after a completed episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action must contain 20 finite values")
        action = np.clip(action, -1, 1)
        target = np.clip(self._reference_pose() + RESIDUAL_LIMITS * action,
                         JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        for _ in range(CONTROL_SUBSTEPS):
            torque = PD_KP * (target - self.data.qpos[self._qpos_indices]) - PD_KD * self.data.qvel[self._qvel_indices]
            self._last_torque[:] = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            self.data.ctrl[:] = self._last_torque
            mujoco.mj_step(self.model, self.data)
            observer = getattr(self, "_physics_contact_observer", None)
            if observer is not None:
                observer(self.model, self.data)
        self._step_count += 1
        # Preserve the original control frame's derived-geometry refresh.
        mujoco.mj_forward(self.model, self.data)
        info = self._info()
        distance = self._goal_distance()
        smoothness = float(np.mean(np.square(action - self._last_action)))
        cost = .015 * float(np.mean(np.square(self._last_torque / TORQUE_LIMITS))) + .02 * smoothness
        if self.task == "stand":
            height_reward = math.exp(-((info["root_height"] - self.standing_root_height) / (.16*self._body_scale)) ** 2)
            speed_cost = .15 * float(np.square(self.data.qvel[:3]).sum())
            reward = 1.2 * max(info["upright"], 0) + .8 * height_reward - speed_cost - cost
            instant_success = (info["upright"] > .94 and abs(info["root_height"] - self.standing_root_height) < .10*self._body_scale
                               and np.linalg.norm(self.data.qvel[:3]) < .30
                               and info["left_foot_contact"] and info["right_foot_contact"])
        elif self.task == "walk":
            progress = (self._previous_goal_distance - distance) / self.dt
            reward = progress + .5 * max(info["upright"], 0) - cost
            instant_success = distance < .25 and np.linalg.norm(self.data.qvel[:3]) < .35 and info["upright"] > .8
        else:
            height_error = float(self.data.qpos[2] - self.goal[2])
            horizontal_error = float(np.linalg.norm(self.data.qpos[:2] - self.goal[:2]))
            reward = math.exp(-(horizontal_error / .35) ** 2 - (height_error / .20) ** 2) + .3 * max(info["upright"], 0) - cost
            instant_success = (horizontal_error < .15 and abs(height_error) < .10
                               and np.linalg.norm(self.data.qvel[:3]) < .20 and info["upright"] > .8
                               and info["pelvis_support_contact"])
        self._success_seconds = self._success_seconds + self.dt if instant_success else 0.0
        info["success"] = self._success_seconds >= .5
        info["success_seconds"] = self._success_seconds
        self._last_action[:] = action
        self._previous_goal_distance = distance
        terminated = bool(info["fall"])
        truncated = self._step_count >= self.max_steps
        if terminated:
            reward -= 3.0
        self._episode_done = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self._observation(info), float(reward), terminated, truncated, info


class SteeringHumanoidEnv(FootstepHumanoidEnv, _ObservedPhysicsEnv):
    """A bounded yaw prior for walking towards the supplied goal.

    A supported leg twists opposite to the requested body turn. Its ground
    reaction turns the pelvis; the unloaded leg points towards the new heading.
    Turning amplitude fades near the goal and during the standing start.
    A residual PPO policy still supplies the balancing joint corrections.
    Navigation supplies its final goal directly; this prior adds no route or
    hidden waypoints. Obstacle avoidance is learned by the residual policy.
    """

    def __init__(self, *, steering_enabled=True, max_turn_radians=.12,
                 steering_gain=.35, steering_damping=.06, **kwargs):
        parameters = (max_turn_radians, steering_gain, steering_damping)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("Steering parameters must be finite")
        if not 0 < max_turn_radians <= .35:
            raise ValueError("max_turn_radians must be in (0, .35]")
        if not 0 <= steering_gain <= 2 or not 0 <= steering_damping <= .5:
            raise ValueError("Steering gain/damping exceed the bounded prior")
        self.steering_enabled = bool(steering_enabled)
        self.max_turn_radians = float(max_turn_radians)
        self.steering_gain = float(steering_gain)
        self.steering_damping = float(steering_damping)
        self._last_steering_error = 0.
        self._last_steering_target = 0.
        kwargs.setdefault("strict_gait_reward", True)
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._last_steering_error = 0.
        self._last_steering_target = 0.
        return super().reset(**kwargs)

    def _steering_command(self):
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        difference = self.goal[:2]-self.data.qpos[:2]
        distance = float(np.linalg.norm(difference))
        if distance <= .18 or not self.steering_enabled:
            return 0., 0.
        desired_heading = math.atan2(difference[0], -difference[1])
        error = wrap_angle(desired_heading-heading)
        ramp = float(np.clip(self.data.time/.8, 0., 1.))
        ramp = ramp*ramp*(3.-2.*ramp)
        near = float(np.clip((distance-.18)/.5, 0., 1.))
        yaw_rate = float(self.data.qvel[5])
        command = float(np.clip(self.steering_gain*error-self.steering_damping*yaw_rate,
                                -self.max_turn_radians, self.max_turn_radians))
        return error, command*ramp*near

    def _reference_pose(self):
        pose = super()._reference_pose()
        error, command = self._steering_command()
        self._last_steering_error = error
        self._last_steering_target = command
        if command == 0.:
            return pose
        phase = self.reference.sample(float(self.data.time)+self.phase_offset).phase
        for side, side_phase in (("left", phase), ("right", (phase+.5)%1.)):
            # Smooth changes avoid a discontinuous twist at liftoff/touchdown.
            if side_phase < .4:
                swing = math.sin(math.pi*side_phase/.4)**2
            else:
                swing = 0.
            requested_yaw = pose[JOINT_NAMES.index(f"{side}_hip_yaw")]+command*(2.*swing-1.)
            if requested_yaw == pose[JOINT_NAMES.index(f"{side}_hip_yaw")]:
                continue
            names = tuple(f"{side}_{suffix}" for suffix in
                          ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
            indices = [JOINT_NAMES.index(name) for name in names]
            pitch, roll, previous_yaw, knee, ankle_pitch, ankle_roll = pose[indices]
            upper = float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_shank").id]))
            lower = float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_foot").id]))
            hip_rotation = rotation_x(pitch) @ rotation_y(roll) @ rotation_z(previous_yaw)
            # Pure leg FK from the original targets; the actual floating body
            # is read-only. Retaining the original sole normal also retains
            # the ankle balance feedback already included by Footstep.
            ankle_origin = hip_rotation @ (np.asarray([0., 0., -upper])
                                           +rotation_x(knee) @ [0., 0., -lower])
            root_rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
            foot_rotation = root_rotation @ hip_rotation @ rotation_x(knee) @ rotation_x(ankle_pitch) @ rotation_y(ankle_roll)
            pose[indices] = solve_leg_with_yaw(ankle_origin, root_rotation, foot_rotation[:, 2],
                                               requested_yaw, upper=upper, lower=lower)
        return pose

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(steering_enabled=self.steering_enabled,
                    steering_heading_error_radians=self._last_steering_error,
                    steering_joint_target_radians=self._last_steering_target,
                    steering_max_joint_target_radians=self.max_turn_radians,
                    steering_prior="bounded hip yaw with analytic leg IK preserving ankle origin and sole normal; physical foot contact reactions")
        return observation, reward, terminated, truncated, info
