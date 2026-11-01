"""Experimental whole-body inverse dynamics; every motion is produced by torque.

The QP selects joint torques and estimated sole wrenches for a floating body.
The integrated MjData never receives pose/velocity/acceleration assignments,
mocap poses, or external root forces. FK uses an independent reference MjData
to select joint targets only. Measured physical rollouts decide gait quality.
"""
from __future__ import annotations

import math
import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .humanoid import CONTROL_SUBSTEPS, JOINT_LIMITS, JOINT_NAMES, TORQUE_LIMITS, STANDING_ROOT_HEIGHT


def _rotation_error(target, current):
    return Rotation.from_matrix(target @ current.T).as_rotvec()


class WholeBodyController:
    """A disclosed model-based controller, not a learned policy.

    ``compute_torques()`` returns 20 bounded joint torques. ``step()`` holds
    them for the environment's ten 2 ms physics steps and returns real stats.
    Standing is the initial baseline; contact-gated footsteps are experimental.
    ``goal_direction=True`` adds a flat-floor final approach with separate
    sole lanes and a measured two-foot stop; the body retains its heading.
    """

    def __init__(self, env, *, walk=False, goal_direction=False):
        self.env = env
        self.model, self.data = env.model, env.data
        self.nv, self.nu = self.model.nv, self.model.nu
        if self.nu != len(TORQUE_LIMITS) or self.nv != self.nu + 6:
            raise ValueError("WBC requires the floating 20-joint physical rig")
        self._actuated = np.asarray(env._qvel_indices)
        self._selection = np.zeros((self.nu, self.nv))
        self._selection[np.arange(self.nu), self._actuated] = 1
        self._foot_bodies = [self.model.body(f"{side}_foot").id for side in ("left", "right")]
        self._foot_geoms = tuple(env._foot_geom_ids)
        self._profiled_body = env.avatar_profile is not None
        self._hip_offsets = [self.model.body_pos[self.model.body(f"{side}_thigh").id].copy()
                             for side in ("left", "right")]
        self._leg_lengths = [(float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_shank").id])),
                              float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_foot").id])))
                             for side in ("left", "right")]
        self._ankle_from_sole = [-self.model.geom_pos[geom] + [0., 0., self.model.geom_size[geom, 2]]
                                 for geom in self._foot_geoms]
        self._body_ids = [index for index in range(1, self.model.nbody)
                          if self._belongs_to_pelvis(index)]
        self.mass = float(self.model.body_subtreemass[env.pelvis_body_id])
        self.weight = self.mass * float(np.linalg.norm(self.model.opt.gravity))
        self.walking = bool(walk)
        self.goal_direction = bool(goal_direction)
        self.stride = .20
        self.swing_seconds = .60
        self.transfer_seconds = 1.80
        self.swing_height = .075
        self.com_height_offset = .060
        self.stance_weight = 3000.
        self.com_weight = 500.
        self.orientation_weight = 1000.
        self.ik_posture_weight = 3. if walk else 0.
        self.goal_tolerance = .12 if self.goal_direction else .20
        self._reference_data = mujoco.MjData(self.model)
        self.diagnostics = {}
        self.reset()

    def _belongs_to_pelvis(self, body):
        while body:
            if body == self.env.pelvis_body_id:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def reset(self):
        """Capture the current physical anchors; resetting physics belongs to env."""
        mujoco.mj_forward(self.model, self.data)
        self.mass = float(self.model.body_subtreemass[self.env.pelvis_body_id])
        self.weight = self.mass * float(np.linalg.norm(self.model.opt.gravity))
        self.anchors = [self._foot_state(index)[:2] for index in range(2)]
        self.com_goal = self.data.subtree_com[self.env.pelvis_body_id].copy()
        self._neutral_com_height = float(self.com_goal[2])
        self._neutral_com_offset = self.com_goal - self.data.qpos[:3]
        self.root_rotation = self.data.xmat[self.env.pelvis_body_id].reshape(3, 3).copy()
        self.posture = self.env.standing_pose.copy()
        self.active_feet = [True, True]
        self.load_fractions = np.asarray([.5, .5])
        self.swing_goal = None
        self.swing_velocity = np.zeros(3)
        self.swing_acceleration = np.zeros(3)
        self.phase = "settling" if self.walking else "standing"
        self._phase_started = float(self.data.time)
        self._previous_solution = None
        self._qp_failures = 0
        self.completed_steps = 0
        self.failure = None
        self._support = 1
        self._swing = 0
        self._stable_seconds = 0.
        self._transfer_start = self.com_goal.copy()
        self._transfer_goal = self.com_goal.copy()
        self._transfer_load_start = self.load_fractions.copy()
        self._swing_start = None
        self._swing_end = None
        self._landed_contact_seconds = 0.
        forward = -self.root_rotation[:2, 1]
        self.forward = forward / max(np.linalg.norm(forward), 1e-9)
        lateral = self.root_rotation[:2, 0]
        self._lane_axis = lateral / max(np.linalg.norm(lateral), 1e-9)
        widths = [sum(abs(self._lane_axis @ rotation[:2, axis]) * self.model.geom_size[geom, axis]
                      for axis in (0, 1)) for (_, rotation), geom in zip(self.anchors, self._foot_geoms)]
        self._minimum_lane_gap = sum(widths) + .025
        self._lane_gap = max(self._minimum_lane_gap + .025,
                             float(self._lane_axis @ (self.anchors[0][0][:2] - self.anchors[1][0][:2])))
        self._final_balance_start = None

    def start_walk(self, goal=None):
        if goal is not None:
            goal = np.asarray(goal, dtype=float)
            if goal.shape != (3,) or not np.isfinite(goal).all():
                raise ValueError("goal must be a finite Z-up three-vector")
            self.env.goal = goal.copy()
        self.walking = True
        self.ik_posture_weight = self.ik_posture_weight or 3.
        self.phase = "settling"
        self._phase_started = float(self.data.time)

    def _begin_transfer(self, support):
        self._support, self._swing = support, 1 - support
        for index in range(2):
            # Position follows measured touchdown only; a stance is world-fixed.
            self.anchors[index] = (self._foot_state(index)[0], self.anchors[index][1])
        self.active_feet = [True, True]
        self.swing_goal = None
        self._transfer_start = self.data.subtree_com[self.env.pelvis_body_id].copy()
        self._transfer_goal = self.anchors[support][0].copy()
        self._transfer_goal[2] = self._neutral_com_height - self.com_height_offset
        self._transfer_duration = self.transfer_seconds
        self._transfer_load_start = self.load_fractions.copy()
        self._stable_seconds = 0.
        self.phase = "transfer"
        self._phase_started = float(self.data.time)

    def _begin_swing(self):
        self.active_feet[self._swing] = False
        self._swing_start = self._foot_state(self._swing)[0]
        self._landed_contact_seconds = 0.
        self._swing_end = self._swing_start.copy()
        if self.goal_direction:
            target = self._goal_foot_targets()[self._swing]
            delta = target - self._swing_start[:2]
            self._swing_end[:2] += delta * min(1., self.stride / max(np.linalg.norm(delta), 1e-9))
            # Keep separate sole lanes throughout the interpolated swing.
            # The closed half-plane is convex, so safe endpoints stay safe.
            sign = 1 if self._swing == 0 else -1
            gap = sign * float(self._lane_axis @ (self._swing_end[:2] - self.anchors[self._support][0][:2]))
            if gap < self._minimum_lane_gap:
                self._swing_end[:2] += sign * self._lane_axis * (self._minimum_lane_gap - gap)
            rotation = self.data.xmat[self.env.pelvis_body_id].reshape(3, 3)
            hip_offset = self._hip_offsets[self._swing] if self._profiled_body else [sign * .10, 0., -.02]
            hip = self.data.qpos[:3] + rotation @ hip_offset
            foot_rotation = self.anchors[self._swing][1]
            endpoint = self._swing_end.copy()
            endpoint[2] = self.env.floor_height + .0005
            ankle_offset = foot_rotation @ (self._ankle_from_sole[self._swing] if self._profiled_body else [0., .065, .08])
            # Leave 15 mm of extension margin in the two 415 mm segments.
            # A nearly straight swing leg can touch without carrying load.
            reach_limit = sum(self._leg_lengths[self._swing]) - .015 if self._profiled_body else .815
            if np.linalg.norm(endpoint + ankle_offset - hip) > reach_limit:
                lower, upper = 0., 1.
                for _ in range(30):
                    fraction = .5 * (lower + upper)
                    candidate = self._swing_start + fraction * (endpoint - self._swing_start)
                    if np.linalg.norm(candidate + ankle_offset - hip) <= reach_limit:
                        lower = fraction
                    else:
                        upper = fraction
                self._swing_end[:2] = self._swing_start[:2] + lower * (endpoint[:2] - self._swing_start[:2])
        else:
            support_progress = float(self.forward @ self.anchors[self._support][0][:2])
            sole_bias = self._ankle_from_sole[self._swing][1] if self._profiled_body else .065
            goal_progress = float(self.forward @ self.env.goal[:2]) - sole_bias
            end_progress = min(support_progress + self.stride, goal_progress)
            self._swing_end[:2] += self.forward * (end_progress - float(self.forward @ self._swing_end[:2]))
        self._swing_end[2] = self.env.floor_height + .0005
        self.swing_goal = self._swing_start.copy()
        self.load_fractions[:] = 0
        self.load_fractions[self._support] = 1
        self.phase = "swing"
        self._phase_started = float(self.data.time)

    def _goal_foot_targets(self):
        centre = self.env.goal[:2] + self._neutral_com_offset[:2]
        return [centre + sign * .5 * self._lane_gap * self._lane_axis for sign in (1, -1)]

    def _goal_support(self):
        targets = self._goal_foot_targets()
        errors = [float(np.linalg.norm(target - anchor[0][:2])) for target, anchor in zip(targets, self.anchors)]
        delta = self.env.goal[:2] - self.data.qpos[:2]
        # On sideways approach, move the outward foot before its follower.
        outward = 0 if self._lane_axis @ delta >= 0 else 1
        swing = outward if abs(errors[0] - errors[1]) < .03 else int(np.argmax(errors))
        other = 1 - swing
        sign = 1 if swing == 0 else -1
        gap = sign * float(self._lane_axis @ (targets[swing] - self.anchors[other][0][:2]))
        if gap < self._minimum_lane_gap:
            safe_target = targets[swing] + sign * self._lane_axis * (self._minimum_lane_gap - gap)
            if np.linalg.norm(safe_target - self.anchors[swing][0][:2]) < .03:
                swing = other
        return 1 - swing

    def _begin_final_balance(self):
        for index in range(2):
            self.anchors[index] = (self._foot_state(index)[0], self.anchors[index][1])
        self.active_feet[:] = [True, True]
        self.swing_goal = None
        self.load_fractions[:] = .5
        self._final_balance_start = self.data.subtree_com[self.env.pelvis_body_id].copy()
        distance = float(np.linalg.norm(self.env.goal[:2] + self._neutral_com_offset[:2] - self._final_balance_start[:2]))
        self._balance_duration = max(1., math.sqrt(6 * distance / .35))
        self.phase = "final_balance"
        self._phase_started = float(self.data.time)
        self._stable_seconds = 0.

    def _goal_inside_support(self):
        corners = []
        for index, geom in enumerate(self._foot_geoms):
            sole, rotation, _, _ = self._foot_state(index)
            for x in (-1, 1):
                for y in (-1, 1):
                    corners.append((sole + rotation @ [x * self.model.geom_size[geom, 0], y * self.model.geom_size[geom, 1], 0.])[:2])
        equations = ConvexHull(np.asarray(corners)).equations
        goal = self.env.goal[:2] + self._neutral_com_offset[:2]
        return bool(np.max(equations[:, :2] @ goal + equations[:, 2]) < -.015)

    def _update_plan(self):
        if not self.walking or self.failure:
            return
        now = float(self.data.time)
        elapsed = now - self._phase_started
        info = self.env._info()
        forces = np.asarray([info["left_foot_force"], info["right_foot_force"]])
        if self.phase == "settling":
            if elapsed >= .6 and forces.min() > .2 * self.weight:
                self._begin_transfer(self._goal_support() if self.goal_direction else 1)
            return
        if self.phase == "transfer":
            fraction = float(np.clip(elapsed / self._transfer_duration, 0, 1))
            blend = fraction * fraction * (3 - 2 * fraction)
            self.com_goal = self._transfer_start + blend * (self._transfer_goal - self._transfer_start)
            target_load = np.zeros(2)
            target_load[self._support] = 1
            self.load_fractions = self._transfer_load_start + blend * (target_load - self._transfer_load_start)
            for side in ("left", "right"):
                for joint, target in (("hip_pitch", -.15), ("knee", .30), ("ankle_pitch", -.15)):
                    index = JOINT_NAMES.index(f"{side}_{joint}")
                    self.posture[index] = self.env.standing_pose[index] + blend * (target - self.env.standing_pose[index])
            com, jac, _ = self._com_state()
            sole, rotation, _, _ = self._foot_state(self._support)
            local_offset = rotation.T @ (com - sole)
            ready = (fraction >= 1 and abs(local_offset[0]) < .025 and abs(local_offset[1]) < .05
                     and np.linalg.norm((jac @ self.data.qvel)[:2]) < .12
                     and forces[self._swing] < .08 * self.weight and forces[self._support] > .6 * self.weight)
            if (self.goal_direction and info["goal_distance"] < .20 and forces.min() > 50
                    and info["upright"] > .95 and self._goal_inside_support()):
                self._begin_final_balance()
                return
            self._stable_seconds = self._stable_seconds + self.env.dt if ready else 0.
            if self._stable_seconds >= .08:
                self._begin_swing()
            elif elapsed > 4:
                self.failure = "Measured COM/load transfer did not establish single support"
            return
        if self.phase == "swing":
            u = float(np.clip(elapsed / self.swing_seconds, 0, 1))
            blend = u**3 * (10 - 15 * u + 6 * u * u)
            velocity = 30 * u * u * (1 - u)**2 / self.swing_seconds
            acceleration = 60 * u * (1 - u) * (1 - 2 * u) / self.swing_seconds**2
            difference = self._swing_end - self._swing_start
            self.swing_goal = self._swing_start + blend * difference
            self.swing_velocity = velocity * difference
            self.swing_acceleration = acceleration * difference
            self.swing_goal[2] += self.swing_height * 64 * u**3 * (1 - u)**3
            self.swing_velocity[2] += self.swing_height * 192 * u * u * (1 - u)**2 * (1 - 2 * u) / self.swing_seconds
            self.swing_acceleration[2] += self.swing_height * 384 * u * (1 - u) * (1 - 5 * u + 5 * u * u) / self.swing_seconds**2
            if elapsed > self.swing_seconds:
                # Seek actual load through compliant contact, never move the
                # physical foot or mark an unloaded proximity as a touchdown.
                self.swing_goal[2] -= min(.006, .02 * (elapsed - self.swing_seconds))
            sole, _, foot_jac, _ = self._foot_state(self._swing)
            minimum_force = 1. if self.goal_direction else 2.
            measured_contact = info[f"{('left', 'right')[self._swing]}_foot_contact"]
            settled_contact = (elapsed > self.swing_seconds + .3 and forces[self._swing] > minimum_force
                               and (not self.goal_direction or measured_contact)
                               and sole[2] - self.env.floor_height < .006
                               and np.linalg.norm(sole[:2] - self._swing_end[:2]) < .025
                               and np.linalg.norm(foot_jac[:3] @ self.data.qvel) < .04)
            self._landed_contact_seconds = self._landed_contact_seconds + self.env.dt if settled_contact else 0.
            landed = ((u > .85 and forces[self._swing] > 10 and sole[2] - self.env.floor_height < .012
                       and np.linalg.norm(sole[:2] - self._swing_end[:2]) < .04)
                      or self._landed_contact_seconds >= .12)
            if landed:
                self.completed_steps += 1
                self.active_feet[:] = [True, True]
                self.anchors[self._swing] = (sole, self.anchors[self._swing][1])
                self.swing_goal = None
                self.swing_velocity[:] = 0
                self.swing_acceleration[:] = 0
                self.phase = "touchdown"
                self._stable_seconds = 0.
                self._phase_started = now
            elif elapsed > self.swing_seconds + 1.2:
                self.failure = "Swing foot did not establish a measured touchdown"
            return
        if self.phase == "touchdown":
            fraction = float(np.clip(elapsed / .25, 0, 1))
            self.load_fractions = np.zeros(2)
            self.load_fractions[self._support] = 1 - .5 * fraction
            self.load_fractions[self._swing] = .5 * fraction
            if fraction >= 1:
                if self.goal_direction:
                    targets = self._goal_foot_targets()
                    close_pair = max(np.linalg.norm(target - anchor[0][:2]) for target, anchor in zip(targets, self.anchors)) < .055
                    if close_pair or (info["goal_distance"] < .20 and forces.min() > 50 and self._goal_inside_support()):
                        self._begin_final_balance()
                    else:
                        self._begin_transfer(self._goal_support())
                    return
                if np.linalg.norm(self.data.qpos[:2] - self.env.goal[:2]) < self.goal_tolerance:
                    self.phase = "done"
                    self.com_goal[:2] = (self.anchors[0][0][:2] + self.anchors[1][0][:2]) / 2
                else:
                    self._begin_transfer(self._swing)
        if self.phase == "final_balance":
            fraction = float(np.clip(elapsed / self._balance_duration, 0, 1))
            blend = fraction * fraction * (3 - 2 * fraction)
            target = self._final_balance_start.copy()
            target[:2] = self.env.goal[:2] + self._neutral_com_offset[:2]
            self.com_goal = self._final_balance_start + blend * (target - self._final_balance_start)
            ready = (np.linalg.norm(self.data.qpos[:2] - self.env.goal[:2]) < self.goal_tolerance
                     and np.linalg.norm(self.data.qvel[:3]) < .15
                     and info["upright"] > .95 and forces.min() > 50)
            self._stable_seconds = self._stable_seconds + self.env.dt if ready else 0.
            if fraction >= 1 and self._stable_seconds >= .5:
                self.phase = "done"
            elif elapsed > 5:
                self.failure = "Final goal balance did not establish measured two-foot support"

    def _foot_state(self, index):
        geom = self._foot_geoms[index]
        rotation = self.data.geom_xmat[geom].reshape(3, 3).copy()
        sole = self.data.geom_xpos[geom] - rotation[:, 2] * self.model.geom_size[geom, 2]
        jacp, jacr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        dotp, dotr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, sole, self._foot_bodies[index])
        mujoco.mj_jacDot(self.model, self.data, dotp, dotr, sole, self._foot_bodies[index])
        return sole.copy(), rotation, np.vstack((jacp, jacr)), np.vstack((dotp, dotr)) @ self.data.qvel

    def _com_state(self):
        jac = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac, self.env.pelvis_body_id)
        derivative = np.zeros(3)
        for body in self._body_ids:
            dot = np.zeros((3, self.nv))
            mujoco.mj_jacDot(self.model, self.data, dot, None, self.data.xipos[body], body)
            derivative += self.model.body_mass[body] / self.mass * (dot @ self.data.qvel)
        return self.data.subtree_com[self.env.pelvis_body_id].copy(), jac, derivative

    def _anchored_posture(self):
        """FK scratch targets only; this never writes the simulated pose.

        Give the optimizer a bent-knee upright solution of the same foot/COM
        task. The independent reference MjData has no role in integration.
        """
        reference = self._reference_data
        reference.qpos[:] = self.model.qpos0
        reference.qpos[self.env._qpos_indices] = self.env.standing_pose
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, self.root_rotation.ravel())
        reference.qpos[3:7] = quaternion
        position = self.com_goal - self._neutral_com_offset
        for _ in range(5):
            reference.qpos[:3] = position
            for index, sign in enumerate((1, -1)):
                rotation = self.anchors[index][1]
                sole = self.anchors[index][0]
                if not self.active_feet[index] and self.swing_goal is not None:
                    sole = self.swing_goal
                ankle = sole + rotation @ (self._ankle_from_sole[index] if self._profiled_body else [0., .065, .08])
                vector = self.root_rotation.T @ (ankle - position) - (self._hip_offsets[index] if self._profiled_body else [sign * .10, 0., -.02])
                if self._profiled_body:
                    upper, lower = self._leg_lengths[index]
                    length = float(np.clip(np.linalg.norm(vector), .25, upper+lower-.002))
                    knee = math.acos(float(np.clip((length**2-upper**2-lower**2)/(2*upper*lower), -1, 1)))
                    down = upper+lower*math.cos(knee)
                else:
                    lower = .415
                    length = float(np.clip(np.linalg.norm(vector), .25, .828))
                    knee = math.acos(float(np.clip((length * length - 2 * .415**2) / (2 * .415**2), -1, 1)))
                    down = .415 * (1 + math.cos(knee))
                roll = math.asin(float(np.clip(-vector[0] / max(down, 1e-8), -.45, .45)))
                pitch = math.atan2(vector[1], -vector[2]) - math.atan2(lower * math.sin(knee), down * math.cos(roll))
                parent = (self.root_rotation @ Rotation.from_rotvec([pitch, 0., 0.]).as_matrix()
                          @ Rotation.from_rotvec([0., roll, 0.]).as_matrix()
                          @ Rotation.from_rotvec([knee, 0., 0.]).as_matrix())
                ankle_rotation = parent.T @ rotation
                ankle_pitch = math.atan2(-ankle_rotation[1, 2], ankle_rotation[2, 2])
                ankle_roll = math.atan2(ankle_rotation[0, 2], math.hypot(ankle_rotation[1, 2], ankle_rotation[2, 2]))
                values = [pitch, roll, 0., knee, ankle_pitch, ankle_roll]
                slots = slice(2 + index * 6, 8 + index * 6)
                reference.qpos[self.env._qpos_indices[slots]] = np.clip(values, JOINT_LIMITS[slots, 0], JOINT_LIMITS[slots, 1])
            mujoco.mj_kinematics(self.model, reference)
            mujoco.mj_comPos(self.model, reference)
            error = self.com_goal - reference.subtree_com[self.env.pelvis_body_id]
            position += error
            if np.linalg.norm(error) < 1e-4:
                break
        return reference.qpos[self.env._qpos_indices].copy()

    def compute_torques(self):
        """Solve free-root dynamics and bounded contact forces, without moving it."""
        mujoco.mj_forward(self.model, self.data)
        mass = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, self.data, mass)
        bias = self.data.qfrc_bias - self.data.qfrc_passive
        feet = [self._foot_state(index) for index in range(2)]
        size = self.nv + self.nu + 12
        tau_slice = slice(self.nv, self.nv + self.nu)
        wrench_start = self.nv + self.nu
        diagonal = np.r_[np.full(self.nv, 1e-6), np.full(self.nu, 1e-4), np.full(12, 1e-6)]
        hessian, gradient = np.diag(diagonal), np.zeros(size)
        rows, lower, upper = [], [], []

        def task(jac, desired, strength):
            augmented = np.zeros((jac.shape[0], size))
            augmented[:, :self.nv] = jac
            nonlocal hessian, gradient
            hessian += strength * augmented.T @ augmented
            gradient -= strength * augmented.T @ desired

        def constrain(matrix, low, high):
            rows.extend(matrix)
            lower.extend(np.atleast_1d(low))
            upper.extend(np.atleast_1d(high))

        dynamics = np.zeros((self.nv, size))
        dynamics[:, :self.nv] = mass
        dynamics[:, tau_slice] = -self._selection.T
        for index, (_, _, jac, _) in enumerate(feet):
            dynamics[:, wrench_start + 6 * index:wrench_start + 6 * (index + 1)] = -jac.T
        constrain(dynamics, -bias, -bias)
        torque_bounds = np.zeros((self.nu, size))
        torque_bounds[:, tau_slice] = np.eye(self.nu)
        constrain(torque_bounds, -TORQUE_LIMITS, TORQUE_LIMITS)

        for index, (sole, rotation, jac, derivative) in enumerate(feet):
            start = wrench_start + 6 * index
            if not self.active_feet[index]:
                inactive = np.zeros((6, size))
                inactive[:, start:start + 6] = np.eye(6)
                constrain(inactive, np.zeros(6), np.zeros(6))
                if self.swing_goal is not None:
                    velocity = jac[:3] @ self.data.qvel
                    desired = (self.swing_acceleration + 150 * (self.swing_goal - sole)
                               + 25 * (self.swing_velocity - velocity) - derivative[:3])
                    task(jac[:3], desired, 500)
                    angular = 80 * _rotation_error(self.anchors[index][1], rotation) - 18 * (jac[3:] @ self.data.qvel) - derivative[3:]
                    task(jac[3:], angular, 150)
                continue
            anchor, target_rotation = self.anchors[index]
            velocity = jac @ self.data.qvel
            desired = np.r_[200 * (anchor - sole), 100 * _rotation_error(target_rotation, rotation)] - 30 * velocity - derivative
            # A soft stance task tolerates real contact compliance and finite torques.
            task(jac, desired, self.stance_weight)
            normal = np.zeros((1, size))
            normal[0, start + 2] = 1
            constrain(normal, 0., 1200.)
            for component, coefficient in ((0, .7), (1, .7), (3, .12), (4, .05), (5, .02)):
                # CoP moment limits use the actual sole's heading coordinates.
                axis = self.anchors[index][1][:, component - 3] if component >= 3 else np.eye(3)[:, component]
                offset = 3 if component >= 3 else 0
                for sign in (-1, 1):
                    row = np.zeros((1, size))
                    row[0, start + offset:start + offset + 3] = sign * axis
                    row[0, start + 2] -= coefficient
                    constrain(row, -np.inf, 0.)
            force_target = self.weight * self.load_fractions[index]
            hessian[start + 2, start + 2] += .001
            gradient[start + 2] -= .001 * force_target

        com, com_jac, com_derivative = self._com_state()
        task(com_jac, 40 * (self.com_goal - com) - 12 * (com_jac @ self.data.qvel) - com_derivative, self.com_weight)
        root_jac, root_dot = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        root_body = self.env.pelvis_body_id
        mujoco.mj_jac(self.model, self.data, None, root_jac, self.data.xpos[root_body], root_body)
        mujoco.mj_jacDot(self.model, self.data, None, root_dot, self.data.xpos[root_body], root_body)
        root_rotation = self.data.xmat[root_body].reshape(3, 3)
        root_desired = 80 * _rotation_error(self.root_rotation, root_rotation) - 18 * (root_jac @ self.data.qvel) - root_dot @ self.data.qvel
        task(root_jac, root_desired, self.orientation_weight)
        target_posture = self._anchored_posture() if self.ik_posture_weight else self.posture
        posture_error = np.clip(target_posture, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]) - self.data.qpos[self.env._qpos_indices]
        task(self._selection, 35 * posture_error - 10 * self.data.qvel[self._actuated], .3)
        if self.ik_posture_weight:
            task(self._selection, 35 * posture_error - 10 * self.data.qvel[self._actuated], self.ik_posture_weight)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(2 * np.triu(hessian)), q=2 * gradient,
                     A=sparse.csc_matrix(np.asarray(rows)), l=np.asarray(lower), u=np.asarray(upper),
                     verbose=False, eps_abs=2e-5, eps_rel=2e-5, max_iter=6000, polishing=True)
        if self._previous_solution is not None:
            solver.warm_start(x=self._previous_solution)
        solution = solver.solve(raise_error=False)
        solved = solution.info.status_val in (1, 2) and solution.x is not None and np.isfinite(solution.x).all()
        self.diagnostics = {"qp_solved": bool(solved), "qp_status": solution.info.status,
                            "qp_iterations": int(solution.info.iter), "controller_phase": self.phase,
                            "body_com": com.tolist(), "com_goal": self.com_goal.tolist(),
                            "active_feet": list(self.active_feet), "completed_steps": self.completed_steps,
                            "goal_direction": self.goal_direction}
        if self._swing_end is not None:
            self.diagnostics["swing_target"] = self._swing_end.tolist()
            self.diagnostics["swing_target_error"] = float(np.linalg.norm(feet[self._swing][0] - self._swing_end))
        if solved:
            self._previous_solution = solution.x.copy()
            torque = np.clip(solution.x[tau_slice], -TORQUE_LIMITS, TORQUE_LIMITS)
            self.diagnostics["estimated_sole_wrenches"] = solution.x[wrench_start:].reshape(2, 6).tolist()
            self.diagnostics["dynamics_residual"] = float(np.max(np.abs(dynamics @ solution.x + bias)))
            self.diagnostics["predicted_com_acceleration"] = (com_jac @ solution.x[:self.nv] + com_derivative).tolist()
            self.diagnostics["root_orientation_error"] = float(np.linalg.norm(_rotation_error(self.root_rotation, root_rotation)))
            self.diagnostics["stance_position_errors"] = [float(np.linalg.norm(self.anchors[index][0] - feet[index][0])) for index in range(2)]
        else:
            self._qp_failures += 1
            self.failure = f"Inverse dynamics QP failed: {solution.info.status}"
            torque = np.zeros(self.nu) if self._previous_solution is None else np.clip(self._previous_solution[tau_slice], -TORQUE_LIMITS, TORQUE_LIMITS)
        self.diagnostics["qp_failures"] = self._qp_failures
        return torque

    def step(self):
        if self.env._episode_done:
            raise RuntimeError("reset() is required after a completed episode")
        self._update_plan()
        torque = self.compute_torques()
        self.env._last_torque[:] = torque
        for _ in range(CONTROL_SUBSTEPS):
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
        self.env._step_count += 1
        mujoco.mj_forward(self.model, self.data)
        info = self.env._info()
        if self.env.task == "stand":
            successful = (info["upright"] > .94 and abs(info["root_height"] - self.env.standing_root_height) < .10
                          and np.linalg.norm(self.data.qvel[:3]) < .30
                          and info["left_foot_contact"] and info["right_foot_contact"])
        else:
            successful = (self.phase == "done" and info["goal_distance"] < .25
                          and np.linalg.norm(self.data.qvel[:3]) < .20 and info["upright"] > .94
                          and info["left_foot_contact"] and info["right_foot_contact"])
        self.env._success_seconds = self.env._success_seconds + self.env.dt if successful else 0.
        info["success_seconds"] = self.env._success_seconds
        info["success"] = self.env._success_seconds >= .5
        info.update(self.diagnostics)
        info["body_com"] = self.data.subtree_com[self.env.pelvis_body_id].tolist()
        info["controller"] = "Whole-body inverse dynamics QP (not RL)"
        info["wbc_failure"] = self.failure
        self.env._episode_done = bool(info["fall"] or self.env._step_count >= self.env.max_steps)
        return info

    def snapshot(self):
        snapshot = self.env.snapshot()
        snapshot["stats"].update(self.diagnostics)
        snapshot["stats"]["body_com"] = self.data.subtree_com[self.env.pelvis_body_id].tolist()
        snapshot["stats"]["controller"] = "Whole-body inverse dynamics QP (not RL)"
        snapshot["stats"]["max_external_body_force"] = float(np.max(np.abs(self.data.xfrc_applied)))
        snapshot["stats"]["max_external_generalized_force"] = float(np.max(np.abs(self.data.qfrc_applied)))
        snapshot["stats"]["wbc_failure"] = self.failure
        foot_force = 0.
        foot_distances = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if {int(contact.geom1), int(contact.geom2)} == set(self._foot_geoms):
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                foot_force += max(0., float(force[0]))
                foot_distances.append(float(contact.dist))
        snapshot["stats"]["foot_pair_contact_force"] = foot_force
        snapshot["stats"]["foot_pair_contact_count"] = len(foot_distances)
        snapshot["stats"]["foot_pair_min_distance"] = min(foot_distances, default=None)
        return snapshot
