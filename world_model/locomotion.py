"""Torque-driven footsteps using T2M timing and stride as a physical prior.

The original T2M pose has asymmetric support timing on this control rig. This
prior adapts the legs with two-link IK targets. IK is never written to qpos;
all motion still comes from bounded joint torques and contact dynamics.
"""

from __future__ import annotations

import math
import numpy as np

from .humanoid import JOINT_NAMES
from .imitation import ReferenceHumanoidEnv


def rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def solve_leg(vector, root_rotation, foot_rotation, upper=.415, lower=.415):
    """Hip Rx,Ry and knee Rx; ankle Rx,Ry aligns the sole with the floor."""
    if not math.isfinite(upper) or not math.isfinite(lower) or min(upper, lower) <= .002:
        raise ValueError("Leg segments must have finite positive metre lengths")
    # Preserve the original generic rig's inner reach and use the physical
    # triangle bound as well for unequal measured segments.
    inner_reach = max(min(.25, upper+lower-.003), abs(upper-lower)+.002)
    length = float(np.clip(np.linalg.norm(vector), inner_reach, upper+lower-.002))
    knee = math.acos(float(np.clip((length**2-upper**2-lower**2)/(2*upper*lower), -1, 1)))
    down = upper+lower*math.cos(knee)
    roll = math.asin(float(np.clip(-vector[0]/max(down, 1e-8), -.45, .45)))
    pitch = math.atan2(vector[1], -vector[2])-math.atan2(lower*math.sin(knee), down*math.cos(roll))
    parent = root_rotation @ rotation_x(pitch) @ rotation_y(roll) @ rotation_x(knee)
    ankle = parent.T @ foot_rotation
    ankle_pitch = math.atan2(-ankle[1, 2], ankle[2, 2])
    ankle_roll = math.atan2(ankle[0, 2], math.hypot(ankle[1, 2], ankle[2, 2]))
    return pitch, roll, knee, ankle_pitch, ankle_roll


class FootstepHumanoidEnv(ReferenceHumanoidEnv):
    """Standing starts; world root is wholly owned by MuJoCo dynamics."""

    def __init__(self, *, strict_gait_reward=False, **kwargs):
        self.strict_gait_reward = strict_gait_reward
        self._swing_durations = np.zeros(2)
        self._swing_counted = np.zeros(2, dtype=bool)
        self._swing_counts = np.zeros(2, dtype=int)
        kwargs.setdefault("task", "walk")
        kwargs.setdefault("reference_start", False)
        if kwargs["task"] != "walk":
            raise ValueError("Footstep prior is only for walking")
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._swing_durations[:] = 0
        self._swing_counted[:] = False
        self._swing_counts[:] = 0
        return super().reset(**kwargs)

    def _reference_pose(self):
        sample = self.reference.sample(float(self.data.time)+self.phase_offset)
        pose = sample.joint_targets.copy()
        matrix = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        heading = math.atan2(matrix[1, 0], matrix[0, 0])
        c, s = math.cos(heading), math.sin(heading)
        horizontal = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        ramp = min(self.data.time/.8, 1.)
        ramp = ramp*ramp*(3-2*ramp)
        forward = horizontal[:2, :2] @ [0., -1.]
        remaining = float(np.dot(self.goal[:2]-self.data.qpos[:2], forward))
        # Signed progress prevents restarting a forward gait after overshoot.
        near = float(np.clip((remaining-.08)/.65, 0, 1))
        weight = ramp*near
        # T2M keeps its native 1.12s cycle. A smaller physical stride gives
        # finite contact transitions during acceleration from standing.
        step_length = float(np.linalg.norm(self.reference.stride))*.46
        virtual_pelvis = self.data.qpos[:3].copy()
        virtual_pelvis[2] = self.floor_height + (
            .90 if self.avatar_profile is None else self.standing_root_height-.03*self._body_scale)
        for side, sign, phase in (("left", 1, sample.phase), ("right", -1, (sample.phase+.5)%1)):
            if phase < .4:
                fraction = phase/.4
                blend = fraction*fraction*(3-2*fraction)
                longitudinal = step_length*(.5-blend)
                lift = .075*math.sin(math.pi*fraction)
            else:
                longitudinal = step_length*(-.5+(phase-.4)/.6)
                lift = 0.
            if self.avatar_profile is None:
                foot_center = np.asarray([sign*.10, -.065+longitudinal*weight,
                                          self.floor_height+.08+lift*weight])
                world_ankle = horizontal @ [foot_center[0], foot_center[1]+.065, 0.]
                world_ankle[:2] += self.data.qpos[:2]
                world_ankle[2] = foot_center[2]
                hip_offset = np.asarray([sign*.10, 0, -.02])
                upper = lower = .415
            else:
                # Read the compiled body, rather than a second set of fixed
                # avatar constants. The centre-to-ankle shift is expressed in
                # the desired flat foot frame, including its X/Y shoe offset.
                hip_offset = self.model.body_pos[self.model.body(f"{side}_thigh").id]
                upper = float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_shank").id]))
                lower = float(np.linalg.norm(self.model.body_pos[self.model.body(f"{side}_foot").id]))
                foot_id = self.model.geom(f"humanoid_{side}_foot").id
                offset = self.model.geom_pos[foot_id]
                foot_center = horizontal @ [hip_offset[0]+offset[0],
                                            hip_offset[1]+offset[1]+longitudinal*weight, 0.]
                foot_center[:2] += self.data.qpos[:2]
                foot_center[2] = self.floor_height+self.model.geom_size[foot_id, 2]+lift*weight
                world_ankle = foot_center-horizontal @ offset
            vector = matrix.T @ (world_ankle-virtual_pelvis)-hip_offset
            hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll = solve_leg(
                vector, matrix, horizontal, upper=upper, lower=lower)
            for joint, value in (("hip_pitch", hip_pitch), ("hip_roll", hip_roll), ("hip_yaw", 0.),
                                 ("knee", knee), ("ankle_pitch", ankle_pitch), ("ankle_roll", ankle_roll)):
                pose[JOINT_NAMES.index(f"{side}_{joint}")] = value
        pose[0] = .05*weight
        pose[1] = 0
        pose = self.standing_pose+weight*(pose-self.standing_pose)
        # Joint balance stays active at a goal and after the gait amplitude
        # fades. Multiplying it by weight made a stopped body unstable.
        local_up = matrix.T @ [0., 0., 1.]
        pitch = math.atan2(local_up[1], local_up[2])
        roll = math.atan2(-local_up[0], local_up[2])
        desired_roll = -.065*math.sin(2*math.pi*sample.phase)*weight
        for side in ("left", "right"):
            pose[JOINT_NAMES.index(f"{side}_ankle_pitch")] += 2*(pitch-.03*weight)+.3*self.data.qvel[3]
            pose[JOINT_NAMES.index(f"{side}_ankle_roll")] += 2*(roll-desired_roll)+.3*self.data.qvel[4]
        return pose

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["physical_gait_prior"] = "T2M timing/stride + two-link joint IK + bounded torque PD"
        if self.strict_gait_reward:
            # The earlier distance reward admitted a sliding shortcut. Use
            # solver support forces and the lowest actual sole corner, never
            # reference positions, to reward unloading and repeated swings.
            forces = np.asarray([info[f"{side}_foot_force"] for side in ("left", "right")])
            clearances = np.asarray(info["foot_clearances"])
            phase = info["reference_phase"]
            distance = info["goal_distance"]
            gait_weight = float(np.clip((distance-.18)/.45, 0, 1))
            swing_score = slip = events = 0.
            from .contact_metrics import measure_foot_contact_slip
            contact_metrics = measure_foot_contact_slip(self.model, self.data, self._foot_geom_ids)
            for index, side_phase in enumerate((phase, (phase+.5)%1)):
                other = 1-index
                valid = (self.data.time >= .5 and clearances[index] > .025
                         and forces[index] < 20 and forces[other] >= 20)
                if valid:
                    self._swing_durations[index] += self.dt
                    if self._swing_durations[index] > .08 and not self._swing_counted[index]:
                        self._swing_counted[index] = True
                        self._swing_counts[index] += 1
                        events += 1
                else:
                    self._swing_durations[index] = 0
                    self._swing_counted[index] = False
                if .08 < side_phase < .32:
                    lift = float(np.clip(clearances[index]/.055, 0, 1))
                    unloaded = math.exp(-forces[index]/40)
                    supported = float(np.clip(forces[other]/300, 0, 1))
                    swing_score += lift*unloaded*supported
                side = ("left", "right")[index]
                slip += min(forces[index]/300, 1)*min(contact_metrics[side]["weightedSlipSpeedMetersPerSecond"], 1)
            completed = min(1., float(self._swing_counts.min())/2)
            # Arrival credit is earned after both legs actually swing twice.
            reward -= 3*math.exp(-(distance/.35)**2)*(1-completed)
            reward += 8*gait_weight*swing_score + 12*events - 4*gait_weight*slip
            info.update(strict_gait_reward=True, actual_swing_counts=self._swing_counts.tolist(),
                        support_swing_reward=swing_score, loaded_foot_slip=slip)
        return observation, reward, terminated, truncated, info
