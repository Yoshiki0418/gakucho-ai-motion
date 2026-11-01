"""Physics control with a T2M joint reference, never a commanded root path."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .environment import HumanoidRoomEnv
from .humanoid import JOINT_NAMES
from .references import load_motion_reference


class ReferenceHumanoidEnv(HumanoidRoomEnv):
    """A reference-start gait is evaluated separately from starting to walk.

    reference_start sets an initial state during reset, just as a motion tracking
    simulator does. Neither reset mode writes the floating root during step.
    """

    def __init__(self, *, reference_start=True, walking_distance_range=None, **kwargs):
        task = kwargs.get("task", "walk")
        if task not in ("walk", "sit"):
            raise ValueError("Reference task must be walk or sit")
        self.reference = load_motion_reference(task, avatar_profile=kwargs.get("avatar_profile"))
        self.reference_start = reference_start
        self.walking_distance_range = walking_distance_range
        self.phase_offset = 0.
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        self.phase_offset = 0.
        observation, info = super().reset(seed=seed, options=options)
        if self.task == "walk" and self.walking_distance_range:
            distance = self.np_random.uniform(*self.walking_distance_range)
            self.goal = self.spawn + [math.sin(self.spawn_heading)*distance,
                                      -math.cos(self.spawn_heading)*distance, 0.]
        self.phase_offset = self.np_random.uniform(0, self.reference.duration) if self.task == "walk" and self.reference_start else 0.
        if self.reference_start:
            sample = self.reference.sample(self.phase_offset)
            self.data.qpos[:3] = self.spawn + [0, 0, sample.root_position[2]]
            heading = np.asarray([math.cos(self.spawn_heading/2), 0., 0., math.sin(self.spawn_heading/2)])
            mujoco.mju_mulQuat(self.data.qpos[3:7], heading, sample.root_quaternion)
            self.data.qpos[self._qpos_indices] = sample.joint_targets
            c, s = math.cos(self.spawn_heading), math.sin(self.spawn_heading)
            rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.data.qvel[:3] = rotation @ sample.root_velocity
            before = self.reference.sample(max(0, self.phase_offset-.005))
            after = self.reference.sample(self.phase_offset+.005)
            derivative_dt = .01 if self.phase_offset >= .005 else .005+self.phase_offset
            self.data.qvel[self._qvel_indices] = (after.joint_targets-before.joint_targets)/derivative_dt
            mujoco.mj_forward(self.model, self.data)
            self._previous_goal_distance = self._goal_distance()
            info = self._info()
            observation = self._observation(info)
        self._previous_goal_distance = self._goal_distance()
        info = self._info()
        return self._observation(info), info

    def _reference_pose(self):
        reference = self.reference.sample(float(self.data.time)+self.phase_offset)
        pose = reference.joint_targets.copy()
        if not self.reference_start:
            weight = min(self.data.time/.8, 1.)
            weight = weight*weight*(3-2*weight)
            pose = self.standing_pose + weight*(pose-self.standing_pose)
        if self.task == "walk":
            # Decelerate near a goal by relaxing the gait towards standing.
            weight = float(np.clip((self._goal_distance()-.18)/.45, 0, 1))
            # Disclosed physical adaptation of the T2M prior: symmetric swing
            # clearance. This changes joint targets, never body/root placement.
            for side, phase in (("left", reference.phase), ("right", reference.phase+.5)):
                swing = max(0., math.sin(2*math.pi*phase))
                for joint, amount in (("hip_pitch", -.175), ("knee", .35), ("ankle_pitch", -.175)):
                    pose[JOINT_NAMES.index(f"{side}_{joint}")] += swing*amount
            pose = self.standing_pose + weight*(pose-self.standing_pose)
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        local_up = rotation.T @ [0., 0., 1.]
        pitch = math.atan2(local_up[1], local_up[2])
        roll = math.atan2(-local_up[0], local_up[2])
        desired_pitch = .03*weight if self.task == "walk" else 0.
        desired_roll = -.06*math.sin(2*math.pi*reference.phase)*weight if self.task == "walk" else 0.
        for side in ("left", "right"):
            pose[JOINT_NAMES.index(f"{side}_ankle_pitch")] += 2*(pitch-desired_pitch)+.3*self.data.qvel[3]
            pose[JOINT_NAMES.index(f"{side}_ankle_roll")] += 2*(roll-desired_roll)+.3*self.data.qvel[4]
        return pose

    def _observation(self, info):
        observation = super()._observation(info)
        # Explicit cycle phase replaces the episode-time feature for this task.
        observation[-1] = self.reference.sample(float(self.data.time)+self.phase_offset).phase
        return observation

    def step(self, action):
        previous_distance = self._goal_distance()
        observation, _, terminated, truncated, info = super().step(action)
        reference = self.reference.sample(float(self.data.time)+self.phase_offset)
        pose_error = float(np.mean((self.data.qpos[self._qpos_indices]-reference.joint_targets)**2))
        imitation = math.exp(-8*pose_error)
        height_error = info["root_height"]-reference.root_position[2]
        height = math.exp(-(height_error/.15)**2)
        upright = max(info["upright"], 0)
        if self.task == "walk":
            direction = self.goal[:2]-self.data.qpos[:2]
            distance = float(np.linalg.norm(direction))
            speed = min(.45, max(0, distance-.15))
            desired = direction/max(distance, 1e-8)*speed
            velocity = math.exp(-6*float(np.sum((self.data.qvel[:2]-desired)**2)))
            near = math.exp(-(distance/.35)**2)
            progress = float(np.clip((previous_distance-distance)/self.dt, -1, 1))
            gait_weight = float(np.clip((distance-.18)/.45, 0, 1))
            clearance_reward = 0.
            clearances = []
            for index, phase in enumerate((reference.phase, reference.phase+.5)):
                geom_id = self._foot_geom_ids[index]
                matrix = self.data.geom_xmat[geom_id].reshape(3, 3)
                sole = float(self.data.geom_xpos[geom_id, 2]-np.abs(matrix[2]).dot(self.model.geom_size[geom_id])-self.floor_height)
                desired_clearance = .055*max(0., math.sin(2*math.pi*phase))*gait_weight
                clearance_reward += math.exp(-((sole-desired_clearance)/.025)**2)
                clearances.append(sole)
            reward = .5*upright+.5*height+.4*imitation+4*velocity+3*progress+3*near+2*clearance_reward
            info.update(foot_clearances=clearances, gait_clearance_reward=clearance_reward)
        else:
            horizontal = float(np.linalg.norm(self.data.qpos[:2]-self.goal[:2]))
            seat_height = float(self.data.qpos[2]-self.goal[2])
            near = math.exp(-(horizontal/.2)**2-(seat_height/.15)**2)
            reward = upright+.8*imitation+2*near+float(info["pelvis_support_contact"])
        reward -= .03*float(np.mean(np.square(action)))
        if terminated:
            reward -= 30
        info.update(reference_pose_error=pose_error, reference_phase=reference.phase,
                    reference_start=self.reference_start, reference_source="cached T2M BVH")
        return observation, float(reward), terminated, truncated, info
