"""Gymnasium interface to a floating, torque-controlled MuJoCo humanoid.

Standing is the first validated task. Walk/sit supply experimental objectives
for later learning; specifying a task is not a guarantee of a learned skill.
There is no root teleport, foot IK, mocap body, or pose assignment in step().
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .avatar_profile import avatar_profile_sha256, load_avatar_profile, physical_assumptions

from .humanoid import (
    CONTROL_SUBSTEPS, JOINT_LIMITS, JOINT_NAMES, PD_KD, PD_KP,
    PHYSICS_TIMESTEP, RESIDUAL_LIMITS, STANDING_ROOT_HEIGHT, TORQUE_LIMITS,
    build_humanoid_xml, standing_joint_angles,
)


class HumanoidRoomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, task="stand", scene=None, randomize=False,
                 max_episode_seconds=6, render_mode=None, avatar_profile=None):
        super().__init__()
        if task not in ("stand", "walk", "sit"):
            raise ValueError("task must be stand, walk, or sit")
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("Unsupported render_mode")
        if not math.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive and finite")
        self.task = task
        self.scene = copy.deepcopy(scene)
        self.avatar_profile = load_avatar_profile(avatar_profile)
        self.body_profile = self.avatar_profile
        self.avatar_profile_sha256 = avatar_profile_sha256(self.avatar_profile)
        self.body_model_sha256 = hashlib.sha256(
            build_humanoid_xml(avatar_profile=self.avatar_profile).encode()).hexdigest()
        self.physical_assumptions = physical_assumptions(self.avatar_profile)
        self.randomize = bool(randomize)
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(build_humanoid_xml(scene, self.avatar_profile))
        self.data = mujoco.MjData(self.model)
        self.joint_names = JOINT_NAMES
        self.dt = PHYSICS_TIMESTEP * CONTROL_SUBSTEPS
        self.max_steps = int(math.ceil(max_episode_seconds / self.dt))
        self.max_episode_seconds = float(max_episode_seconds)
        self.pelvis_body_id = self.model.body("pelvis").id
        self.root_body_id = self.pelvis_body_id
        self._qpos_indices = np.asarray([self.model.jnt_qposadr[self.model.joint(name).id]
                                         for name in JOINT_NAMES])
        self._qvel_indices = np.asarray([self.model.jnt_dofadr[self.model.joint(name).id]
                                         for name in JOINT_NAMES])
        self._foot_geom_ids = tuple(self.model.geom(f"humanoid_{side}_foot").id
                                    for side in ("left", "right"))
        self._seat_geom_ids = set()
        for geom in (scene or {}).get("geoms", []):
            if geom.get("semantic") in ("seat", "chair_seat", "chair-seat"):
                self._seat_geom_ids.add(self.model.geom(geom["id"]).id)
        self._base_mass = self.model.body_mass.copy()
        self._base_inertia = self.model.body_inertia.copy()
        self._base_friction = self.model.geom_friction.copy()
        self.standing_pose = standing_joint_angles()
        self.standing_root_height = STANDING_ROOT_HEIGHT
        self._body_scale = 1.0
        self.seat_pelvis_offset = .09
        rest_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, rest_data)
        self.physics_rest_bodies = self._body_snapshot(rest_data)
        if self.avatar_profile is not None:
            self._body_scale = float(self.avatar_profile["physical"]["rootHeight"]) / STANDING_ROOT_HEIGHT
            pelvis_geom = self.model.geom("humanoid_pelvis").id
            # Measure the compiled capsule's lowest point relative to its body
            # origin, including the avatar's seated skin-envelope offset. Its
            # axis is horizontal, rather than an assumed root-centred sphere.
            self.seat_pelvis_offset = float(rest_data.xpos[self.pelvis_body_id, 2]
                - rest_data.geom_xpos[pelvis_geom, 2] + self.model.geom_size[pelvis_geom, 0]
                + self.model.geom_size[pelvis_geom, 1]*abs(rest_data.geom_xmat[pelvis_geom].reshape(3, 3)[2, 2]))
            # Calibrate contact once in independent reset scratch data. The
            # rollout root remains exclusively integrated by MuJoCo in step.
            rest_data.qpos[self._qpos_indices] = self.standing_pose
            mujoco.mj_forward(self.model, rest_data)
            bottoms = [float(rest_data.geom_xpos[foot, 2] - np.abs(
                rest_data.geom_xmat[foot].reshape(3, 3)[2]).dot(self.model.geom_size[foot]))
                for foot in self._foot_geom_ids]
            self.standing_root_height = float(rest_data.qpos[2]) - min(bottoms) + .0005
        self.action_space = spaces.Box(-1.0, 1.0, (len(JOINT_NAMES),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (59,), dtype=np.float32)
        self.spawn = np.asarray((scene or {}).get("spawn", [0, 0, 0]), dtype=np.float64)
        if self.spawn.shape != (3,) or not np.isfinite(self.spawn).all():
            raise ValueError("scene.spawn must be a finite Z-up position")
        self.floor_height = float(self.spawn[2])
        self.spawn_heading = float((scene or {}).get("spawnHeading", 0))
        if not math.isfinite(self.spawn_heading):
            raise ValueError("scene.spawnHeading must be a finite angle in radians")
        self.goal = self.spawn + np.asarray([0., -2.0 if task == "walk" else 0.,
                                             0.57 if task == "sit" else self.standing_root_height])
        goals = (scene or {}).get("goals", [])
        wanted = "seat" if task == "sit" else task
        for goal in goals:
            if (goal.get("kind") == wanted or (task == "sit" and goal.get("kind") == "sit")
                    or (task == "walk" and goal.get("kind") in ("step", "stairs"))):
                value = np.asarray(goal.get("position"), dtype=np.float64)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError("scene goal position must be finite")
                self.goal = value.copy()
                if task == "sit":
                    self.goal[2] += self.seat_pelvis_offset
                break
        self._renderer = None
        self._viewer = None
        self._step_count = 0
        self._last_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._last_torque = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._previous_goal_distance = 0.0
        self._success_seconds = 0.0
        self._episode_done = False
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.model.body_mass[:] = self._base_mass
        self.model.body_inertia[:] = self._base_inertia
        self.model.geom_friction[:] = self._base_friction
        if self.randomize:
            mass_scale = self.np_random.uniform(0.95, 1.05, self.model.nbody)
            self.model.body_mass[:] *= mass_scale
            self.model.body_inertia[:] *= mass_scale[:, None]
            self.model.geom_friction[:, 0] *= self.np_random.uniform(0.85, 1.15, self.model.ngeom)
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.spawn + [0, 0, self.standing_root_height]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self._qpos_indices] = self.standing_pose
        if self.randomize:
            tilt = self.np_random.uniform(-0.025, 0.025, 2)
            quaternion = np.empty(4)
            mujoco.mju_axisAngle2Quat(quaternion, np.asarray([1., 0., 0.]), tilt[0])
            second = np.empty(4)
            mujoco.mju_axisAngle2Quat(second, np.asarray([0., 1., 0.]), tilt[1])
            mujoco.mju_mulQuat(self.data.qpos[3:7], quaternion, second)
            self.data.qvel[:2] = self.np_random.uniform(-0.04, 0.04, 2)
            self.data.qpos[self._qpos_indices] += self.np_random.uniform(-0.012, 0.012, len(JOINT_NAMES))
        self.data.qpos[self._qpos_indices] = np.clip(self.data.qpos[self._qpos_indices],
                                                    JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        heading = np.asarray([math.cos(self.spawn_heading/2), 0., 0., math.sin(self.spawn_heading/2)])
        original_orientation = self.data.qpos[3:7].copy()
        mujoco.mju_mulQuat(self.data.qpos[3:7], heading, original_orientation)
        c, s = math.cos(self.spawn_heading), math.sin(self.spawn_heading)
        self.data.qvel[:2] = np.asarray([[c, -s], [s, c]]) @ self.data.qvel[:2]
        self._step_count = 0
        self._last_action.fill(0)
        self._last_torque.fill(0)
        self._success_seconds = 0.0
        self._episode_done = False
        mujoco.mj_forward(self.model, self.data)
        self._previous_goal_distance = self._goal_distance()
        info = self._info()
        return self._observation(info), info

    def _reference_pose(self):
        pose = self.standing_pose.copy()
        if self.task == "sit":
            # An untrained crouching reference; only physics/policy can find a
            # chair and establish support. The pelvis is never placed on it.
            phase = min(float(self.data.time) / 1.5, 1.0)
            blend = phase * phase * (3 - 2 * phase)
            for side in ("left", "right"):
                for joint, target in (("hip_pitch", -1.0), ("knee", 1.65),
                                      ("ankle_pitch", -0.65)):
                    index = JOINT_NAMES.index(f"{side}_{joint}")
                    pose[index] += blend * (target - pose[index])
            pose[0] = 0.12 * blend
        # Joint-only feedback: ankles react to measured floating-root tilt and
        # angular velocity. This is a disclosed PD balance baseline, not RL and
        # not an assignment of the root pose. The residual policy can amend it.
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        local_up = rotation.T @ [0., 0., 1.]
        pitch = math.atan2(local_up[1], local_up[2])
        roll = math.atan2(-local_up[0], local_up[2])
        for side in ("left", "right"):
            pose[JOINT_NAMES.index(f"{side}_ankle_pitch")] += 2.0 * pitch + .3 * self.data.qvel[3]
            pose[JOINT_NAMES.index(f"{side}_ankle_roll")] += 2.0 * roll + .3 * self.data.qvel[4]
        return pose

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
        self._step_count += 1
        # mj_step integrates state after its position stage; recompute derived
        # geometry/contact data for an accurate current-state snapshot.
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

    def apply_push(self, force):
        """Apply a persistent world-space pelvis force; call with zeros to stop."""
        force = np.asarray(force, dtype=np.float64)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force must contain three finite Newton values")
        self.data.xfrc_applied[self.pelvis_body_id, :3] = force

    def _goal_distance(self):
        return float(np.linalg.norm(self.data.qpos[:2] - self.goal[:2]))

    def _info(self):
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        upright = float(rotation[2, 2])
        root_height = float(self.data.qpos[2] - self.floor_height)
        feet = [False, False]
        forces = [0., 0.]
        pelvis_support = False
        pelvis_support_force = 0.
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(self.model.geom_bodyid[geom]) for geom in geom_ids)
            if 0 not in bodies:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            vertical_normal = abs(float(contact.frame[2]))
            for foot, geom_id in enumerate(self._foot_geom_ids):
                if geom_id in geom_ids and contact.dist <= .002 and vertical_normal > .7 and force[0] > 1:
                    feet[foot] = True
                    forces[foot] += max(float(force[0]), 0) * vertical_normal
            if (self.pelvis_body_id in bodies and force[0] > 1
                    and vertical_normal > .7 and any(geom in self._seat_geom_ids for geom in geom_ids)):
                pelvis_support = True
                pelvis_support_force += float(force[0]) * vertical_normal
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        fall_height = (.34 if self.task == "sit" else .50)*self._body_scale
        return {
            "upright": upright, "root_height": root_height,
            "root_position": self.data.qpos[:3].tolist(), "root_velocity": self.data.qvel[:3].tolist(),
            "contact_count": int(self.data.ncon), "left_foot_contact": feet[0], "right_foot_contact": feet[1],
            "left_foot_force": forces[0], "right_foot_force": forces[1],
            "pelvis_support_contact": pelvis_support,
            "pelvis_support_force": pelvis_support_force,
            "fall": not finite or upright < .45 or root_height < fall_height,
            "success": self._success_seconds >= .5, "success_seconds": self._success_seconds,
            "goal_distance": self._goal_distance(), "time": float(self.data.time),
            "torque_rms": float(np.sqrt(np.mean(np.square(self._last_torque)))),
            "max_abs_torque": float(np.max(np.abs(self._last_torque))),
            "task": self.task,
        }

    def _observation(self, info):
        rotation = self.data.xmat[self.pelvis_body_id].reshape(3, 3)
        task = [float(self.task == name) for name in ("stand", "walk", "sit")]
        observation = np.concatenate((
            self.data.qpos[self._qpos_indices], self.data.qvel[self._qvel_indices],
            [info["root_height"]], rotation.T @ [0., 0., 1.],
            rotation.T @ self.data.qvel[:3], self.data.qvel[3:6],
            rotation.T @ (self.goal - self.data.qpos[:3]), task,
            [float(info["left_foot_contact"]), float(info["right_foot_contact"])],
            [self._step_count / self.max_steps],
        )).astype(np.float32)
        return observation

    def snapshot(self):
        """JSON-ready actual simulated geoms, native MuJoCo size conventions."""
        geoms = []
        kinds = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                 int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                 int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}
        for index in range(self.model.ngeom):
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, self.data.geom_xmat[index])
            geom = {"id": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index),
                    "type": kinds.get(int(self.model.geom_type[index]), "unknown"),
                    "position": self.data.geom_xpos[index].tolist(), "quaternion": quaternion.tolist(),
                    "size": self.model.geom_size[index].tolist(), "rgba": self.model.geom_rgba[index].tolist(),
                    "friction": self.model.geom_friction[index].tolist(),
                    "body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[index]))}
            if geom["type"] == "mesh":
                mesh_id = int(self.model.geom_dataid[index])
                vertex_start, vertex_count = int(self.model.mesh_vertadr[mesh_id]), int(self.model.mesh_vertnum[mesh_id])
                face_start, face_count = int(self.model.mesh_faceadr[mesh_id]), int(self.model.mesh_facenum[mesh_id])
                geom["mesh"] = {"vertices": self.model.mesh_vert[vertex_start:vertex_start + vertex_count].tolist(),
                                "faces": self.model.mesh_face[face_start:face_start + face_count].tolist()}
            geoms.append(geom)
        return {"time": float(self.data.time), "task": self.task, "geoms": geoms,
                "bodies": self._body_snapshot(self.data), "stats": self._info(),
                "bodyModelSha256": self.body_model_sha256,
                "avatarProfileSha256": self.avatar_profile_sha256}

    def _body_snapshot(self, data):
        return [{"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index),
                 "position": data.xpos[index].tolist(), "quaternion": data.xquat[index].tolist()}
                for index in range(1, self.model.nbody)]

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
