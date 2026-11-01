"""Episode-wise, torque-driven phase-two navigation with real contact auditing.

The policy gets its final goal in the original 59 inputs and 51 actual scene
sensor values. Furniture changes only at reset. The reward-only visibility
graph never supplies a waypoint or changes the root, controls or observation.
Every 2ms physics substep is observed for collisions before arrival can count.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .avatar_profile import avatar_profile_sha256, load_avatar_profile
from .generalization import _difficulty_range, _integer, validate_scene_seed
from .humanoid import CONTROL_SUBSTEPS
from .navigation_scenes import ACTOR_RADIUS, NavigationPotential, generate_navigation_scene, point_box_distance
from .sensors import SceneSensorWrapper, measure_scene

NAVIGATION_OBSERVATION_CONTRACT = "base59_local_final_goal_xyz_task_bits_phase_then_actual_scene_rays51_v1"
OBSERVATION_COUNT = 110
SUCCESS_HOLD_SECONDS = .5
MINIMUM_FLOOR_SUPPORT_NEWTONS = 20.
OBSTACLE_REWARD_FLOOR = -8.
OBSTACLE_FAILURE_TERMINAL_COST = 1500.
NAVIGATION_SUCCESS_BONUS = 1500.
_FLOOR_SEMANTICS = {"floor"}


class EpisodeNavigationEnv(gym.Wrapper):
    """New physical room each episode; exact avatar contract stays unchanged.

    ``turn`` teaches modest heading changes on an empty finite floor.
    ``obstacle`` has one furniture box intersecting every direct goal route.
    No navigation skill is claimed by merely constructing these environments.
    """

    def __init__(self, *, mode="obstacle", split="train", base_seed=0,
                 difficulty_range=(0., .15), randomize=True, max_episode_seconds=30.,
                 avatar_profile=None, render_mode=None, terminate_on_success=True,
                 env_factory=None):
        if mode not in ("turn", "obstacle"):
            raise ValueError("mode must be turn or obstacle")
        validate_scene_seed(0 if split == "train" else 1_000_000, split)
        if not math.isfinite(max_episode_seconds) or max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive and finite")
        self.mode, self._split = mode, split
        self.base_seed = _integer(base_seed, "base_seed")
        self.difficulty_range = _difficulty_range(difficulty_range)
        self.randomize, self.max_episode_seconds = bool(randomize), float(max_episode_seconds)
        self._avatar_profile = load_avatar_profile(avatar_profile)
        self._body_model_sha256 = None
        self._render_mode = render_mode
        self.terminate_on_success = bool(terminate_on_success)
        self._env_factory = env_factory
        self._scene_rng = np.random.default_rng(self.base_seed)
        self._last_scene_seed = None
        self._episode_index = -1
        self._physics_seed = self.base_seed
        scene_seed = 0 if split == "train" else 1_000_000
        self.current_scene = generate_navigation_scene(scene_seed, mode=mode,
                                                       difficulty=self.difficulty_range[0])
        self._scene_sha256 = self._hash_scene(self.current_scene)
        self._potential = NavigationPotential(self.current_scene)
        self._clear_episode_audit()
        super().__init__(self._build_env(self.current_scene))
        self._install_contact_observer()
        self._last_position = self.unwrapped.data.qpos[:2].copy()
        self._previous_potential = self._potential.distance(self._last_position)
        self._previous_direct_distance = self.unwrapped._goal_distance()

    @property
    def split(self):
        return self._split

    @property
    def scene(self):
        return self.current_scene

    @property
    def avatar_profile(self):
        return self._avatar_profile

    @property
    def avatar_profile_sha256(self):
        return avatar_profile_sha256(self._avatar_profile)

    @property
    def body_model_sha256(self):
        return self._body_model_sha256

    @property
    def physical_assumptions(self):
        return self.unwrapped.physical_assumptions

    @staticmethod
    def _hash_scene(scene):
        return hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    def _build_env(self, scene):
        factory = self._env_factory
        if factory is None:
            from .steering import SteeringHumanoidEnv
            factory = SteeringHumanoidEnv
        base = factory(task="walk", scene=scene, randomize=self.randomize,
                       max_episode_seconds=self.max_episode_seconds,
                       avatar_profile=self._avatar_profile, render_mode=self._render_mode,
                       strict_gait_reward=True, reference_start=False)
        if base.observation_space.shape != (59,) or base.action_space.shape != (20,):
            base.close()
            raise ValueError("steering prior must expose 59 inputs and 20 physical joint controls")
        if base.avatar_profile_sha256 != avatar_profile_sha256(self._avatar_profile):
            base.close()
            raise ValueError("navigation scene rebuild changed the avatar profile")
        if self._body_model_sha256 is not None and base.body_model_sha256 != self._body_model_sha256:
            base.close()
            raise ValueError("navigation scene rebuild changed the physical body contract")
        self._body_model_sha256 = base.body_model_sha256
        wrapped = SceneSensorWrapper(base)
        if not isinstance(wrapped.observation_space, spaces.Box) or wrapped.observation_space.shape != (110,):
            wrapped.close()
            raise ValueError("navigation requires the unchanged base59 plus real sensor51 contract")
        return wrapped

    def _install_contact_observer(self):
        base = self.unwrapped
        self._floor_geom_ids = {base.model.geom(g["id"]).id for g in self.current_scene["geoms"]
                                if g.get("semantic") in _FLOOR_SEMANTICS}
        self._static_geom_names = {base.model.geom(g["id"]).id: g["id"] for g in self.current_scene["geoms"]}
        base._physics_contact_observer = self._observe_physics_contacts
        self._contact_force = np.zeros(6, dtype=np.float64)

    def _clear_episode_audit(self):
        self._qualified_success_seconds = 0.
        self._qualified_success_awarded = False
        self._collision_occurred = False
        self._collision_geom_ids = set()
        self._collision_steps = 0
        self._collision_substeps = 0
        self._collision_pair_samples = 0
        self._collision_seconds = 0.
        self._collision_force_peak = 0.
        self._collision_first_time = None
        self._collision_history = []
        self._previous_substep_collision = False
        self._path_length = 0.
        self._min_ray_clearance = math.inf
        self._episode_navigation_done = False
        self._last_terminal_reason = None
        self._last_reward_info = None
        self._clear_step_audit()

    def _clear_step_audit(self):
        self._step_collision_ids = set()
        self._step_collision_force_peak = 0.
        self._step_collision_substeps = 0
        self._step_observer_calls = 0

    def _contacts(self, model, data):
        obstacle_ids, force_peak, pair_count = set(), 0., 0
        floor_forces = [0., 0.]
        for index in range(data.ncon):
            contact = data.contact[index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(model.geom_bodyid[g]) for g in geom_ids)
            if not (0 in bodies and any(body > 0 for body in bodies)):
                continue
            mujoco.mj_contactForce(model, data, index, self._contact_force)
            normal_force = max(0., float(self._contact_force[0]))
            if normal_force <= 1.:
                continue
            static_id = geom_ids[bodies.index(0)]
            if static_id in self._floor_geom_ids:
                vertical_normal = abs(float(contact.frame[2]))
                if contact.dist <= .002 and vertical_normal > .7:
                    for side, foot_id in enumerate(self.unwrapped._foot_geom_ids):
                        if foot_id in geom_ids:
                            floor_forces[side] += normal_force * vertical_normal
            else:
                obstacle_ids.add(self._static_geom_names.get(static_id, f"geom-{static_id}"))
                force_peak = max(force_peak, normal_force)
                pair_count += 1
        return obstacle_ids, force_peak, pair_count, floor_forces

    def _record_contacts(self, model, data, *, physics_substep):
        ids, peak, count, _ = self._contacts(model, data)
        if ids:
            self._collision_occurred = True
            self._collision_geom_ids.update(ids)
            self._step_collision_ids.update(ids)
            self._collision_force_peak = max(self._collision_force_peak, peak)
            self._step_collision_force_peak = max(self._step_collision_force_peak, peak)
            if self._collision_first_time is None:
                self._collision_first_time = float(data.time)
            if physics_substep:
                self._collision_substeps += 1
                self._step_collision_substeps += 1
                self._collision_pair_samples += count
                self._collision_seconds += float(model.opt.timestep)
            if not self._previous_substep_collision and len(self._collision_history) < 128:
                self._collision_history.append({"time_seconds": float(data.time), "geom_ids": sorted(ids),
                                                "peak_normal_force_newtons": peak})
            elif self._collision_history:
                self._collision_history[-1]["peak_normal_force_newtons"] = max(
                    self._collision_history[-1]["peak_normal_force_newtons"], peak)
        if physics_substep:
            self._previous_substep_collision = bool(ids)

    def _observe_physics_contacts(self, model, data):
        """Read-only callback run after each physical 2ms MuJoCo substep."""
        self._step_observer_calls += 1
        self._record_contacts(model, data, physics_substep=True)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options) - {"scene_seed", "physics_seed"}:
            raise ValueError("navigation reset options may only select scene_seed or physics_seed")
        explicit_scene = (validate_scene_seed(options["scene_seed"], self.split)
                          if "scene_seed" in options else None)
        explicit_physics = (_integer(options["physics_seed"], "physics_seed")
                            if "physics_seed" in options else None)
        if seed is not None:
            self._scene_rng = np.random.default_rng(_integer(seed, "seed"))
            self._last_scene_seed = None
        domain_start = 0 if self.split == "train" else 1_000_000
        scene_seed = (explicit_scene if explicit_scene is not None else
                      int(self._scene_rng.integers(domain_start, domain_start + 1_000_000)))
        if explicit_scene is None:
            while scene_seed == self._last_scene_seed:
                scene_seed = int(self._scene_rng.integers(domain_start, domain_start + 1_000_000))
        difficulty = float(self._scene_rng.uniform(*self.difficulty_range))
        physics_seed = (explicit_physics if explicit_physics is not None else
                        int(self._scene_rng.integers(0, 2**31)))
        scene = generate_navigation_scene(scene_seed, mode=self.mode, difficulty=difficulty)
        scene["metadata"]["split"] = self.split
        scene["metadata"]["directRouteBlocked"] = scene["metadata"]["obstacleBlockedDirectRoute"]
        candidate = self._build_env(scene)
        try:
            observation, info = candidate.reset(seed=physics_seed)
        except Exception:
            candidate.close()
            raise
        previous = self.env
        self.env, self.current_scene = candidate, scene
        previous.close()
        self._physics_seed = physics_seed
        self._scene_sha256 = self._hash_scene(scene)
        self._potential = NavigationPotential(scene)
        self._last_scene_seed = scene_seed
        self._episode_index += 1
        self._clear_episode_audit()
        self._install_contact_observer()
        self._last_position = self.unwrapped.data.qpos[:2].copy()
        self._previous_potential = self._potential.distance(self._last_position)
        self._previous_direct_distance = self.unwrapped._goal_distance()
        return observation, self._add_info(info, advance=False)

    def _metrics(self, info):
        base = self.unwrapped
        _, current_peak, _, floor_forces = self._contacts(base.model, base.data)
        floor_support = [force >= MINIMUM_FLOOR_SUPPORT_NEWTONS for force in floor_forces]
        goal_error = float(np.linalg.norm(base.data.qpos[:2] - base.goal[:2]))
        root_speed = float(np.linalg.norm(base.data.qvel[:3]))
        root_height_error = abs(float(base.data.qpos[2]) - (base.goal[2] + base.standing_root_height))
        sensor = info["scene_sensors"]
        ray_clearance = float(min(value for layer in sensor["horizontal_ranges_m"] for value in layer))
        centre_floor = sensor["floor_heights_m"][4]
        actual_floor = (centre_floor is not None and abs(float(centre_floor) - float(base.goal[2])) < .04)
        instant_success = bool(not info["fall"] and not self._collision_occurred
                               and goal_error < .25 and root_speed < .25
                               and info["upright"] > .8 and root_height_error < .15
                               and all(floor_support) and actual_floor)
        return {"goal_distance_m": goal_error, "root_speed_m_s": root_speed,
                "root_height_error_m": root_height_error,
                "floor_support": floor_support, "floor_support_forces_newtons": floor_forces,
                "measured_centre_surface_height_m": centre_floor, "goal_floor_matches": bool(actual_floor),
                "instant_success": instant_success, "base_success": bool(info.get("success", False)),
                "collision_occurred": self._collision_occurred,
                "current_collision_geom_ids": sorted(self._step_collision_ids),
                "obstacle_contact_geom_ids": sorted(self._collision_geom_ids),
                "obstacle_contact_steps": self._collision_steps,
                "obstacle_contact_substeps": self._collision_substeps,
                "obstacle_contact_pair_samples": self._collision_pair_samples,
                "obstacle_contact_seconds": self._collision_seconds,
                "collision_force_newtons": max(current_peak, self._step_collision_force_peak),
                "peak_collision_force_newtons": self._collision_force_peak,
                "first_collision_time_seconds": self._collision_first_time,
                "collision_history": copy.deepcopy(self._collision_history),
                "current_min_ray_clearance_m": ray_clearance,
                "minimum_ray_clearance_m": min(self._min_ray_clearance, ray_clearance),
                "path_length_m": self._path_length,
                "reward_navigation_potential_distance_m": self._potential.distance(base.data.qpos[:2])}

    def _add_info(self, info, *, advance):
        result = dict(info)
        metrics = self._metrics(info)
        if advance:
            self._qualified_success_seconds = (self._qualified_success_seconds + self.unwrapped.dt
                                               if metrics["instant_success"] else 0.)
            self._min_ray_clearance = min(self._min_ray_clearance, metrics["current_min_ray_clearance_m"])
        qualified = self._qualified_success_seconds + 1e-12 >= SUCCESS_HOLD_SECONDS
        result["navigation"] = {
            "mode": self.mode, "split": self.split,
            "sceneSeed": self.current_scene["seed"], "scene_seed": self.current_scene["seed"],
            "physics_seed": self._physics_seed, "episode_index": self._episode_index,
            "scene_sha256": self._scene_sha256, "difficulty": self.current_scene["metadata"]["difficulty"],
            "goal": copy.deepcopy(self.current_scene["goals"][0]),
            "direct_route_blocked": self.current_scene["metadata"]["obstacleBlockedDirectRoute"],
            "collision_occurred": self._collision_occurred,
            "qualified_success": bool(qualified), "qualified_success_seconds": self._qualified_success_seconds,
            "observation_count": OBSERVATION_COUNT, "base_observation_count": 59,
            "sensor_feature_count": 51, "policy_input_contract": NAVIGATION_OBSERVATION_CONTRACT,
            "avatar_profile_sha256": self.avatar_profile_sha256,
            "body_model_sha256": self.body_model_sha256,
            "root_step_control": "MuJoCo dynamics only", "scene_rebuilt_at_reset": True,
            "waypoint_controller": False, "reward_only_visibility_graph": True,
            "collision_audit": "every physics substep plus final control-frame contacts",
            "collision_audit_timestep_seconds": float(self.unwrapped.model.opt.timestep),
            "success_hold_seconds_required": SUCCESS_HOLD_SECONDS,
            "minimum_floor_support_newtons_per_foot": MINIMUM_FLOOR_SUPPORT_NEWTONS,
            "proficiencyPassed": False, "metrics": metrics}
        result["base_success"] = bool(info.get("success", False))
        result["success"] = bool(qualified)
        result["success_seconds"] = self._qualified_success_seconds
        result["navigation_floor_support"] = metrics["floor_support"]
        result["navigation_success_seconds"] = self._qualified_success_seconds
        result["collision_occurred"] = self._collision_occurred
        result["collision_force_newtons"] = metrics["collision_force_newtons"]
        result["navigation"]["terminal_reason"] = self._last_terminal_reason
        if self._last_reward_info is not None:
            result["navigation"]["reward"] = copy.deepcopy(self._last_reward_info)
        return result

    def step(self, action):
        if self._episode_navigation_done:
            raise RuntimeError("reset() is required after a completed navigation episode")
        self._clear_step_audit()
        collision_before = self._collision_occurred
        observation, reward, terminated, truncated, info = self.env.step(action)
        base = self.unwrapped
        if self._step_observer_calls != CONTROL_SUBSTEPS:
            raise RuntimeError("navigation requires a collision observer at every physical substep")
        inherited_reward = float(reward)
        self._record_contacts(base.model, base.data, physics_substep=False)
        if self._step_collision_ids:
            self._collision_steps += 1
        position = base.data.qpos[:2]
        self._path_length += float(np.linalg.norm(position - self._last_position))
        self._last_position = position.copy()
        distance = base._goal_distance()
        direct_progress = float(np.clip((self._previous_direct_distance - distance) / base.dt, -1, 1))
        direction = base.goal[:2] - position
        desired = direction / max(distance, 1e-8) * min(.45, max(0., distance - .15))
        direct_velocity_reward = math.exp(-6 * float(np.sum((base.data.qvel[:2] - desired) ** 2)))
        direct_near_reward = math.exp(-(distance / .35) ** 2)
        potential = self._potential.distance(position)
        progress = float(np.clip((self._previous_potential - potential) / base.dt, -1, 1))
        self._previous_potential, self._previous_direct_distance = potential, distance
        ray_clearance = min(value for layer in info["scene_sensors"]["horizontal_ranges_m"] for value in layer)
        furniture = self._potential.obstacle
        geometric_clearance = point_box_distance(position, furniture) if furniture is not None else math.inf
        clearance = min(ray_clearance, geometric_clearance)
        safety_gate = float(np.clip((clearance - ACTOR_RADIUS) / .16, 0, 1))
        safe_progress = min(progress, 0.) + max(progress, 0.) * safety_gate
        risk = float(np.clip((ACTOR_RADIUS + .18 - clearance) / .30, 0, 1))
        removed_progress = removed_velocity = removed_near = 0.
        progress_reward = risk_penalty = collision_penalty = time_penalty = 0.
        if self.mode == "obstacle":
            # Furniture tasks need a collision-free reward potential instead
            # of direct-line incentives. On the empty turning floor preserve
            # the full Footstep reward, including its bounded speed tracking.
            removed_progress, removed_velocity, removed_near = (
                3 * direct_progress, 4 * direct_velocity_reward, 3 * direct_near_reward)
            reward -= removed_progress + removed_velocity + removed_near
            progress_reward, risk_penalty, time_penalty = 20 * safe_progress, 5 * risk * risk, 6.
            collision_penalty = (100. if self._collision_occurred and not collision_before else 0.)
            if self._step_collision_ids:
                collision_penalty += 20. + min(self._step_collision_force_peak / 200., 10.)
            reward += progress_reward - risk_penalty - collision_penalty - time_penalty
        info = self._add_info(info, advance=True)
        success_bonus = 0.
        if info["navigation"]["qualified_success"] and not self._qualified_success_awarded:
            success_bonus = NAVIGATION_SUCCESS_BONUS
            reward += success_bonus
            self._qualified_success_awarded = True
        reward_info = {
            "inherited_footstep_reward": inherited_reward,
            "footstep_velocity_tracking_preserved": self.mode == "turn",
            "retained_goal_speed_tracking_reward": 4 * direct_velocity_reward if self.mode == "turn" else 0.,
            "goal_speed_tracking_target_m_s": min(.45, max(0., distance - .15)),
            "removed_straight_goal_progress": removed_progress,
            "removed_straight_goal_velocity": removed_velocity,
            "removed_straight_goal_near": removed_near,
            "geometry_potential_progress": progress_reward, "proximity_penalty": risk_penalty,
            "collision_penalty": collision_penalty, "time_penalty": time_penalty, "success_bonus": success_bonus,
            "safety_gate": safety_gate,
            "scope": ("inherited Footstep posture/gait/goal-speed reward on empty turning floor"
                      if self.mode == "turn" else
                      "geometry-only potential shapes reward; final goal and policy controls are unchanged")}
        if self.terminate_on_success and info["navigation"]["qualified_success"]:
            terminated = True
        # Negative waiting returns otherwise make a cheap early fall more
        # profitable than trying to recover. At the trained gamma=.99, this
        # task-reward floor bounds negative continuing value at -800. The
        # 1500 failure cost lies outside this floor. An irreversible collision
        # or real fall ends the task; physical integration remains untouched.
        unclamped_reward = float(reward)
        clamp_adjustment = float(max(OBSTACLE_REWARD_FLOOR - reward, 0.))
        reward += clamp_adjustment
        failure_cost = 0.
        if self._collision_occurred or info["fall"]:
            terminal_reason = ("collision_and_fall" if self._collision_occurred and info["fall"]
                               else ("collision" if self._collision_occurred else "fall"))
            failure_cost = OBSTACLE_FAILURE_TERMINAL_COST
            reward -= failure_cost
            terminated = True
        elif terminated and info["navigation"]["qualified_success"]:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        else:
            terminal_reason = None
        reward_info.update(terminal_reason=terminal_reason, terminal_cost=failure_cost,
                           unclamped_reward=unclamped_reward,
                           task_reward_lower_bound=OBSTACLE_REWARD_FLOOR,
                           reward_clamp_adjustment=clamp_adjustment,
                           failure_bound_discount_factor=.99)
        self._last_terminal_reason = terminal_reason
        self._last_reward_info = copy.deepcopy(reward_info)
        info["navigation"]["terminal_reason"] = terminal_reason
        info["navigation"]["reward"] = reward_info
        self._episode_navigation_done = bool(terminated or truncated)
        return observation, float(reward), bool(terminated), bool(truncated), info

    def snapshot(self):
        snapshot = self.unwrapped.snapshot()
        info = dict(snapshot["stats"])
        info["scene_sensors"] = measure_scene(self.unwrapped)
        snapshot["stats"] = self._add_info(info, advance=False)
        snapshot["scene"] = copy.deepcopy(self.current_scene)
        return snapshot
