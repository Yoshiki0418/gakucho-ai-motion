"""Episode-wise procedural worlds for future 110-input physics policies.

The free body is simulated by the existing MuJoCo environment. Only reset may
initialize its state; this wrapper never writes poses, velocities or forces in
step(). Geometry, sensors and contact all use the same rebuilt physical model.
These are training/evaluation foundations, not evidence of a learned room,
multi-room or stair skill. No 59-input checkpoint is silently adapted to 110.
An optional measured avatar profile is owned by the wrapper and retained when
the scene is rebuilt. A policy must match both that profile and its body model;
an appearance match does not make an old checkpoint compatible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .environment import HumanoidRoomEnv
from .avatar_profile import avatar_profile_sha256, load_avatar_profile
from .body_contract import assert_policy_body
from .scenes import HELD_OUT_SEED_BASE, generate_scene
from .sensors import SceneSensorWrapper

OBSERVATION_COUNT = 110
BASE_OBSERVATION_COUNT = 59
HELD_OUT_SEED_LIMIT = 2 * HELD_OUT_SEED_BASE
_TASKS = ("stand", "walk", "sit")
_WALK_KINDS = ("walk", "step", "stairs")
_SEAT_KINDS = ("seat", "sit")
_TERRAIN_SEMANTICS = {"floor", "step", "stair", "landing", "upper-floor"}
_SUCCESS_HOLD_SECONDS = .5


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def validate_scene_seed(seed: int, split: str) -> int:
    """Reject scene-seed leakage, including explicit reset/evaluation overrides."""
    seed = _integer(seed, "scene_seed")
    if split == "train":
        valid = seed < HELD_OUT_SEED_BASE
    elif split == "heldout":
        valid = HELD_OUT_SEED_BASE <= seed < HELD_OUT_SEED_LIMIT
    else:
        raise ValueError("split must be train or heldout")
    if not valid:
        raise ValueError(f"scene_seed {seed} does not belong to the {split} split")
    return seed


def _difficulty_range(value) -> tuple[float, float]:
    bounds = value if isinstance(value, (list, tuple)) else (value, value)
    if len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in bounds):
        raise ValueError("difficulty must be finite or a finite (low, high) range")
    low, high = map(float, bounds)
    if not 0 <= low <= high <= 1:
        raise ValueError("difficulty must lie between 0 and 1")
    return low, high


def _goal_task(goal: dict) -> str | None:
    kind = goal.get("kind")
    return "walk" if kind in _WALK_KINDS else ("sit" if kind in _SEAT_KINDS else
                                               ("stand" if kind == "stand" else None))


def _select_goal(scene: dict, requested_task: str, rng: np.random.Generator,
                 goal_id: str | None = None) -> tuple[str, dict]:
    goals = [copy.deepcopy(goal) for goal in scene["goals"]]
    goals.append({"id": "spawn-stability", "kind": "stand",
                  "position": list(scene["spawn"]),
                  "surfaceHeight": float(scene["spawn"][2])})
    candidates = [goal for goal in goals if requested_task == "mixed" or _goal_task(goal) == requested_task]
    # A target at the spawn does not demonstrate walking or room navigation.
    candidates = [goal for goal in candidates if _goal_task(goal) != "walk"
                  or np.linalg.norm(np.asarray(goal["position"])[:2] - np.asarray(scene["spawn"])[:2]) >= .4]
    if goal_id is not None:
        if not isinstance(goal_id, str):
            raise ValueError("goal_id must be a string")
        candidates = [goal for goal in candidates if goal["id"] == goal_id]
    if not candidates:
        raise ValueError(f"scene has no usable goal for task={requested_task}, goal_id={goal_id!r}")
    goal = candidates[int(rng.integers(len(candidates)))]
    return _goal_task(goal), goal


def _prepare_scene(stage: int, seed: int, difficulty: float, task: str, split: str,
                   rng: np.random.Generator, goal_id: str | None, heading_mode: str) -> tuple[dict, str, dict]:
    scene = generate_scene(stage, seed=seed, difficulty=difficulty)
    selected_task, goal = _select_goal(scene, task, rng, goal_id)
    delta = np.asarray(goal["position"])[:2] - np.asarray(scene["spawn"])[:2]
    initial_goal_heading = (math.atan2(float(delta[0]), float(-delta[1]))
                            if np.linalg.norm(delta) > 1e-9 else float(scene.get("spawnHeading", 0.)))
    spawn_heading = initial_goal_heading if heading_mode == "face_goal" else float(scene.get("spawnHeading", 0.))
    goal_heading = goal.get("heading")
    if goal_heading is None and selected_task == "sit":
        seat = next((geom for geom in scene["geoms"] if geom.get("entityId") == goal.get("entityId")
                     and geom.get("semantic") in ("seat", "chair-seat", "chair_seat")), None)
        goal_heading = seat.get("yaw", 0.) if seat else None
    goal["heading"] = float(goal_heading if goal_heading is not None else initial_goal_heading)
    goal["positionReference"] = ("seat_surface" if selected_task == "sit" else
                                 ("standing_pelvis" if selected_task == "stand" else "supporting_surface"))
    scene["spawnHeading"] = spawn_heading
    # The core selects the first matching task goal. Reordering selects a real
    # goal without moving furniture, placing the body nearby, or changing it later.
    scene["goals"] = [goal] + [other for other in scene["goals"] if other["id"] != goal["id"]]
    scene["metadata"]["generalization"] = {
        "stage": stage, "split": split, "sceneSeed": seed, "selectedTask": selected_task,
        "selectedGoalId": goal["id"], "spawnHeading": spawn_heading,
        "initialGoalHeading": initial_goal_heading, "goalHeading": goal["heading"],
        "spawnPositionChanged": False, "proficiencyPassed": False,
        "learningStatus": "not_trained_or_validated", "geometryOnly": True,
    }
    return scene, selected_task, goal


def assert_policy_compatible(policy, env) -> None:
    """Check the input space before predicting, never pad an old 59-input policy."""
    policy_space = getattr(policy, "observation_space", None)
    if not isinstance(policy_space, spaces.Box) or policy_space.shape != (OBSERVATION_COUNT,):
        shape = getattr(policy_space, "shape", None)
        raise ValueError(f"generalization requires a 110-input policy; checkpoint has {shape}")
    if env.observation_space.shape != (OBSERVATION_COUNT,):
        raise ValueError("generalization environment must have 110 observations")
    policy_action = getattr(policy, "action_space", None)
    if policy_action is not None and policy_action.shape != env.action_space.shape:
        raise ValueError("policy action space does not match the 20 physical joint controls")
    if not callable(getattr(policy, "predict", None)):
        raise ValueError("policy must expose predict(observation, deterministic=True)")
    assert_policy_body(policy, env.unwrapped)


class GeneralizationWrapper(gym.Wrapper):
    """Rebuild a physical scene and its sensor wrapper for every reset.

    ``seed`` resets the deterministic episode-selection RNG. ``scene_seed`` in
    reset options selects a specific world within this immutable split. Physics
    randomization has a separate ``physics_seed`` so a held-out world cannot
    accidentally be used for training. Goal orientation is metadata; the 110
    inputs contain the existing local goal XYZ/task bits and actual ray sensors,
    without adding an unannounced heading feature to a trained input contract.
    """

    def __init__(self, stage=2, *, split="train", task="walk", base_seed=0, difficulty=.5,
                 randomize=True, max_episode_seconds=12., heading_mode="face_goal",
                 motion_prior="base", env_factory: Callable[..., gym.Env] | None = None,
                 env_kwargs: dict | None = None, render_mode=None, avatar_profile=None):
        stage = _integer(stage, "stage")
        if stage not in (2, 3, 4):
            raise ValueError("generalization stage must be 2, 3 or 4")
        if task not in (*_TASKS, "mixed"):
            raise ValueError("task must be stand, walk, sit or mixed")
        if heading_mode not in ("face_goal", "scene"):
            raise ValueError("heading_mode must be face_goal or scene")
        if motion_prior not in ("base", "footsteps"):
            raise ValueError("motion_prior must be base or footsteps")
        if motion_prior == "footsteps" and (task != "walk" or env_factory is not None):
            raise ValueError("footsteps is an explicit walk-only prior; do not also pass env_factory")
        validate_scene_seed(0 if split == "train" else HELD_OUT_SEED_BASE, split)
        self.stage, self.requested_task = stage, task
        self._split = split
        self.base_seed = _integer(base_seed, "base_seed")
        self.difficulty_range = _difficulty_range(difficulty)
        self.randomize = bool(randomize)
        self.max_episode_seconds = max_episode_seconds
        self.heading_mode, self.motion_prior = heading_mode, motion_prior
        self.env_factory = env_factory
        self.env_kwargs = dict(env_kwargs or {})
        if avatar_profile is not None and "avatar_profile" in self.env_kwargs:
            raise ValueError("avatar_profile must be supplied once, directly or through env_kwargs")
        profile_value = self.env_kwargs.pop("avatar_profile", avatar_profile)
        self._avatar_profile = load_avatar_profile(profile_value)
        self._body_model_sha256 = None
        if set(self.env_kwargs) & {"task", "scene", "randomize", "max_episode_seconds", "render_mode"}:
            raise ValueError("env_kwargs cannot override task, scene, randomization, time or render mode")
        self._requested_render_mode = render_mode
        self._scene_rng = np.random.default_rng(self.base_seed)
        self._episode_index = -1
        self._last_scene_seed = None
        self._qualified_success_seconds = 0.
        initial_seed = 0 if split == "train" else HELD_OUT_SEED_BASE
        scene, selected_task, goal = _prepare_scene(stage, initial_seed, self.difficulty_range[0], task, split,
                                                   np.random.default_rng(self.base_seed), None, heading_mode)
        self.current_scene, self.selected_task, self.selected_goal = scene, selected_task, goal
        super().__init__(self._build_env(scene, selected_task))
        self._scene_hash = self._hash_scene(scene)
        self._current_physics_seed = self.base_seed

    @property
    def split(self):
        return self._split

    @staticmethod
    def _hash_scene(scene):
        return hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
                              .encode("utf-8")).hexdigest()

    def _build_env(self, scene, task):
        factory = self.env_factory or HumanoidRoomEnv
        kwargs = dict(self.env_kwargs)
        if self._avatar_profile is not None:
            kwargs["avatar_profile"] = self._avatar_profile
        if self.motion_prior == "footsteps":
            from .locomotion import FootstepHumanoidEnv
            factory = FootstepHumanoidEnv
            kwargs["reference_start"] = False
        base = factory(task=task, scene=scene, randomize=self.randomize,
                       max_episode_seconds=self.max_episode_seconds, render_mode=self._requested_render_mode, **kwargs)
        if not isinstance(base.observation_space, spaces.Box) or base.observation_space.shape != (BASE_OBSERVATION_COUNT,):
            base.close()
            raise ValueError("env_factory must return the 59-input physical environment before sensors")
        if base.avatar_profile_sha256 != avatar_profile_sha256(self._avatar_profile):
            base.close()
            raise ValueError("env_factory must preserve the requested avatar body profile")
        if self._body_model_sha256 is not None and base.body_model_sha256 != self._body_model_sha256:
            base.close()
            raise ValueError("scene reset must preserve the physical body contract")
        self._body_model_sha256 = base.body_model_sha256
        # The fallback goal names the spawn floor, then the actual compiled
        # body's grounded standing pelvis height. Furniture/terrain stay fixed.
        goal = scene["goals"][0]
        if task == "stand" and "surfaceHeight" in goal:
            goal["position"][2] = float(goal["surfaceHeight"]) + base.standing_root_height
            base.goal[2] = goal["position"][2]
            base.scene["goals"][0]["position"][2] = goal["position"][2]
        wrapped = SceneSensorWrapper(base)
        if wrapped.observation_space.shape != (OBSERVATION_COUNT,) or wrapped.action_space.shape != (20,):
            wrapped.close()
            raise ValueError("scene policy must expose 110 observations and 20 physical joint actions")
        return wrapped

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options) - {"scene_seed", "physics_seed", "goal_id"}:
            raise ValueError("reset options may only select scene_seed, physics_seed or goal_id")
        if "scene_seed" in options:
            validate_scene_seed(options["scene_seed"], self.split)
        if seed is not None:
            self._scene_rng = np.random.default_rng(_integer(seed, "seed"))
            self._last_scene_seed = None
        low = 0 if self.split == "train" else HELD_OUT_SEED_BASE
        scene_seed = (validate_scene_seed(options["scene_seed"], self.split) if "scene_seed" in options
                      else int(self._scene_rng.integers(low, low + HELD_OUT_SEED_BASE)))
        if "scene_seed" not in options:
            while scene_seed == self._last_scene_seed:
                scene_seed = int(self._scene_rng.integers(low, low + HELD_OUT_SEED_BASE))
        difficulty = float(self._scene_rng.uniform(*self.difficulty_range))
        physics_seed = (_integer(options["physics_seed"], "physics_seed") if "physics_seed" in options
                        else int(self._scene_rng.integers(0, 2**31)))
        scene, task, goal = _prepare_scene(self.stage, scene_seed, difficulty, self.requested_task, self.split,
                                          self._scene_rng, options.get("goal_id"), self.heading_mode)
        new_env = self._build_env(scene, task)
        try:
            observation, info = new_env.reset(seed=physics_seed)
        except Exception:
            new_env.close()
            raise
        previous = self.env
        self.env = new_env
        previous.close()
        self.current_scene, self.selected_task, self.selected_goal = scene, task, goal
        self._scene_hash = self._hash_scene(scene)
        self._episode_index += 1
        self._last_scene_seed = scene_seed
        self._current_physics_seed = physics_seed
        self._qualified_success_seconds = 0.
        return observation, self._add_info(info, advance=False)

    def _task_metrics(self, info):
        base = self.unwrapped
        position = base.data.qpos[:3]
        goal = np.asarray(self.selected_goal["position"])
        surface_height = (float(self.selected_goal.get("surfaceHeight", goal[2] - base.standing_root_height))
                          if self.selected_task == "stand" else float(goal[2]))
        target_root_height = float(goal[2]) if self.selected_task == "stand" else float(
            goal[2] + (base.seat_pelvis_offset if self.selected_task == "sit" else base.standing_root_height))
        horizontal_error = float(np.linalg.norm(position[:2] - goal[:2]))
        vertical_error = abs(float(position[2]) - target_root_height)
        speed = float(np.linalg.norm(base.data.qvel[:3]))
        centre_surface = info["scene_sensors"]["floor_heights_m"][4]
        surface_matches = centre_surface is not None and abs(centre_surface - surface_height) < .04
        selected_seat_ids = set()
        if self.selected_task == "sit":
            entity = self.selected_goal.get("entityId")
            selected_seat_ids = {base.model.geom(geom["id"]).id for geom in self.current_scene["geoms"]
                                 if geom.get("semantic") in ("seat", "chair-seat", "chair_seat")
                                 and (geom.get("entityId") == entity if entity is not None
                                      else abs(geom["position"][2] + geom["size"][2] / 2 - surface_height) < .001)}
        seat_contact, obstacle_ids = False, set()
        semantics = {geom["id"]: geom.get("semantic", "furniture") for geom in self.current_scene["geoms"]}
        force = np.zeros(6)
        for index, contact in enumerate(base.data.contact):
            ids = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(base.model.geom_bodyid[g]) for g in ids)
            if 0 not in bodies:
                continue
            mujoco.mj_contactForce(base.model, base.data, index, force)
            if force[0] <= 1:
                continue
            if base.pelvis_body_id in bodies and any(g in selected_seat_ids for g in ids) and abs(contact.frame[2]) > .7:
                seat_contact = True
            for geom, body in zip(ids, bodies):
                if body == 0:
                    name = mujoco.mj_id2name(base.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    if semantics.get(name) not in _TERRAIN_SEMANTICS and geom not in selected_seat_ids:
                        obstacle_ids.add(name)
        support = (seat_contact if self.selected_task == "sit" else
                   bool(info["left_foot_contact"] and info["right_foot_contact"] and surface_matches))
        instant = bool(not info["fall"] and horizontal_error < (.15 if self.selected_task == "sit" else .25)
                       and vertical_error < (.10 if self.selected_task == "sit" else .15)
                       and speed < (.20 if self.selected_task == "sit" else .35)
                       and info["upright"] > .8 and support)
        return {"horizontal_error_m": horizontal_error, "target_root_height_m": target_root_height,
                "root_height_error_m": vertical_error, "goal_surface_height_m": surface_height,
                "measured_surface_height_m": centre_surface, "surface_level_matches": bool(surface_matches),
                "selected_seat_support_contact": seat_contact, "support_contact": support,
                "root_speed_m_s": speed, "obstacle_contact_geom_ids": sorted(obstacle_ids),
                "instant_success": instant, "base_success": bool(info.get("success", False))}

    def _add_info(self, info, *, advance):
        result = dict(info)
        metrics = self._task_metrics(info)
        if advance:
            self._qualified_success_seconds = (self._qualified_success_seconds + self.unwrapped.dt
                                               if metrics["instant_success"] else 0.)
        scene_meta = self.current_scene["metadata"]["generalization"]
        result["generalization"] = {
            **scene_meta, "episode_index": self._episode_index, "scene_sha256": self._scene_hash,
            "physics_seed": self._current_physics_seed, "difficulty": self.current_scene["metadata"]["difficulty"],
            "goal": copy.deepcopy(self.selected_goal), "observation_count": OBSERVATION_COUNT,
            "base_observation_count": BASE_OBSERVATION_COUNT, "sensor_feature_count": 51,
            "policy_input_contract": "base59_local_goal_xyz_task_bits_then_actual_scene_rays51_v1",
            "goal_heading_observed": False, "motion_prior": self.motion_prior,
            "base_env_class": type(self.unwrapped).__name__, "terrain_prior_supported": self.stage != 4,
            "reward_scope": "unchanged_base_reward; terrain_height_and_selected_seat_checked_separately",
            "root_step_control": "MuJoCo dynamics only", "scene_rebuilt_at_reset": True,
            "avatar_profile_sha256": self.unwrapped.avatar_profile_sha256,
            "body_model_sha256": self.unwrapped.body_model_sha256,
            "physical_assumptions": copy.deepcopy(self.unwrapped.physical_assumptions),
            "metrics": metrics, "qualified_success_seconds": self._qualified_success_seconds,
            "qualified_success": self._qualified_success_seconds >= _SUCCESS_HOLD_SECONDS,
        }
        return result

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self._add_info(info, advance=True)

    def snapshot(self):
        snapshot = self.unwrapped.snapshot()
        snapshot["scene"] = copy.deepcopy(self.current_scene)
        return snapshot


def make_generalization_env(stage=2, **kwargs) -> GeneralizationWrapper:
    """Factory suitable for a Monitor or SB3 vector-environment factory."""
    return GeneralizationWrapper(stage, **kwargs)


def _rollout(env, policy, scene_seed, physics_seed):
    observation, info = env.reset(options={"scene_seed": scene_seed, "physics_seed": physics_seed})
    if policy is not None:
        assert_policy_compatible(policy, env)
    initial = copy.deepcopy(info["generalization"])
    max_force = 0.
    collision_steps = 0
    reward_sum, steps = 0., 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = (np.zeros(env.action_space.shape, np.float32) if policy is None else
                  policy.predict(observation, deterministic=True)[0])
        observation, reward, terminated, truncated, info = env.step(action)
        base = env.unwrapped
        max_force = max(max_force, float(np.max(np.abs(base.data.xfrc_applied))),
                        float(np.max(np.abs(base.data.qfrc_applied))))
        collision_steps += bool(info["generalization"]["metrics"]["obstacle_contact_geom_ids"])
        reward_sum += reward
        steps += 1
    return {"stage": env.stage, "scene_seed": scene_seed, "physics_seed": physics_seed,
            "goal_id": initial["selectedGoalId"], "task": initial["selectedTask"],
            "scene_sha256": initial["scene_sha256"], "initial": initial,
            "physics_seconds": float(env.unwrapped.data.time), "steps": steps, "return": reward_sum,
            "fell": bool(info["fall"]), "terminated": bool(terminated), "truncated": bool(truncated),
            "qualified_success": bool(info["generalization"]["qualified_success"]),
            "max_external_force": max_force, "obstacle_contact_steps": collision_steps,
            "final": info["generalization"], "root_step_control": "MuJoCo dynamics only"}


def evaluate_generalization(policy=None, *, stages: Iterable[int] = (2, 3, 4),
                            scene_seeds: Iterable[int] = range(HELD_OUT_SEED_BASE, HELD_OUT_SEED_BASE + 8),
                            task="walk", difficulty=.5, seconds=12., randomize=True, motion_prior="base", base_seed=0,
                            env_factory=None, env_kwargs=None, avatar_profile=None) -> dict:
    """Report held-out physical failures; a run does not open proficiency gates."""
    seeds = [validate_scene_seed(seed, "heldout") for seed in scene_seeds]
    stages = list(stages)
    if not seeds or not stages:
        raise ValueError("evaluation requires at least one stage and held-out scene seed")
    episodes = []
    for stage in stages:
        env = make_generalization_env(stage, split="heldout", task=task, difficulty=difficulty,
                                      randomize=randomize, max_episode_seconds=seconds,
                                      motion_prior=motion_prior, base_seed=base_seed,
                                      env_factory=env_factory, env_kwargs=env_kwargs, avatar_profile=avatar_profile)
        try:
            if policy is not None:
                assert_policy_compatible(policy, env)
            for seed in seeds:
                episodes.append(_rollout(env, policy, seed, seed + 31))
        finally:
            env.close()
    stage_metrics = {str(stage): {
        "episode_count": sum(e["stage"] == stage for e in episodes),
        "qualified_success_rate": float(np.mean([e["qualified_success"] for e in episodes if e["stage"] == stage])),
        "fall_rate": float(np.mean([e["fell"] for e in episodes if e["stage"] == stage])),
        "proficiencyPassed": False, "learningStatus": "not_trained_or_validated",
    } for stage in stages}
    return {"version": 1, "evaluation": "heldout_actual_physics_with_scene_sensors", "split": "heldout",
            "controller": "110-input policy" if policy is not None else "zero-residual disclosed PD baseline",
            "observation_count": OBSERVATION_COUNT, "episode_count": len(episodes), "episodes": episodes,
            "stages": stage_metrics, "proficiencyPassed": False, "learningStatus": "not_trained_or_validated",
            "training_seed_domain": [0, HELD_OUT_SEED_BASE - 1],
            "heldout_seed_domain": [HELD_OUT_SEED_BASE, HELD_OUT_SEED_LIMIT - 1],
            "terrain_prior_supported": False if 4 in stages else True,
            "avatarProfileSha256": episodes[0]["initial"]["avatar_profile_sha256"],
            "bodyModelSha256": episodes[0]["initial"]["body_model_sha256"],
            "limitations": ["a task rollout does not verify gait quality or general skill",
                            "stage4 prior is not terrain-aware; elevated-goal success needs real level/contact",
                            "goal heading is metadata and is not an extra policy input"],
            "root_step_control": "MuJoCo dynamics only"}


def verify_generalization(*, stages=(2, 3, 4), episodes=10, split="train", base_seed=20261014,
                          task="walk", difficulty=.5, seconds=.2, motion_prior="base",
                          avatar_profile=None, env_kwargs=None) -> dict:
    """Small deterministic rebuilding/sensor audit, without training a policy."""
    episodes = _integer(episodes, "episodes")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    results = []
    for stage in stages:
        first = make_generalization_env(stage, split=split, task=task, base_seed=base_seed, difficulty=difficulty,
                                        randomize=False, max_episode_seconds=seconds, motion_prior=motion_prior,
                                        avatar_profile=avatar_profile, env_kwargs=env_kwargs)
        second = make_generalization_env(stage, split=split, task=task, base_seed=base_seed, difficulty=difficulty,
                                         randomize=False, max_episode_seconds=seconds, motion_prior=motion_prior,
                                         avatar_profile=avatar_profile, env_kwargs=env_kwargs)
        try:
            for _ in range(episodes):
                observation, info = first.reset()
                other_observation, other_info = second.reset()
                np.testing.assert_array_equal(observation, other_observation)
                assert info["generalization"]["scene_sha256"] == other_info["generalization"]["scene_sha256"]
                assert first.observation_space.contains(observation) and np.isfinite(observation).all()
                assert not np.any(first.unwrapped.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
                _, _, _, _, stepped = first.step(np.zeros(20, np.float32))
                assert not np.any(first.unwrapped.data.xfrc_applied) and not np.any(first.unwrapped.data.qfrc_applied)
                results.append({"stage": stage, "scene_seed": info["generalization"]["sceneSeed"],
                                "scene_sha256": info["generalization"]["scene_sha256"],
                                "goal_id": info["generalization"]["selectedGoalId"],
                                "observation_count": len(observation), "contact_count": stepped["contact_count"],
                                "avatarProfileSha256": info["generalization"]["avatar_profile_sha256"],
                                "bodyModelSha256": info["generalization"]["body_model_sha256"],
                                "physics_time": stepped["time"], "external_force": 0.})
        finally:
            first.close()
            second.close()
    return {"version": 1, "verification": "deterministic_episode_scene_rebuild_and_physical_sensor",
            "episodes": results, "episode_count": len(results), "split": split, "passed": True,
            "proficiencyPassed": False, "learningStatus": "not_trained_or_validated",
            "observation_count": OBSERVATION_COUNT, "root_step_control": "MuJoCo dynamics only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "evaluate"))
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", choices=("train", "heldout"))
    parser.add_argument("--base-seed", type=int, default=20261014)
    parser.add_argument("--task", choices=(*_TASKS, "mixed"), default="walk")
    parser.add_argument("--difficulty", type=float, default=.5)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--motion-prior", choices=("base", "footsteps"), default="base")
    parser.add_argument("--policy", help="Optional independently trained 110-input SB3 checkpoint")
    parser.add_argument("--avatar-profile", help="Measured physical avatar profile; policies must match its body contract")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.command == "verify":
        if args.policy:
            parser.error("verify does not load or train a policy")
        report = verify_generalization(stages=args.stages, episodes=args.episodes, split=args.split or "train",
                                        base_seed=args.base_seed, task=args.task, difficulty=args.difficulty,
                                        seconds=.2 if args.seconds is None else args.seconds, motion_prior=args.motion_prior,
                                        avatar_profile=args.avatar_profile)
    else:
        if args.split not in (None, "heldout"):
            parser.error("evaluate accepts only held-out scenes")
        policy = None
        if args.policy:
            from stable_baselines3 import PPO
            policy = PPO.load(args.policy, device="cpu")
        report = evaluate_generalization(policy, stages=args.stages,
                                          scene_seeds=range(HELD_OUT_SEED_BASE, HELD_OUT_SEED_BASE + args.episodes),
                                          task=args.task, difficulty=args.difficulty,
                                          seconds=12. if args.seconds is None else args.seconds,
                                          motion_prior=args.motion_prior, base_seed=args.base_seed,
                                          avatar_profile=args.avatar_profile)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
