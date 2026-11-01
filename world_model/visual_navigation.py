"""Depth-image navigation commands above the unchanged, frozen 5aa joint motor.

The high controller receives a finite-FOV depth stack and eleven goal/body
values. The existing low motor retains its original actual sensor51 input;
this is vision-based high-level control, not a vision-only joint policy.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .body_contract import assert_policy_body
from .command_motor import CommandMotorEnv, assert_command_motor_policy
from .command_motor_training import _state_sha256
from .navigation import NAVIGATION_SUCCESS_BONUS, OBSTACLE_FAILURE_TERMINAL_COST


FROZEN_MOTOR_CHECKPOINT_SHA256 = "5aa9536b35defac5d715f98c75f558f6f5cdaf9a26344672d02e7e4d4358252b"
VISUAL_OBSERVATION_CONTRACT = {
    "version": 1, "role": "finite-FOV depth high controller with goal/body proprioception",
    "keys": ["depth", "state"], "depthShape": [3, 48, 64],
    "depthNormalization": "clipped first-intersection pinhole ray range, normalized near=0/far or no hit=1",
    "depthStackOrder": "oldest to newest; repeat the first actual image at reset",
    "stateCount": 11,
    "stateOrder": "local actual final-goal xyz3, actual root local linear velocity3, local up3, effective heading/bound, previous pace action",
    "statePositionVelocityClip": 5.0, "actionCount": 2,
    "sceneInputs": "depth image stack only; no sparse rays, map, route, geometry IDs or teacher path",
    "goalAndBodyStateSource": "actual simulator final goal and physical body; no image-based localization claim",
    "lowMotorObservationCount": 110, "lowMotorActionCount": 20,
    "lowMotorStillUsesActualSparseGeometrySensor51": True,
}


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def capture_depth(body, config):
    """Lazy camera boundary, also allowing a declared synthetic sensor in tests."""
    from .depth_perception import capture_depth as capture
    return capture(body, config)


class _ExternalCommandMotorEnv(CommandMotorEnv):
    """Replace only the command generator; retain the original physical step."""

    def __init__(self, **kwargs):
        self.requested_heading_offset = 0.
        self.requested_lookahead_m = 0.
        self.previous_pace_action = -1.
        super().__init__(**kwargs)
        self.training_command_generator_contract.update({
            "role": "external two-action visual-navigation command driver; no scene planner",
            "geometryPathsOrWaypointsUsed": False,
            "highLevelActionSuppliedByCaller": True,
            "analyticFinalGoalBearingDriverUsed": False,
            "zeroLookaheadForExplicitStopAllowed": True,
        })

    def _update_next_command(self, *, advance):
        body = self.unwrapped
        if advance:
            maximum = self.heading_slew_radians_per_second * body.dt
            self._effective_heading_offset += float(np.clip(
                self.requested_heading_offset-self._effective_heading_offset, -maximum, maximum))
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        heading = math.atan2(rotation[1, 0], rotation[0, 0]) + self._effective_heading_offset
        goal_distance = float(np.linalg.norm(body.goal[:2]-body.data.qpos[:2]))
        distance = min(self.requested_lookahead_m, goal_distance)
        self._next_command_goal = np.asarray([
            body.data.qpos[0]+distance*math.sin(heading),
            body.data.qpos[1]-distance*math.cos(heading), body.goal[2]], dtype=np.float64)
        body.command_goal = self._next_command_goal.copy()
        self._command = {
            "source": "external visual-navigation action; no internal geometry planner",
            "requestedHeadingOffsetRadians": self.requested_heading_offset,
            "effectiveHeadingOffsetRadians": self._effective_heading_offset,
            "headingRadians": heading, "requestedLookaheadMeters": self.requested_lookahead_m,
            "effectiveLookaheadMeters": distance,
            "worldCommandGoal": self._next_command_goal.tolist(),
            "actualFinalGoal": body.goal.tolist(), "geometryPathTeacherUsed": False,
        }


def assert_visual_policy(policy, env):
    """Require the exact new image/body/action/motor contract and same avatar."""
    if policy is None:
        return
    observation = getattr(policy, "observation_space", None)
    if not isinstance(observation, spaces.Dict) or set(observation.spaces) != {"depth", "state"}:
        raise ValueError("visual navigation requires exactly depth and state dictionary inputs")
    for key in ("depth", "state"):
        actual, expected = observation[key], env.observation_space[key]
        if (not isinstance(actual, spaces.Box) or actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or not np.array_equal(actual.low, expected.low)
                or not np.array_equal(actual.high, expected.high)):
            raise ValueError(f"visual navigation {key} observation bounds or dimensions differ")
    action = getattr(policy, "action_space", None)
    if (not isinstance(action, spaces.Box) or action.shape != (2,)
            or action.dtype != env.action_space.dtype
            or not np.array_equal(action.low, env.action_space.low)
            or not np.array_equal(action.high, env.action_space.high)):
        raise ValueError("visual navigation requires two bounded command actions")
    if getattr(policy, "visual_observation_contract", None) != env.observation_contract:
        raise ValueError("visual navigation observation semantics differ")
    if getattr(policy, "visual_control_contract", None) != env.control_contract:
        raise ValueError("visual navigation camera, frozen motor or control contract differs")
    if getattr(policy, "gamma", None) != .99:
        raise ValueError("visual navigation requires gamma .99 per actual 100ms decision")
    if hasattr(policy, "command_observation_contract") or hasattr(policy, "navigation_observation_contract"):
        raise ValueError("legacy navigation contracts require an explicit modality transfer")
    assert_policy_body(policy, env.unwrapped)


class VisualNavigationEnv(gym.Wrapper):
    """Two high actions per 100ms, actual frozen-motor torques every 20ms."""

    def __init__(self, *, motor_checkpoint, decision_steps=5, depth_config=None, depth_stack=3,
                 max_heading_radians=.45, max_lookahead_m=.5,
                 heading_slew_radians_per_second=.175, record_trajectory=False, **kwargs):
        if isinstance(decision_steps, bool) or not isinstance(decision_steps, int) or decision_steps != 5:
            raise ValueError("visual navigation requires exactly five physical controls per 10Hz decision")
        if isinstance(depth_stack, bool) or not isinstance(depth_stack, int) or depth_stack not in (2, 3):
            raise ValueError("visual depth stack must contain two or three actual camera frames")
        max_heading_radians = _finite(max_heading_radians, "heading bound")
        max_lookahead_m = _finite(max_lookahead_m, "walking lookahead")
        heading_slew_radians_per_second = _finite(heading_slew_radians_per_second, "heading slew")
        if not 0 < max_heading_radians <= .45:
            raise ValueError("visual heading bound must remain in (0, .45]")
        if not .35 <= max_lookahead_m <= .5:
            raise ValueError("visual walking lookahead must remain in [.35, .5]")
        if heading_slew_radians_per_second != .175:
            raise ValueError("visual control must preserve the frozen motor's .175 heading slew")
        forbidden = {"env_factory", "command_lookahead_range", "reward_scale", "motor_hip_yaw_action_limit"}
        if forbidden.intersection(kwargs):
            raise ValueError("visual navigation cannot replace the frozen motor's physical factory or contract")
        if kwargs.get("terminate_on_success", True) is not True:
            raise ValueError("visual navigation must terminate immediately on qualified success")
        from .depth_perception import DepthCameraConfig
        self.depth_config = DepthCameraConfig() if depth_config is None else depth_config
        if not isinstance(self.depth_config, DepthCameraConfig):
            raise ValueError("visual navigation requires a DepthCameraConfig")
        if (self.depth_config.height, self.depth_config.width) != (48, 64):
            raise ValueError("visual navigation uses actual depth images of 48 by 64 pixels")
        self.motor_checkpoint = Path(motor_checkpoint)
        self.motor_checkpoint_sha256 = hashlib.sha256(self.motor_checkpoint.read_bytes()).hexdigest()
        if self.motor_checkpoint_sha256 != FROZEN_MOTOR_CHECKPOINT_SHA256:
            raise ValueError("visual navigation requires the immutable trained 5aa command motor")
        if kwargs.get("avatar_profile") is None:
            kwargs["avatar_profile"] = Path(__file__).resolve().parents[1]/"frontend/public/world-model/gakucho-avatar-profile.json"
        kwargs.setdefault("mode", "obstacle")
        self.motor_env = _ExternalCommandMotorEnv(command_lookahead_range=(.35, .5),
            max_heading_radians=.45, heading_slew_radians_per_second=.175,
            record_trajectory=bool(record_trajectory), **kwargs)
        super().__init__(self.motor_env)
        self.navigation_env = self.motor_env.navigation_env
        self.decision_steps = int(decision_steps)
        self.depth_stack = int(depth_stack)
        self.max_heading_radians = max_heading_radians
        self.max_lookahead_m = max_lookahead_m
        self.min_walking_lookahead_m = .35
        self.stop_action_threshold = -.8
        self.heading_slew_radians_per_second = heading_slew_radians_per_second
        self.reward_scale = .01
        self.record_trajectory = bool(record_trajectory)
        self._has_reset = False
        self._done = False
        self._depth_frames = deque(maxlen=self.depth_stack)
        self._latest_depth = None
        self._depth_metadata = None
        self._depth_sample_count = 0
        self._last_reward_summary = None
        self._last_control_steps = self._last_physics_substeps = 0
        self.last_control_frames = []
        try:
            import torch
            from stable_baselines3 import PPO
            torch.set_num_threads(1)
            self._motor = PPO.load(self.motor_checkpoint, device="cpu")
            assert_command_motor_policy(self._motor, self.motor_env)
            self._motor.policy.set_training_mode(False)
            for parameter in self._motor.policy.parameters():
                parameter.requires_grad_(False)
            self._motor_tensor_sha256 = _state_sha256(self._motor.policy.state_dict())
            self._motor_optimizer_sha256 = _state_sha256(self._motor.policy.optimizer.state_dict())
            self._motor_counts = self._motor_training_counts()
            if self.unwrapped.dt != .02:
                raise ValueError("the trained visual motor requires actual physical controls every .02 seconds")
        except Exception:
            self.motor_env.close()
            raise
        self.observation_contract = copy.deepcopy(VISUAL_OBSERVATION_CONTRACT)
        self.observation_contract["depthShape"] = [self.depth_stack, 48, 64]
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(0., 1., (self.depth_stack, 48, 64), dtype=np.float32),
            "state": spaces.Box(np.asarray([-5.]*6+[-1.]*5, dtype=np.float32),
                                np.asarray([5.]*6+[1.]*5, dtype=np.float32), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.control_contract = {
            "version": 1, "role": "depth high-level navigation over frozen same-avatar command motor",
            "motorCheckpointSha256": self.motor_checkpoint_sha256,
            "motorControlContract": copy.deepcopy(self.motor_env.control_contract),
            "motorInputCount": 110, "motorJointActionCount": 20,
            "motorWeightsFrozen": True, "motorOptimizerAndTrainingCountersFrozen": True,
            "lowMotorUsesActualSparseGeometrySensor51": True,
            "highControllerUsesSparseRaysOrWorldMap": False,
            # SB3 serializes metadata through JSON: use its persistent array
            # representation now so a save/load preserves exact contracts.
            "depthCameraConfig": {**asdict(self.depth_config),
                "mount_offset_local": list(self.depth_config.mount_offset_local)},
            "depthStack": self.depth_stack,
            "cameraSamplingHz": 10., "depthShape": [self.depth_stack, 48, 64],
            "stateCount": 11, "navigatorActionCount": 2,
            "maxHeadingRadians": self.max_heading_radians,
            "headingSlewRadiansPerSecond": self.heading_slew_radians_per_second,
            "headingBasis": "current actual pelvis heading; effective offset advances after each physical control",
            "minWalkingLookaheadMeters": .35, "maxLookaheadMeters": self.max_lookahead_m,
            "stopLookaheadActionThreshold": self.stop_action_threshold,
            "lookaheadMapping": "a1 below -.8 gives zero; otherwise linear [.35,max] at [-.8,1]; capped at actual final-goal distance",
            "zeroCommandScope": "explicit stop; source motor's walking range [.35,.5] and near-goal distance cap unchanged",
            "decisionControlSteps": self.decision_steps, "physicalControlDtSeconds": .02,
            "physicsSubstepsPerControl": 10, "decisionDtSeconds": .1,
            "rewardDiscountPerDecision": .99, "rewardScale": self.reward_scale,
            "terminalRewardScale": self.decision_steps,
            "highReward": "sum actual unscaled navigation task rewards, add (decisionSteps-1)*(success bonus-failure cost), then scale .01; exclude intrinsic low motor tracking reward",
            "privilegedGeometryMayShapeTaskReward": True,
            "runtimeGeometryPlannerUsed": False, "rootStateControl": "MuJoCo dynamics and existing bounded joint torques only",
        }

    @property
    def scene(self):
        return self.navigation_env.scene

    @property
    def avatar_profile(self):
        return self.navigation_env.avatar_profile

    @property
    def avatar_profile_sha256(self):
        return self.navigation_env.avatar_profile_sha256

    @property
    def body_model_sha256(self):
        return self.navigation_env.body_model_sha256

    @property
    def latest_depth(self):
        return None if self._latest_depth is None else self._latest_depth.copy()

    @property
    def depth_metadata(self):
        return copy.deepcopy(self._depth_metadata)

    def _motor_training_counts(self):
        return tuple(getattr(self._motor, name, None) for name in (
            "num_timesteps", "command_motor_physical_control_timesteps", "_n_updates",
            "command_motor_reward_v3_initial_physical_controls", "command_motor_curriculum_initial_physical_controls"))

    def assert_frozen_motor(self, *, check_checkpoint=False):
        if (self._motor_training_counts() != self._motor_counts
                or any(parameter.requires_grad for parameter in self._motor.policy.parameters())
                or _state_sha256(self._motor.policy.state_dict()) != self._motor_tensor_sha256
                or _state_sha256(self._motor.policy.optimizer.state_dict()) != self._motor_optimizer_sha256):
            raise RuntimeError("visual navigation changed the frozen motor weights, Adam or training counts")
        assert_command_motor_policy(self._motor, self.motor_env)
        if check_checkpoint and hashlib.sha256(self.motor_checkpoint.read_bytes()).hexdigest() != self.motor_checkpoint_sha256:
            raise RuntimeError("the frozen source motor checkpoint changed")

    def _capture_depth(self, *, reset=False):
        body = self.unwrapped
        names = ("qpos", "qvel", "qacc", "ctrl", "xmat", "xfrc_applied", "qfrc_applied")
        before = {name: getattr(body.data, name).copy() for name in names}
        before_time = float(body.data.time)
        groups = body.model.geom_group.copy()
        image, metadata = capture_depth(body, self.depth_config)
        if (float(body.data.time) != before_time
                or any(not np.array_equal(before[name], getattr(body.data, name)) for name in names)
                or not np.array_equal(groups, body.model.geom_group)):
            raise RuntimeError("capturing depth changed actual physics state or collision/render groups")
        image = np.asarray(image)
        if (image.shape != (1, 48, 64) or image.dtype != np.float32
                or not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1)
                or not isinstance(metadata, dict)):
            raise ValueError("depth capture must return finite normalized float32 [1,48,64] and metadata")
        self._latest_depth = image.copy()
        self._depth_sample_count += 1
        self._depth_metadata = {"sampleIndex": self._depth_sample_count,
            "actualSimTimeSeconds": before_time, "samplingHz": 10.,
            "capture": copy.deepcopy(metadata)}
        if reset:
            self._depth_frames.clear()
            for _ in range(self.depth_stack):
                self._depth_frames.append(image[0].copy())
        else:
            self._depth_frames.append(image[0].copy())

    def _observation(self):
        body = self.unwrapped
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        goal = rotation.T @ (body.goal-body.data.qpos[:3])
        velocity = rotation.T @ body.data.qvel[:3]
        up = rotation.T @ np.asarray([0., 0., 1.])
        state = np.concatenate((np.clip(goal, -5., 5.), np.clip(velocity, -5., 5.),
            np.clip(up, -1., 1.),
            [self.motor_env._effective_heading_offset/self.max_heading_radians,
             self.motor_env.previous_pace_action])).astype(np.float32)
        return {"depth": np.stack(self._depth_frames).astype(np.float32, copy=False), "state": state}

    def _add_visual_info(self, info):
        result = dict(info)
        result["command_control_steps"] = self._last_control_steps
        result["command_physics_substeps"] = self._last_physics_substeps
        result["visual_navigation"] = {
            "observationContract": copy.deepcopy(self.observation_contract),
            "controlContract": copy.deepcopy(self.control_contract),
            "depth": self.depth_metadata, "reward": copy.deepcopy(self._last_reward_summary),
            "finalGoalFixed": True, "motorWeightsFrozen": True,
            "lowMotorUsesSparseGeometrySensor51": True,
            "highControllerReceivesWorldMapOrSparseRays": False,
            "runtimeGeometryPlannerUsed": False, "rootPositionOverwrittenDuringStep": False,
            "effectiveHeadingOffsetRadians": self.motor_env._effective_heading_offset,
            "previousLookaheadAction": self.motor_env.previous_pace_action,
        }
        return result

    def reset(self, *, seed=None, options=None):
        self._has_reset = False
        self._done = False
        self._last_reward_summary = None
        self._last_control_steps = self._last_physics_substeps = 0
        self._depth_sample_count = 0
        self.last_control_frames = []
        self.motor_env.requested_heading_offset = 0.
        self.motor_env.requested_lookahead_m = 0.
        self.motor_env.previous_pace_action = -1.
        _, info = self.motor_env.reset(seed=seed, options=options)
        self.assert_frozen_motor()
        self._capture_depth(reset=True)
        self._has_reset = True
        return self._observation(), self._add_visual_info(info)

    def step(self, action):
        if not self._has_reset or self._done:
            raise RuntimeError("reset() is required before a visual navigation step")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("visual navigation requires two finite actions")
        action = np.clip(action, -1., 1.)
        self.motor_env.requested_heading_offset = float(action[0])*self.max_heading_radians
        self.motor_env.previous_pace_action = float(action[1])
        self.motor_env.requested_lookahead_m = (0. if action[1] < self.stop_action_threshold else
            .35+(float(action[1])-self.stop_action_threshold)/(1.-self.stop_action_threshold)
            *(self.max_lookahead_m-.35))
        self.motor_env._update_next_command(advance=False)
        motor_observation = self.motor_env._observation()
        self._last_control_steps = self._last_physics_substeps = 0
        self.last_control_frames = []
        physical_reward_sum = terminal_adjustment = 0.
        body = self.unwrapped
        goal_object, goal_value = body.goal, body.goal.copy()
        for _ in range(self.decision_steps):
            motor_action = self._motor.predict(motor_observation, deterministic=True)[0]
            motor_observation, _, terminated, truncated, info = self.motor_env.step(motor_action)
            observed = info["command_motor"]["observedPhysicsSubstepsThisControl"]
            if isinstance(observed, bool) or observed != 10:
                raise RuntimeError("visual navigation missed actual 2ms physical substeps")
            if body.goal is not goal_object or not np.array_equal(body.goal, goal_value):
                raise RuntimeError("visual commands changed the actual final destination")
            if np.any(body.data.xfrc_applied) or np.any(body.data.qfrc_applied):
                raise RuntimeError("visual navigation applied external body forces")
            navigation_reward = info["navigation"]["reward"]
            raw = float(info["command_motor"]["inheritedNavigationReward"])
            bonus, cost = float(navigation_reward["success_bonus"]), float(navigation_reward["terminal_cost"])
            expected_raw = (float(navigation_reward["unclamped_reward"])
                + float(navigation_reward["reward_clamp_adjustment"])-cost)
            if not all(math.isfinite(value) for value in (raw, bonus, cost, expected_raw)) or raw != expected_raw:
                raise RuntimeError("visual task reward diverged from actual unscaled navigation reward")
            physical_reward_sum += raw
            terminal_adjustment += (self.decision_steps-1)*(bonus-cost)
            self._last_control_steps += 1
            self._last_physics_substeps += int(observed)
            total = (physical_reward_sum+terminal_adjustment)*self.reward_scale
            self._last_reward_summary = {
                "physicalControlRewardSum": physical_reward_sum,
                "terminalRewardAdjustment": terminal_adjustment,
                "unscaledCommandDecisionReward": physical_reward_sum+terminal_adjustment,
                "commandDecisionReward": total, "rewardScale": self.reward_scale,
                "executedPhysicalControlSteps": self._last_control_steps,
                "observedPhysicsSubsteps": self._last_physics_substeps,
                "terminalFailureCost": OBSTACLE_FAILURE_TERMINAL_COST*self.decision_steps*self.reward_scale,
                "terminalSuccessBonus": NAVIGATION_SUCCESS_BONUS*self.decision_steps*self.reward_scale,
                "negativeContinuingRewardLowerBound": -8.*self.decision_steps/(1.-.99)*self.reward_scale,
                "lowIntrinsicTrackingRewardUsed": False,
            }
            if self.record_trajectory:
                frames = self.motor_env.last_control_frames
                if len(frames) != 1:
                    raise RuntimeError("visual navigation requires every actual 20ms control snapshot")
                frame = dict(frames[0])
                frame["stats"] = self._add_visual_info(frame["stats"])
                self.last_control_frames.append(frame)
            if terminated or truncated:
                break
        self._done = bool(terminated or truncated)
        self._capture_depth()
        if self.last_control_frames:
            # The final saved native state already exists. Refresh only the
            # 10Hz visual metadata; do not resample physics or sparse sensors.
            self.last_control_frames[-1]["stats"] = self._add_visual_info(self.last_control_frames[-1]["stats"])
        self.assert_frozen_motor()
        return self._observation(), total, bool(terminated), bool(truncated), self._add_visual_info(info)

    def snapshot(self):
        if not self._has_reset:
            raise RuntimeError("reset() is required before a visual navigation snapshot")
        frame = self.motor_env.snapshot()
        frame["stats"] = self._add_visual_info(frame["stats"])
        return frame
