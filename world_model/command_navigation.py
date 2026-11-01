"""Learn direction/pace above a frozen, same-avatar physical walking motor.

The navigator sees the real final destination and real geometry sensors.
Only the motor's reference computation and its three local goal features use
the requested short-horizon command. The actual task goal, free-body state,
contact observer, and torque limits are never replaced by those commands.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .body_contract import assert_policy_body
from .command_gait import command_gait_pose
from .navigation import EpisodeNavigationEnv, NAVIGATION_SUCCESS_BONUS, OBSTACLE_FAILURE_TERMINAL_COST
from .navigation_training import assert_navigation_policy
from .steering import SteeringHumanoidEnv

COMMAND_OBSERVATION_CONTRACT = {
    "version": 1,
    "observationCount": 112,
    "physicalAndSceneObservationCount": 110,
    "commandStateCount": 2,
    "actionCount": 2,
    "motorObservationCount": 110,
    "motorActionCount": 20,
    "physicalAndSceneObservation": "unchanged real-final-goal base59 plus actual geometry sensor51",
    "commandState": "effective body-relative heading offset normalized by bound; previous lookahead action",
    "actions": "body-relative heading offset and short-horizon walking distance",
}


class CommandSteeringHumanoidEnv(SteeringHumanoidEnv):
    """Separate command targets with pace and physical foot lift decoupled."""

    def __init__(self, **kwargs):
        self.command_goal = None
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self.command_goal = None
        return super().reset(**kwargs)

    def _reference_pose(self):
        if self.command_goal is None:
            return super()._reference_pose()
        return command_gait_pose(self, self.command_goal)


class _CommandTaskNavigationEnv(EpisodeNavigationEnv):
    def _metrics(self, info):
        metrics = super()._metrics(info)
        swings = self.unwrapped._swing_counts.tolist()
        metrics["actual_gait_swing_counts"] = swings
        metrics["minimum_required_swing_count_per_foot"] = 2
        # Training uses actual sustained, supported swing events. The final
        # evaluator also checks independent event clustering and alternation.
        metrics["instant_success"] = bool(metrics["instant_success"] and min(swings) >= 2)
        return metrics


def assert_command_policy(policy, env):
    if policy is None:
        return
    if (not isinstance(policy.observation_space, spaces.Box)
            or policy.observation_space.shape != (112,)
            or not isinstance(policy.action_space, spaces.Box)
            or policy.action_space.shape != (2,)
            or not np.array_equal(policy.action_space.low, env.action_space.low)
            or not np.array_equal(policy.action_space.high, env.action_space.high)):
        raise ValueError("command navigation requires 112 observations and two bounded actions")
    if getattr(policy, "command_observation_contract", None) != COMMAND_OBSERVATION_CONTRACT:
        raise ValueError("command navigation observation semantics differ")
    if (not np.array_equal(policy.observation_space.low, env.observation_space.low)
            or not np.array_equal(policy.observation_space.high, env.observation_space.high)):
        raise ValueError("command navigation observation bounds differ")
    if getattr(policy, "command_control_contract", None) != env.control_contract:
        raise ValueError("command navigation motor or control parameters differ")
    if getattr(policy, "gamma", None) != env.control_contract["rewardDiscountPerDecision"]:
        raise ValueError("command navigation discount horizon differs")
    assert_policy_body(policy, env.unwrapped)


class CommandNavigationEnv(gym.Wrapper):
    """Two learned commands per 100ms; frozen joint motor runs every 20ms."""

    def __init__(self, *, motor_checkpoint, decision_steps=5,
                 max_heading_radians=.45, max_lookahead_m=.5,
                 heading_slew_radians_per_second=.175,
                 motor_hip_yaw_action_limit=None,
                 record_trajectory=False, **kwargs):
        if isinstance(decision_steps, bool) or not isinstance(decision_steps, int) or not 1 <= decision_steps <= 10:
            raise ValueError("decision_steps must be an integer in 1..10")
        if not (math.isfinite(max_heading_radians) and 0 < max_heading_radians <= .6):
            raise ValueError("heading bound must be finite and in (0, .6]")
        if not (math.isfinite(max_lookahead_m) and .35 <= max_lookahead_m <= .5):
            raise ValueError("walking lookahead must stay in the verified [.35, .5] range")
        if not (math.isfinite(heading_slew_radians_per_second) and 0 < heading_slew_radians_per_second <= .35):
            raise ValueError("heading slew must be in (0, .35]")
        if motor_hip_yaw_action_limit is not None and (
                isinstance(motor_hip_yaw_action_limit, bool)
                or not math.isfinite(motor_hip_yaw_action_limit)
                or not 0 < motor_hip_yaw_action_limit <= 1):
            raise ValueError("motor hip-yaw action limit must be finite and in (0, 1]")
        if "env_factory" in kwargs:
            raise ValueError("the command motor factory is fixed")
        self.navigation_env = _CommandTaskNavigationEnv(env_factory=CommandSteeringHumanoidEnv, **kwargs)
        super().__init__(self.navigation_env)
        self.decision_steps = decision_steps
        self.max_heading_radians = float(max_heading_radians)
        self.max_lookahead_m = float(max_lookahead_m)
        self.min_walking_lookahead_m = .35
        self.stop_action_threshold = -.8
        self.reward_scale = .01
        self.heading_slew_radians_per_second = float(heading_slew_radians_per_second)
        self.record_trajectory = bool(record_trajectory)
        self.motor_hip_yaw_action_limit = motor_hip_yaw_action_limit
        self.motor_checkpoint = Path(motor_checkpoint)
        try:
            self.motor_checkpoint_sha256 = hashlib.sha256(self.motor_checkpoint.read_bytes()).hexdigest()
            import torch
            from stable_baselines3 import PPO
            torch.set_num_threads(1)
            self._motor = PPO.load(self.motor_checkpoint, device="cpu")
            assert_navigation_policy(self._motor, self.navigation_env)
            self._motor.policy.set_training_mode(False)
            for parameter in self._motor.policy.parameters():
                parameter.requires_grad_(False)
        except Exception:
            self.navigation_env.close()
            raise
        self.observation_space = spaces.Box(
            low=np.concatenate([self.env.observation_space.low, [-1., -1.]]).astype(np.float32),
            high=np.concatenate([self.env.observation_space.high, [1., 1.]]).astype(np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.control_contract = {
            "version": 1,
            "motorCheckpointSha256": self.motor_checkpoint_sha256,
            "motorInputCount": 110, "motorJointActionCount": 20,
            "motorGoalFeatures": "explicitly transferred command-goal xyz in indices 50:53; real sensor51 unchanged",
            "motorReference": "command_gait_pose: T2M cadence, pace-scaled stride, retained swing-lift plateau and lateral unloading",
            "headingBasis": "current physical pelvis heading",
            "maxHeadingRadians": self.max_heading_radians,
            "maxLookaheadMeters": self.max_lookahead_m,
            "minWalkingLookaheadMeters": self.min_walking_lookahead_m,
            "stopLookaheadActionThreshold": self.stop_action_threshold,
            "lookaheadMapping": "explicit zero below stop threshold; continuous min-to-max verified walking range above it; capped near actual goal",
            "headingSlewRadiansPerSecond": self.heading_slew_radians_per_second,
            "decisionControlSteps": self.decision_steps,
            "physicalControlDtSeconds": float(self.unwrapped.dt),
            "decisionDtSeconds": float(self.unwrapped.dt) * self.decision_steps,
            "terminalRewardScale": self.decision_steps,
            "rewardDiscountPerDecision": .99,
            "rewardScale": self.reward_scale,
            "trainingSuccessRequiresActualSwingsPerFoot": 2,
            "motorWeightsFrozen": True, "fixedPathController": False,
            "goalDistanceCappedLookahead": True,
            "rootStepControl": "MuJoCo dynamics only",
        }
        if self.motor_hip_yaw_action_limit is not None:
            self.control_contract.update(version=2,
                motorHipYawResidualActionLimit=float(self.motor_hip_yaw_action_limit))
        self._effective_heading_offset = 0.
        self._last_lookahead_action = -1.
        self._last_command = None
        self._last_reward_summary = None
        self._physical_observation = None
        self._done = False
        self.last_control_frames = []

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

    def _observation(self):
        return np.concatenate([self._physical_observation,
            [self._effective_heading_offset/self.max_heading_radians, self._last_lookahead_action]]).astype(np.float32)

    def _add_command_info(self, info, *, control_steps=0):
        result = dict(info)
        result["command_control_steps"] = int(control_steps)
        result["command_navigation"] = {
            "controlContract": copy.deepcopy(self.control_contract),
            "observationContract": copy.deepcopy(COMMAND_OBSERVATION_CONTRACT),
            "effectiveHeadingOffsetRadians": self._effective_heading_offset,
            "previousLookaheadAction": self._last_lookahead_action,
            "command": copy.deepcopy(self._last_command),
            "finalGoalFixed": True, "motorWeightsFrozen": True,
            "physicalJointTargets": 20, "navigatorActions": 2,
            "rootPositionOverwrittenDuringStep": False,
            "reward": copy.deepcopy(self._last_reward_summary),
        }
        return result

    def reset(self, *, seed=None, options=None):
        self._effective_heading_offset = 0.
        self._last_lookahead_action = -1.
        self._last_command = None
        self._last_reward_summary = None
        self._done = False
        self.last_control_frames = []
        observation, info = self.env.reset(seed=seed, options=options)
        assert_navigation_policy(self._motor, self.navigation_env)
        self._physical_observation = observation
        return self._observation(), self._add_command_info(info)

    def step(self, action):
        if self._done or self._physical_observation is None:
            raise RuntimeError("reset() is required before a command navigation step")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("command navigation requires two finite actions")
        action = np.clip(action, -1., 1.)
        requested_heading_offset = float(action[0]) * self.max_heading_radians
        self._last_lookahead_action = float(action[1])
        requested_distance = (0. if action[1] < self.stop_action_threshold else
            self.min_walking_lookahead_m + (float(action[1])-self.stop_action_threshold)
            / (1.-self.stop_action_threshold) * (self.max_lookahead_m-self.min_walking_lookahead_m))
        self.last_control_frames = []
        total_reward = 0.
        physical_reward_sum = 0.
        terminal_reward_adjustment = 0.
        control_steps = 0
        body = self.unwrapped
        actual_goal = body.goal
        goal_copy = actual_goal.copy()
        for _ in range(self.decision_steps):
            maximum_change = self.heading_slew_radians_per_second * body.dt
            self._effective_heading_offset += float(np.clip(
                requested_heading_offset-self._effective_heading_offset, -maximum_change, maximum_change))
            rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
            heading = math.atan2(rotation[1, 0], rotation[0, 0]) + self._effective_heading_offset
            distance = min(requested_distance, float(np.linalg.norm(actual_goal[:2]-body.data.qpos[:2])))
            command_goal = np.array([body.data.qpos[0]+distance*math.sin(heading),
                                     body.data.qpos[1]-distance*math.cos(heading), actual_goal[2]])
            body.command_goal = command_goal
            self._last_command = {"headingRadians": heading, "lookaheadMeters": requested_distance,
                                  "effectiveLookaheadMeters": distance, "position": command_goal.tolist()}
            motor_observation = self._physical_observation.copy()
            motor_observation[50:53] = rotation.T @ (command_goal-body.data.qpos[:3])
            motor_action = self._motor.predict(motor_observation, deterministic=True)[0]
            if self.motor_hip_yaw_action_limit is not None:
                from .command_motor_limits import bounded_motor_actions
                motor_action = bounded_motor_actions(motor_action, self.motor_hip_yaw_action_limit)
            observation, reward, terminated, truncated, info = self.env.step(motor_action)
            if body.goal is not actual_goal or not np.array_equal(body.goal, goal_copy):
                raise RuntimeError("a motor command changed the actual task destination")
            if np.any(body.data.xfrc_applied) or np.any(body.data.qfrc_applied):
                raise RuntimeError("command navigation unexpectedly applied an external force")
            self._physical_observation = observation
            physical_reward_sum += float(reward)
            reward_info = info["navigation"]["reward"]
            terminal_reward_adjustment += (self.decision_steps-1) * (
                float(reward_info.get("success_bonus", 0.)) - float(reward_info.get("terminal_cost", 0.)))
            total_reward = (physical_reward_sum + terminal_reward_adjustment) * self.reward_scale
            control_steps += 1
            self._last_reward_summary = {
                "physicalControlRewardSum": physical_reward_sum,
                "terminalRewardAdjustment": terminal_reward_adjustment,
                "commandDecisionReward": total_reward,
                "unscaledCommandDecisionReward": physical_reward_sum + terminal_reward_adjustment,
                "rewardScale": self.reward_scale,
                "executedPhysicalControlSteps": control_steps,
                "unscaledTerminalFailureCost": OBSTACLE_FAILURE_TERMINAL_COST * self.decision_steps,
                "terminalFailureCost": OBSTACLE_FAILURE_TERMINAL_COST * self.decision_steps * self.reward_scale,
                "unscaledTerminalSuccessBonus": NAVIGATION_SUCCESS_BONUS * self.decision_steps,
                "terminalSuccessBonus": NAVIGATION_SUCCESS_BONUS * self.decision_steps * self.reward_scale,
                "unscaledNegativeContinuingRewardLowerBound": -8. * self.decision_steps / (1.-.99),
                "negativeContinuingRewardLowerBound": -8. * self.decision_steps / (1.-.99) * self.reward_scale,
                "scope": "sum physical tick rewards; scale terminal incentives for the longer command discount horizon",
            }
            if self.record_trajectory:
                self.last_control_frames.append(self.snapshot())
            if terminated or truncated:
                break
        self._done = bool(terminated or truncated)
        return self._observation(), total_reward, bool(terminated), bool(truncated), self._add_command_info(info, control_steps=control_steps)

    def snapshot(self):
        frame = self.env.snapshot()
        frame["stats"] = self._add_command_info(frame["stats"])
        return frame
