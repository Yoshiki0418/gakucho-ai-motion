"""Learn joint residuals for the short-command gait without moving the root.

This isolated motor environment is not a room-navigation policy. A disclosed
training-only bearing driver supplies short commands; the actor sees that
command and real physical/scene observations. The task's final destination
remains in the real navigation environment for terminal gates and contact auditing.
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
from .humanoid import (CONTROL_SUBSTEPS, JOINT_LIMITS, JOINT_NAMES, PD_KD, PD_KP,
                       RESIDUAL_LIMITS, TORQUE_LIMITS)


COMMAND_MOTOR_OBSERVATION_CONTRACT = {
    "version": 1, "observationCount": 110, "actionCount": 20,
    "baseObservationCount": 59, "sceneSensorCount": 51,
    "goalFeatureIndices": [50, 51, 52],
    "goalFeatures": "next torque-reference short command ground point in current pelvis frame; not task final destination",
    "otherFeatures": "unchanged actual base joint/body/task/phase observations and actual scene sensor51",
    "actions": "twenty bounded joint residuals over command_gait_pose; no additional hip-yaw clamp",
}

COMMAND_MOTOR_REWARD_V2_CONTRACT = {
    "version": 2, "scale": .01, "gammaPerPhysicalControl": .99,
    "denseScope": "consumed short command and measured physical transition only; no actual final-goal dense terms",
    "desiredSpeedMetersPerSecond": ".35 * clip(consumed command distance/.5, 0, 1) * smoothstep(time/.8)",
    "commandProgress": "root displacement projected onto the pre-step consumed command direction / .02, clipped [-1,1]",
    "heading": "2 * walkingWeight * (cos(actual post-step heading - consumed command bearing)-1)",
    "physicalTerms": "upright/height, cached T2M pose error, actual sole clearance/support, actual swing events, loaded contact slip, mean actual substep torque cost and action costs",
    "denseRawLowerBound": -8., "successRawBonus": 1500., "failureRawCostOutsideDenseClamp": 1500.,
    "negativeContinuingValueLowerBoundScaled": -8., "failureTerminalCostScaled": 15.,
    "terminalScope": "actual final-goal stopped support/gait-qualified success, real collision or fall; inherited navigation gates unchanged",
}

COMMAND_MOTOR_REWARD_CONTRACT = copy.deepcopy(COMMAND_MOTOR_REWARD_V2_CONTRACT)
COMMAND_MOTOR_REWARD_CONTRACT.update({
    "version": 3,
    "commandProgress": "3 * clip(root displacement projected onto pre-step consumed command direction / .02, -1, desired command speed)",
    "heading": "replaced by observed actual pre/post world-heading rate tracking; no requested teacher heading used",
    "desiredYawRateRadiansPerSecond": "clip(.3 * consumed command-relative heading, -.135, .135) * clip(command distance/.15, 0, 1)",
    "actualYawRateRadiansPerSecond": "wrapped actual post-heading minus pre-heading / .02",
    "yawRateReward": "4 * (exp(-((actual yaw rate - desired yaw rate)/.5)^2)-1); bounded [-4,0]",
    "yawRateTargetObservable": "short command features50:53 and actual local up41:44; no world goal or unslewed request",
})


def observed_command_angle(observation):
    """Recover the consumed world-horizontal bearing offset from actor input.

    Heading uses the horizontal pelvis X axis, so its forward basis is the
    cross product of that projected axis and world up, rather than simply
    projecting the pitched/rolled local -Y axis.
    """
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (110,) or not np.isfinite(observation).all():
        raise ValueError("observed command angle requires the finite 110-input motor observation")
    up = observation[41:44]
    norm = float(np.linalg.norm(up))
    if norm < 1e-8:
        raise ValueError("observed up vector must have a nonzero norm")
    up = up/norm
    goal = observation[50:53]
    horizontal = goal-up*float(np.dot(goal, up))
    distance = float(np.linalg.norm(horizontal))
    if distance <= 1e-6:
        return 0., distance
    right = np.asarray([1., 0., 0.])-up[0]*up
    forward = np.cross([1., 0., 0.], up)
    if float(np.linalg.norm(right)) <= 1e-8:
        raise ValueError("pelvis X axis has no horizontal heading")
    return math.atan2(float(np.dot(horizontal, right)), float(np.dot(horizontal, forward))), distance


def command_tracking_reward(measurements):
    """Compute dense reward from physical measurements without task-goal access.

    The command is the one actually consumed before integration. Regenerating
    the rolling command first would move the target with the root and obscure
    progress. Terminal task credit is deliberately handled separately.
    """
    m = measurements
    distance = float(m["commandDistanceMeters"])
    direction = np.asarray(m["commandDirectionWorldXY"], dtype=np.float64)
    ramp = float(m["startRamp"])
    weight = ramp*float(np.clip(distance/.15, 0., 1.))
    speed = .35*float(np.clip(distance/.5, 0., 1.))*ramp
    desired_velocity = speed*direction
    velocity_error = np.asarray(m["rootVelocityWorldXY"])-desired_velocity
    progress = float(np.clip(np.dot(m["rootDisplacementWorldXY"], direction)/m["controlDtSeconds"], -1., speed))
    heading_error = math.atan2(math.sin(m["rootHeadingRadians"]-m["commandBearingRadians"]),
                              math.cos(m["rootHeadingRadians"]-m["commandBearingRadians"]))
    consumed_offset = math.atan2(math.sin(m["commandBearingRadians"]-m["rootPreHeadingRadians"]),
                                math.cos(m["commandBearingRadians"]-m["rootPreHeadingRadians"]))
    desired_yaw_rate = float(np.clip(.3*consumed_offset, -.135, .135))*float(np.clip(distance/.15, 0., 1.))
    actual_yaw_rate = math.atan2(math.sin(m["rootHeadingRadians"]-m["rootPreHeadingRadians"]),
                               math.cos(m["rootHeadingRadians"]-m["rootPreHeadingRadians"]))/m["controlDtSeconds"]
    height_target = m["standingRootHeightMeters"]-.03*m["bodyScale"]*weight
    height = math.exp(-((m["rootHeightMeters"]-height_target)/(.15*m["bodyScale"]))**2)
    clearance = 0.
    for sole, phase in zip(m["footClearancesMeters"],
                           (m["referencePhase"], (m["referencePhase"]+.5)%1.)):
        desired = .055*max(0., math.sin(2.*math.pi*phase))*weight
        clearance += math.exp(-((sole-desired)/.025)**2)
    components = {
        "upright": .5*max(float(m["rootUpright"]), 0.), "height": .5*height,
        "referencePose": .4*math.exp(-8.*m["referencePoseError"]),
        "commandVelocity": 4.*math.exp(-6.*float(np.dot(velocity_error, velocity_error))),
        "commandProgress": 3.*progress,
        "commandYawRate": 4.*(math.exp(-((actual_yaw_rate-desired_yaw_rate)/.5)**2)-1.),
        "actualSoleClearance": 2.*clearance,
        "actualSupportSwing": 8.*weight*m["supportSwingScore"],
        "actualSwingEvents": 12.*m["actualSwingEventCount"],
        "loadedFootSlip": -4.*weight*m["loadedFootSlipMetersPerSecond"],
        "actualTorqueCost": -.015*m["meanSquaredTorqueFraction"],
        "actionCost": -.03*m["meanSquaredAction"],
        "actionChangeCost": -.02*m["meanSquaredActionChange"],
    }
    unclamped = float(sum(components.values()))
    if not math.isfinite(unclamped):
        raise RuntimeError("command tracking reward received non-finite physical measurements")
    raw = max(-8., unclamped)
    return {"denseRawReward": raw, "denseRawRewardBeforeClamp": unclamped,
            "denseClampAdjustment": raw-unclamped, "components": components,
            "desiredVelocityWorldXYMetersPerSecond": desired_velocity.tolist(),
            "desiredSpeedMetersPerSecond": speed, "commandProgressMetersPerSecond": progress,
            "headingErrorRadians": heading_error, "walkingWeight": weight,
            "consumedCommandRelativeHeadingRadians": consumed_offset,
            "desiredYawRateRadiansPerSecond": desired_yaw_rate,
            "actualYawRateRadiansPerSecond": actual_yaw_rate,
            "measurements": copy.deepcopy(measurements)}


def command_motor_control_contract(env):
    body = env.unwrapped
    return {"version": 1, "role": "command-conditioned physical joint residual motor",
        "observationContract": copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT),
        "avatarProfileSha256": body.avatar_profile_sha256,
        "bodyModelSha256": body.body_model_sha256,
        "reference": "command_gait_pose: T2M cadence, independent swing lift and pace-scaled stride",
        "commandGaitSourceSha256": hashlib.sha256((Path(__file__).parent/"command_gait.py").read_bytes()).hexdigest(),
        "jointNames": list(JOINT_NAMES), "jointLimitsRadians": JOINT_LIMITS.tolist(),
        "residualLimitsRadians": RESIDUAL_LIMITS.tolist(),
        "torqueLimitsNm": TORQUE_LIMITS.tolist(), "pdKp": PD_KP.tolist(), "pdKd": PD_KD.tolist(),
        "physicalControlDtSeconds": float(body.dt), "physicsSubstepsPerControl": CONTROL_SUBSTEPS,
        "commandHeadingBoundRadians": env.max_heading_radians,
        "allowedCommandHeadingSlewRadiansPerSecond": .175,
        "commandLookaheadAllowedMeters": [.35, .5], "goalDistanceCapped": True,
        "rewardScale": env.reward_scale, "gammaPerPhysicalControl": .99,
        "rewardContract": copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT),
        "rootStateControl": "MuJoCo gravity, contacts and bounded joint torques only",
        "legacyNavigationContractCompatible": False}


def assert_command_motor_policy(policy, env):
    if policy is None:
        return
    if (not isinstance(policy.observation_space, spaces.Box)
            or policy.observation_space.shape != (110,)
            or not isinstance(policy.action_space, spaces.Box)
            or policy.action_space.shape != (20,)
            or not np.array_equal(policy.action_space.low, env.action_space.low)
            or not np.array_equal(policy.action_space.high, env.action_space.high)
            or not np.array_equal(policy.observation_space.low, env.observation_space.low)
            or not np.array_equal(policy.observation_space.high, env.observation_space.high)):
        raise ValueError("command motor requires the exact 110 observation / 20 joint-action spaces")
    if getattr(policy, "command_motor_observation_contract", None) != COMMAND_MOTOR_OBSERVATION_CONTRACT:
        raise ValueError("command motor observation semantics differ from this environment")
    if getattr(policy, "command_motor_control_contract", None) != env.control_contract:
        raise ValueError("command motor reference, body, reward or control contract differs")
    if getattr(policy, "command_motor_reward_contract", None) != COMMAND_MOTOR_REWARD_CONTRACT:
        raise ValueError("command motor reward contract differs; explicit reward transfer is required")
    if hasattr(policy, "navigation_observation_contract"):
        raise ValueError("command motor must not retain the incompatible legacy navigation contract")
    if getattr(policy, "gamma", None) != .99:
        raise ValueError("command motor requires gamma .99 per actual 20ms physical control")
    assert_policy_body(policy, env.unwrapped)


class CommandMotorEnv(gym.Wrapper):
    """An isolated 110-input/20-action motor with a training bearing driver."""

    def __init__(self, *, command_lookahead_range=(.5, .5), max_heading_radians=.45,
                 heading_slew_radians_per_second=.175, reward_scale=.01,
                 record_trajectory=False, **kwargs):
        if "env_factory" in kwargs:
            raise ValueError("the command motor physics factory is fixed")
        if not (math.isfinite(max_heading_radians) and 0 < max_heading_radians <= .45):
            raise ValueError("motor command heading bound must be in (0, .45]")
        if not (math.isfinite(heading_slew_radians_per_second)
                and 0 < heading_slew_radians_per_second <= .175):
            raise ValueError("motor command slew must be in (0, .175]")
        if reward_scale != .01:
            raise ValueError("isolated command motor reward scale is fixed at .01")
        limits = tuple(float(value) for value in command_lookahead_range)
        if len(limits) != 2 or not all(math.isfinite(value) for value in limits) or not .35 <= limits[0] <= limits[1] <= .5:
            raise ValueError("motor command lookahead range must lie within [.35, .5]")
        # Lazy import prevents the new isolated trainer from introducing a
        # cycle into the established navigation/training modules.
        from .command_navigation import _CommandTaskNavigationEnv, CommandSteeringHumanoidEnv
        navigation = _CommandTaskNavigationEnv(env_factory=CommandSteeringHumanoidEnv, **kwargs)
        super().__init__(navigation)
        self.navigation_env = navigation
        self.command_lookahead_range = limits
        self.max_heading_radians = float(max_heading_radians)
        self.heading_slew_radians_per_second = float(heading_slew_radians_per_second)
        self.reward_scale = float(reward_scale)
        self.record_trajectory = bool(record_trajectory)
        self.observation_space = navigation.observation_space
        self.action_space = navigation.action_space
        self.control_contract = command_motor_control_contract(self)
        self._physical_observation = None
        self._effective_heading_offset = 0.
        self._lookahead = limits[1]
        self._next_command_goal = None
        self._command = None
        self._done = False
        self._step_observed_substeps = 0
        self._episode_observed_substeps = 0
        self._step_torque_squared_sum = 0.
        self._last_reward_info = None
        self.last_control_frames = []
        self.training_command_generator_contract = {
            "role": "training/evaluation-only analytic final-goal bearing driver; not learned room navigation",
            "headingOffsetBoundRadians": self.max_heading_radians,
            "headingSlewRadiansPerSecond": self.heading_slew_radians_per_second,
            "sampledLookaheadRangeMeters": list(self.command_lookahead_range),
            "commandRegeneratedAfterEachActual20msControl": True,
            "actualTaskGoalFixed": True, "geometryPathsOrWaypointsUsed": False}

    @property
    def scene(self):
        return self.navigation_env.scene

    @property
    def next_command_goal(self):
        return None if self._next_command_goal is None else self._next_command_goal.copy()

    @property
    def physical_observation(self):
        return None if self._physical_observation is None else self._physical_observation.copy()

    def _update_next_command(self, *, advance):
        body = self.unwrapped
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        current_heading = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = body.goal[:2] - body.data.qpos[:2]
        distance = float(np.linalg.norm(delta))
        goal_bearing = math.atan2(delta[0], -delta[1])
        error = math.atan2(math.sin(goal_bearing-current_heading), math.cos(goal_bearing-current_heading))
        requested = float(np.clip(error, -self.max_heading_radians, self.max_heading_radians))
        if advance:
            change = self.heading_slew_radians_per_second * body.dt
            self._effective_heading_offset += float(np.clip(
                requested-self._effective_heading_offset, -change, change))
        command_heading = current_heading + self._effective_heading_offset
        command_distance = min(self._lookahead, distance)
        self._next_command_goal = np.asarray([
            body.data.qpos[0]+command_distance*math.sin(command_heading),
            body.data.qpos[1]-command_distance*math.cos(command_heading), body.goal[2]], dtype=np.float64)
        body.command_goal = self._next_command_goal.copy()
        self._command = {"source": "training-only analytic actual final-goal bearing",
            "requestedHeadingOffsetRadians": requested,
            "effectiveHeadingOffsetRadians": self._effective_heading_offset,
            "headingRadians": command_heading, "requestedLookaheadMeters": self._lookahead,
            "effectiveLookaheadMeters": command_distance,
            "worldCommandGoal": self._next_command_goal.tolist(),
            "actualFinalGoal": body.goal.tolist(), "geometryPathTeacherUsed": False}

    def _observation(self):
        body = self.unwrapped
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        observation = self._physical_observation.copy()
        observation[50:53] = rotation.T @ (self._next_command_goal-body.data.qpos[:3])
        return observation

    def _add_info(self, info):
        info = dict(info)
        info["command_motor"] = {"observationContract": copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT),
            "controlContract": copy.deepcopy(self.control_contract), "nextCommand": copy.deepcopy(self._command),
            "trainingCommandGeneratorContract": copy.deepcopy(self.training_command_generator_contract),
            "teacherDriverUsed": True, "isLearnedRoomNavigator": False,
            "actualFinalGoalFixed": True, "rootStateOverwrittenDuringStep": False,
            "physicalControlsExecuted": 1 if self._physical_observation is not None and self.unwrapped._step_count else 0,
            "observedPhysicsSubstepsThisControl": self._step_observed_substeps,
            "observedPhysicsSubstepsThisEpisode": self._episode_observed_substeps,
            "rewardScale": self.reward_scale, "rewardContract": copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT)}
        if self._last_reward_info is not None:
            info["command_motor"].update(copy.deepcopy(self._last_reward_info))
        return info

    def _install_physics_invariant_observer(self):
        body = self.unwrapped
        actual_goal, goal_copy = body.goal, body.goal.copy()
        contact_observer = body._physics_contact_observer
        if not callable(contact_observer):
            raise RuntimeError("command motor requires the original 2ms contact observer")

        def observe(model, data):
            if body.goal is not actual_goal or not np.array_equal(body.goal, goal_copy):
                raise RuntimeError("command motor changed the actual goal during a 2ms physics substep")
            if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                raise RuntimeError("command motor applied an external force during a 2ms physics substep")
            if not np.array_equal(body.command_goal, self._next_command_goal):
                raise RuntimeError("motor command changed inside a physics control step")
            contact_observer(model, data)
            self._step_torque_squared_sum += float(np.mean(np.square(data.ctrl/TORQUE_LIMITS)))
            self._step_observed_substeps += 1
            self._episode_observed_substeps += 1

        body._physics_contact_observer = observe

    def reset(self, *, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=options)
        self._physical_observation = raw
        self._effective_heading_offset = 0.
        self._lookahead = float(self.unwrapped.np_random.uniform(*self.command_lookahead_range))
        self._done = False
        self._step_observed_substeps = 0
        self._episode_observed_substeps = 0
        self._step_torque_squared_sum = 0.
        self._last_reward_info = None
        self.last_control_frames = []
        self._update_next_command(advance=False)
        self._install_physics_invariant_observer()
        if command_motor_control_contract(self) != self.control_contract:
            raise RuntimeError("reset changed the command motor physical body or reference contract")
        return self._observation(), self._add_info(info)

    def step(self, action):
        if self._done or self._physical_observation is None:
            raise RuntimeError("reset() is required before a command motor step")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (20,) or not np.isfinite(action).all():
            raise ValueError("command motor requires twenty finite joint residuals")
        body = self.unwrapped
        actual_goal, goal_copy = body.goal, body.goal.copy()
        if not np.array_equal(body.command_goal, self._next_command_goal):
            raise RuntimeError("the exposed motor observation and reference command diverged")
        self._step_observed_substeps = 0
        self._step_torque_squared_sum = 0.
        consumed_command = self._next_command_goal.copy()
        previous_position = body.data.qpos[:2].copy()
        previous_action = body._last_action.copy()
        previous_swings = body._swing_counts.copy()
        command_delta = consumed_command[:2]-previous_position
        command_distance = float(np.linalg.norm(command_delta))
        command_direction = command_delta/max(command_distance, 1e-8)
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        pre_heading = math.atan2(rotation[1, 0], rotation[0, 0])
        command_bearing = (math.atan2(command_delta[0], -command_delta[1])
                           if command_distance > 1e-8 else pre_heading)
        ramp = float(np.clip(body.data.time/.8, 0., 1.))
        ramp = ramp*ramp*(3.-2.*ramp)
        raw, reward, terminated, truncated, info = self.env.step(np.clip(action, -1., 1.))
        if self._step_observed_substeps != CONTROL_SUBSTEPS:
            raise RuntimeError("motor did not observe every actual 2ms physics substep")
        if body.goal is not actual_goal or not np.array_equal(body.goal, goal_copy):
            raise RuntimeError("command motor changed the actual final destination")
        if np.any(body.data.qfrc_applied) or np.any(body.data.xfrc_applied):
            raise RuntimeError("command motor unexpectedly applied an external force")
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        measurements = {
            "commandDistanceMeters": command_distance, "commandDirectionWorldXY": command_direction.tolist(),
            "commandBearingRadians": command_bearing, "consumedCommandWorldXYZ": consumed_command.tolist(),
            "rootDisplacementWorldXY": (body.data.qpos[:2]-previous_position).tolist(),
            "rootVelocityWorldXY": body.data.qvel[:2].tolist(),
            "rootHeadingRadians": math.atan2(rotation[1, 0], rotation[0, 0]),
            "rootPreHeadingRadians": pre_heading,
            "controlDtSeconds": float(body.dt), "startRamp": ramp,
            "rootUpright": float(info["upright"]), "rootHeightMeters": float(info["root_height"]),
            "standingRootHeightMeters": float(body.standing_root_height), "bodyScale": float(body._body_scale),
            "referencePoseError": float(info["reference_pose_error"]), "referencePhase": float(info["reference_phase"]),
            "footClearancesMeters": list(info["foot_clearances"]),
            "footForcesNewtons": [float(info[f"{side}_foot_force"]) for side in ("left", "right")],
            "actualSwingEventCount": int(np.maximum(body._swing_counts-previous_swings, 0).sum()),
            "supportSwingScore": float(info["support_swing_reward"]),
            "loadedFootSlipMetersPerSecond": float(info["loaded_foot_slip"]),
            "meanSquaredTorqueFraction": self._step_torque_squared_sum/CONTROL_SUBSTEPS,
            "meanSquaredAction": float(np.mean(np.square(np.clip(action, -1., 1.)))),
            "meanSquaredActionChange": float(np.mean(np.square(np.clip(action, -1., 1.)-previous_action))),
        }
        reward_info = command_tracking_reward(measurements)
        navigation_reward = info["navigation"]["reward"]
        terminal_cost = float(navigation_reward["terminal_cost"])
        success_bonus = float(navigation_reward["success_bonus"])
        physical_reward = reward_info["denseRawReward"]+success_bonus-terminal_cost
        self._last_reward_info = {"trackingReward": reward_info,
            "inheritedNavigationReward": float(reward), "physicalReward": physical_reward,
            "scaledMotorReward": physical_reward*self.reward_scale,
            "terminalCostRawOutsideDenseClamp": terminal_cost, "successBonusRaw": success_bonus,
            "terminalReason": info["navigation"]["terminal_reason"],
            "actualFinalGoalUsedOnlyForTerminalGatesInMotorReward": True}
        self._physical_observation = raw
        # Update in the new pose before exposing next obs; exactly this point
        # will be read by command_gait_pose during the following torque step.
        self._update_next_command(advance=True)
        self._done = bool(terminated or truncated)
        self.last_control_frames = [self.snapshot()] if self.record_trajectory else []
        info = self._add_info(info)
        return self._observation(), physical_reward*self.reward_scale, bool(terminated), bool(truncated), info

    def snapshot(self):
        frame = self.env.snapshot()
        frame["stats"] = self._add_info(frame["stats"])
        return frame
