"""Audit command semantics and actual torque-driven motor adaptation.

These checks deliberately compare two real MuJoCo integrations. They do not
infer a lack of root correction from a bookkeeping flag in a trace.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .body_contract import bind_policy_body
from .command_motor import (COMMAND_MOTOR_OBSERVATION_CONTRACT,
                            COMMAND_MOTOR_REWARD_CONTRACT, COMMAND_MOTOR_REWARD_V2_CONTRACT,
                            CommandMotorEnv, assert_command_motor_policy,
                            command_tracking_reward, observed_command_angle)
from .command_navigation import (CommandSteeringHumanoidEnv,
                                 _CommandTaskNavigationEnv)
from .navigation_training import (NAVIGATION_OBSERVATION_CONTRACT,
                                  assert_navigation_policy)


_PROFILE = Path(__file__).resolve().parents[1] / "frontend/public/world-model/gakucho-avatar-profile.json"
_PHYSICAL_ARGUMENTS = {
    "mode": "turn", "split": "train", "base_seed": 88501,
    "difficulty_range": (.25, .25), "randomize": False,
    "max_episode_seconds": 1., "avatar_profile": _PROFILE,
}
_RESET_OPTIONS = {"scene_seed": 88501, "physics_seed": 89501}
_NON_COMMAND = np.asarray([index for index in range(110) if not 50 <= index < 53])


@pytest.fixture
def physical_motor():
    env = CommandMotorEnv(record_trajectory=True, **_PHYSICAL_ARGUMENTS)
    try:
        yield env
    finally:
        env.close()


def _assert_command_observation(env, observation):
    body = env.unwrapped
    command = env.next_command_goal
    rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
    assert observation.shape == (110,)
    np.testing.assert_array_equal(observation[_NON_COMMAND], env.physical_observation[_NON_COMMAND])
    np.testing.assert_array_equal(observation[50:53],
                                  (rotation.T @ (command-body.data.qpos[:3])).astype(np.float32))
    np.testing.assert_array_equal(body.command_goal, command)
    # The underlying physical observations retain the actual final goal.
    np.testing.assert_array_equal(env.physical_observation[50:53],
                                  (rotation.T @ (body.goal-body.data.qpos[:3])).astype(np.float32))
    assert not np.array_equal(observation[50:53], env.physical_observation[50:53])


def test_actual_200ms_integration_keeps_goal_and_observed_command_coherent(physical_motor, monkeypatch):
    from . import command_navigation as commands

    env = physical_motor
    observation, reset_info = env.reset(seed=88501, options=_RESET_OPTIONS)
    _assert_command_observation(env, observation)
    body = env.unwrapped
    final_goal_object, final_goal_value = body.goal, body.goal.copy()
    observed_substeps = []
    consumed_reference_commands = []
    original_observer = body._physics_contact_observer
    original_reference = commands.command_gait_pose
    expected_command = env.next_command_goal

    def observer(model, data):
        assert body.goal is final_goal_object
        np.testing.assert_array_equal(body.goal, final_goal_value)
        np.testing.assert_array_equal(body.command_goal, expected_command)
        assert not np.any(data.qfrc_applied)
        assert not np.any(data.xfrc_applied)
        observed_substeps.append(float(data.time))
        original_observer(model, data)

    def reference(env_body, command_goal):
        np.testing.assert_array_equal(command_goal, expected_command)
        consumed_reference_commands.append(np.asarray(command_goal).copy())
        return original_reference(env_body, command_goal)

    body._physics_contact_observer = observer
    monkeypatch.setattr(commands, "command_gait_pose", reference)
    assert reset_info["command_motor"]["physicalControlsExecuted"] == 0
    for index in range(10):
        expected_command = env.next_command_goal
        consumed_angle, consumed_distance = observed_command_angle(observation)
        previous_offset = env._effective_heading_offset
        observation, reward, done, truncated, info = env.step(np.zeros(20))
        assert not done and not truncated
        _assert_command_observation(env, observation)
        assert abs(env._effective_heading_offset-previous_offset) <= .175*.02 + 1e-12
        assert reward == pytest.approx(info["command_motor"]["physicalReward"]*.01)
        assert info["command_motor"]["physicalControlsExecuted"] == 1
        assert info["command_motor"]["observedPhysicsSubstepsThisControl"] == 10
        assert info["command_motor"]["observedPhysicsSubstepsThisEpisode"] == (index+1)*10
        assert info["command_motor"]["isLearnedRoomNavigator"] is False
        tracking = info["command_motor"]["trackingReward"]
        assert tracking["consumedCommandRelativeHeadingRadians"] == pytest.approx(consumed_angle, abs=2e-6)
        assert tracking["measurements"]["commandDistanceMeters"] == pytest.approx(consumed_distance, abs=2e-6)
        assert len(env.last_control_frames) == 1
        assert env.last_control_frames[0]["time"] == pytest.approx(body.data.time)
    assert len(consumed_reference_commands) == 10
    assert len(observed_substeps) == 100
    np.testing.assert_allclose(observed_substeps, np.arange(1, 101)*.002, rtol=0, atol=2e-15)
    assert body.data.time == pytest.approx(.2)


def test_motor_wrapper_matches_unmodified_physics_with_the_same_commands(physical_motor):
    env = physical_motor
    shadow = _CommandTaskNavigationEnv(env_factory=CommandSteeringHumanoidEnv, **_PHYSICAL_ARGUMENTS)
    try:
        env.reset(seed=88501, options=_RESET_OPTIONS)
        shadow.reset(seed=88501, options=_RESET_OPTIONS)
        for _ in range(10):
            shadow.unwrapped.command_goal = env.next_command_goal
            obs, reward, done, truncated, motor_info = env.step(np.zeros(20))
            raw_obs, raw_reward, raw_done, raw_truncated, _ = shadow.step(np.zeros(20))
            assert (done, truncated) == (raw_done, raw_truncated)
            assert motor_info["command_motor"]["inheritedNavigationReward"] == raw_reward
            assert reward == motor_info["command_motor"]["physicalReward"]*.01
            np.testing.assert_array_equal(obs[_NON_COMMAND], raw_obs[_NON_COMMAND])
            for name in ("qpos", "qvel", "qacc", "ctrl", "xfrc_applied", "qfrc_applied",
                         "xpos", "geom_xpos", "actuator_force"):
                np.testing.assert_array_equal(getattr(env.unwrapped.data, name),
                                              getattr(shadow.unwrapped.data, name), err_msg=name)
            assert env.unwrapped.data.ncon == shadow.unwrapped.data.ncon
            assert env.navigation_env._step_observer_calls == shadow._step_observer_calls == 10
    finally:
        shadow.close()


def test_dense_motor_reward_tracks_consumed_command_independently_of_global_goal(physical_motor, monkeypatch):
    from . import navigation

    first = physical_motor
    first.reset(seed=88501, options=_RESET_OPTIONS)
    generator = navigation.generate_navigation_scene

    def different_final_goal(*args, **kwargs):
        scene = generator(*args, **kwargs)
        goal = np.asarray(scene["goals"][0]["position"], dtype=np.float64)
        scene["goals"][0]["position"] = (goal+[1., 1., 0.]).tolist()
        return scene

    with monkeypatch.context() as construction_patch:
        construction_patch.setattr(navigation, "generate_navigation_scene", different_final_goal)
        second = CommandMotorEnv(record_trajectory=True, **_PHYSICAL_ARGUMENTS)
        second.reset(seed=88501, options=_RESET_OPTIONS)
    try:
        assert not np.array_equal(first.unwrapped.goal, second.unwrapped.goal)
        fixed_command = first.next_command_goal
        final_goals = [first.unwrapped.goal.copy(), second.unwrapped.goal.copy()]
        observer_times = [[], []]
        for index, env in enumerate((first, second)):
            # Supply the same physical short command to both motors. The two
            # actual final goals already differ before their reset observers
            # capture them; neither goal changes during integration.
            def fixed_driver(*, advance, env=env):
                env._next_command_goal = fixed_command.copy()
                env.unwrapped.command_goal = fixed_command.copy()
                env._command = {"source": "fixed command for goal-independence test",
                                "worldCommandGoal": fixed_command.tolist()}

            monkeypatch.setattr(env, "_update_next_command", fixed_driver)
            env._update_next_command(advance=False)
            original = env.unwrapped._physics_contact_observer

            def observer(model, data, index=index, env=env, original=original):
                np.testing.assert_array_equal(env.unwrapped.goal, final_goals[index])
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                observer_times[index].append(float(data.time))
                original(model, data)

            env.unwrapped._physics_contact_observer = observer
        inherited_rewards_differ = False
        for _ in range(5):
            results = [env.step(np.zeros(20)) for env in (first, second)]
            left, right = [result[4]["command_motor"] for result in results]
            assert left["trackingReward"] == right["trackingReward"]
            assert results[0][1] == results[1][1]
            assert left["terminalCostRawOutsideDenseClamp"] == right["terminalCostRawOutsideDenseClamp"] == 0.
            assert left["successBonusRaw"] == right["successBonusRaw"] == 0.
            inherited_rewards_differ |= left["inheritedNavigationReward"] != right["inheritedNavigationReward"]
            measurements = left["trackingReward"]["measurements"]
            assert not any("goal" in key.lower() for key in measurements)
            assert command_tracking_reward(measurements) == left["trackingReward"]
            np.testing.assert_array_equal(measurements["consumedCommandWorldXYZ"], fixed_command)
            for name in ("qpos", "qvel", "qacc", "ctrl", "actuator_force", "geom_xpos"):
                np.testing.assert_array_equal(getattr(first.unwrapped.data, name),
                                              getattr(second.unwrapped.data, name), err_msg=name)
            for env, expected_goal in zip((first, second), final_goals):
                np.testing.assert_array_equal(env.unwrapped.goal, expected_goal)
                frame = env.last_control_frames[0]
                assert frame["time"] == pytest.approx(env.unwrapped.data.time)
                assert frame["stats"]["command_motor"]["rewardContract"] == COMMAND_MOTOR_REWARD_CONTRACT
        assert inherited_rewards_differ
        for times in observer_times:
            np.testing.assert_allclose(times, np.arange(1, 51)*.002, rtol=0, atol=1e-15)
    finally:
        second.close()


def test_reward_snapshots_do_not_advance_physics_or_change_tracking_and_contact_audits(physical_motor):
    env = physical_motor
    env.reset(seed=88501, options=_RESET_OPTIONS)
    for _ in range(5):
        env.step(np.zeros(20))
    body = env.unwrapped
    physical_before = {name: getattr(body.data, name).copy() for name in (
        "qpos", "qvel", "qacc", "ctrl", "xfrc_applied", "qfrc_applied", "xpos", "geom_xpos", "actuator_force")}
    audit_before = (env._step_observed_substeps, env._episode_observed_substeps,
                    env.navigation_env._step_observer_calls, env.navigation_env._collision_substeps,
                    env.navigation_env._qualified_success_seconds, body._step_count, float(body.data.time))
    reward_before = copy.deepcopy(env._last_reward_info)
    command_before, goal_before = env.next_command_goal, body.goal.copy()
    first, second = env.snapshot(), env.snapshot()
    assert first == second
    assert first["stats"]["command_motor"]["trackingReward"] == reward_before["trackingReward"]
    for name, expected in physical_before.items():
        np.testing.assert_array_equal(getattr(body.data, name), expected, err_msg=name)
    assert audit_before == (env._step_observed_substeps, env._episode_observed_substeps,
        env.navigation_env._step_observer_calls, env.navigation_env._collision_substeps,
        env.navigation_env._qualified_success_seconds, body._step_count, float(body.data.time))
    assert env._last_reward_info == reward_before
    np.testing.assert_array_equal(env.next_command_goal, command_before)
    np.testing.assert_array_equal(body.goal, goal_before)
    assert body.data.time == pytest.approx(.1)


def _tracking_measurements():
    return {"commandDistanceMeters": .5, "commandDirectionWorldXY": [0., -1.],
        "commandBearingRadians": 0., "consumedCommandWorldXYZ": [0., -.5, 0.],
        "rootDisplacementWorldXY": [0., -.007], "rootVelocityWorldXY": [0., -.35],
        "rootPreHeadingRadians": 0., "rootHeadingRadians": 0., "controlDtSeconds": .02, "startRamp": 1.,
        "rootUpright": 1., "rootHeightMeters": .97, "standingRootHeightMeters": 1.,
        "bodyScale": 1., "referencePoseError": 0., "referencePhase": 0.,
        "footClearancesMeters": [0., 0.], "footForcesNewtons": [300., 300.],
        "actualSwingEventCount": 0, "supportSwingScore": 0., "loadedFootSlipMetersPerSecond": 0.,
        "meanSquaredTorqueFraction": 0., "meanSquaredAction": 0., "meanSquaredActionChange": 0.}


def test_command_reward_prefers_measured_command_velocity_and_direction_without_mutating_inputs():
    measurements = _tracking_measurements()
    preserved = copy.deepcopy(measurements)
    matched = command_tracking_reward(measurements)
    assert measurements == preserved
    assert matched["desiredSpeedMetersPerSecond"] == .35
    assert matched["commandProgressMetersPerSecond"] == pytest.approx(.35)
    assert matched["desiredVelocityWorldXYMetersPerSecond"] == [0., -.35]
    backward = copy.deepcopy(measurements)
    backward.update(rootVelocityWorldXY=[0., .35], rootDisplacementWorldXY=[0., .007])
    opposite = command_tracking_reward(backward)
    lateral = copy.deepcopy(measurements)
    lateral.update(rootVelocityWorldXY=[.35, 0.], rootDisplacementWorldXY=[.007, 0.])
    sidestep = command_tracking_reward(lateral)
    assert matched["denseRawReward"] > sidestep["denseRawReward"] > opposite["denseRawReward"]
    turned = copy.deepcopy(measurements)
    turned["rootHeadingRadians"] = np.pi
    assert command_tracking_reward(turned)["components"]["commandYawRate"] < 0
    assert "commandHeadingAlignment" not in matched["components"]
    stop = copy.deepcopy(measurements)
    stop.update(commandDistanceMeters=0., commandDirectionWorldXY=[0., 0.],
                rootVelocityWorldXY=[0., 0.], rootDisplacementWorldXY=[0., 0.])
    stopping = command_tracking_reward(stop)
    assert stopping["desiredSpeedMetersPerSecond"] == 0
    assert stopping["walkingWeight"] == 0


def test_motor_dense_reward_has_declared_lower_bound_and_rejects_nonfinite_measurements():
    measurements = _tracking_measurements()
    measurements["loadedFootSlipMetersPerSecond"] = 1000.
    reward = command_tracking_reward(measurements)
    assert reward["denseRawReward"] == COMMAND_MOTOR_REWARD_CONTRACT["denseRawLowerBound"]
    assert reward["denseRawRewardBeforeClamp"] < reward["denseRawReward"]
    invalid = _tracking_measurements()
    invalid["rootVelocityWorldXY"] = [np.nan, 0.]
    with pytest.raises(RuntimeError, match="non-finite"):
        command_tracking_reward(invalid)


@pytest.mark.parametrize("direction", [-1., 1.])
def test_yaw_reward_tracks_the_consumed_signed_command_in_every_gait_phase(direction):
    measurements = _tracking_measurements()
    measurements["commandBearingRadians"] = direction*.45
    measurements["rootHeadingRadians"] = direction*.135*.02
    for phase in (0., .12, .27, .5, .75):
        measurements["referencePhase"] = phase
        reward = command_tracking_reward(measurements)
        assert reward["consumedCommandRelativeHeadingRadians"] == pytest.approx(direction*.45)
        assert reward["desiredYawRateRadiansPerSecond"] == pytest.approx(direction*.135)
        assert reward["actualYawRateRadiansPerSecond"] == pytest.approx(direction*.135)
        assert reward["components"]["commandYawRate"] == pytest.approx(0., abs=1e-12)
    stationary = copy.deepcopy(measurements)
    stationary["rootHeadingRadians"] = 0.
    opposite = copy.deepcopy(measurements)
    opposite["rootHeadingRadians"] *= -1.
    assert (reward["components"]["commandYawRate"]
            > command_tracking_reward(stationary)["components"]["commandYawRate"]
            > command_tracking_reward(opposite)["components"]["commandYawRate"])


@pytest.mark.parametrize("direction", [-1., 1.])
def test_actual_yaw_rate_wraps_the_world_heading_boundary(direction):
    import math
    measurements = _tracking_measurements()
    measurements["rootPreHeadingRadians"] = direction*(math.pi-.001)
    measurements["rootHeadingRadians"] = -direction*(math.pi-.001)
    desired_bearing = measurements["rootPreHeadingRadians"]+direction*.45
    measurements["commandBearingRadians"] = math.atan2(math.sin(desired_bearing), math.cos(desired_bearing))
    reward = command_tracking_reward(measurements)
    assert reward["actualYawRateRadiansPerSecond"] == pytest.approx(direction*.1, abs=1e-12)
    assert reward["consumedCommandRelativeHeadingRadians"] == pytest.approx(direction*.45)
    assert reward["desiredYawRateRadiansPerSecond"] == pytest.approx(direction*.135)


def test_yaw_target_is_command_observable_at_start_and_disappears_for_a_stop_command():
    measurements = _tracking_measurements()
    measurements.update(commandBearingRadians=.45, startRamp=0.)
    beginning = command_tracking_reward(measurements)
    assert beginning["desiredYawRateRadiansPerSecond"] == pytest.approx(.135)
    assert beginning["desiredSpeedMetersPerSecond"] == 0.
    assert beginning["commandProgressMetersPerSecond"] == 0.
    measurements["commandDistanceMeters"] = .075
    assert command_tracking_reward(measurements)["desiredYawRateRadiansPerSecond"] == pytest.approx(.0675)
    measurements.update(commandDistanceMeters=0., commandDirectionWorldXY=[0., 0.], rootHeadingRadians=.01)
    stopped = command_tracking_reward(measurements)
    assert stopped["desiredYawRateRadiansPerSecond"] == 0.
    assert stopped["desiredSpeedMetersPerSecond"] == 0.
    assert stopped["components"]["commandYawRate"] < 0.


def test_fast_root_movement_cannot_increase_command_progress_credit():
    measurements = _tracking_measurements()
    matched = command_tracking_reward(measurements)
    measurements.update(rootVelocityWorldXY=[0., -1.], rootDisplacementWorldXY=[0., -.02])
    overspeed = command_tracking_reward(measurements)
    assert overspeed["commandProgressMetersPerSecond"] == matched["desiredSpeedMetersPerSecond"]
    assert overspeed["components"]["commandProgress"] == matched["components"]["commandProgress"]
    assert overspeed["denseRawReward"] < matched["denseRawReward"]
    measurements["rootDisplacementWorldXY"] = [0., .04]
    assert command_tracking_reward(measurements)["commandProgressMetersPerSecond"] == -1.


def test_actor_command_angle_matches_world_geometry_in_native_mujoco_poses(physical_motor):
    import math
    import mujoco
    from .locomotion import rotation_x, rotation_y
    from .steering import rotation_z

    env = physical_motor
    env.reset(seed=88501, options=_RESET_OPTIONS)
    body = env.unwrapped
    original_qpos = body.data.qpos.copy()
    try:
        # Reset-only forward-kinematics probes cover rolled/pitched bodies and
        # the heading discontinuity. No integration or learned motion is
        # claimed for these deliberately configured native MuJoCo poses.
        for yaw, roll, pitch in ((0., 0., 0.), (3.13, .22, -.18), (-3.13, -.27, .31)):
            desired_rotation = rotation_z(yaw) @ rotation_x(roll) @ rotation_y(pitch)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, desired_rotation.reshape(-1))
            body.data.qpos[3:7] = quaternion
            mujoco.mj_forward(body.model, body.data)
            rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
            actual_heading = math.atan2(rotation[1, 0], rotation[0, 0])
            for angle, distance in ((-.45, .35), (.45, .5), (math.pi-.01, .5), (-math.pi+.01, .35), (0., 0.)):
                bearing = actual_heading+angle
                command = np.asarray([body.data.qpos[0]+distance*math.sin(bearing),
                    body.data.qpos[1]-distance*math.cos(bearing), body.goal[2]])
                env._next_command_goal = command
                body.command_goal = command.copy()
                info = body._info()
                env._physical_observation, _ = env.navigation_env.env._append(body._observation(info), info)
                actor_input = env._observation()
                preserved = actor_input.copy()
                inferred_angle, inferred_distance = observed_command_angle(actor_input)
                np.testing.assert_array_equal(actor_input, preserved)
                assert inferred_distance == pytest.approx(distance, abs=2e-6)
                if distance:
                    angular_error = math.atan2(math.sin(inferred_angle-angle), math.cos(inferred_angle-angle))
                    assert abs(angular_error) < 2e-6
                else:
                    assert inferred_angle == 0.
            assert not np.any(body.data.xfrc_applied) and not np.any(body.data.qfrc_applied)
        assert body.data.time == 0.
    finally:
        body.data.qpos[:] = original_qpos
        mujoco.mj_forward(body.model, body.data)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "zero_up", "vertical_pelvis_x"])
def test_unobservable_or_invalid_actor_command_geometry_rejects(invalid):
    observation = np.zeros(110, dtype=np.float32)
    observation[43] = 1.
    observation[51:53] = [-.5, -1.]
    if invalid == "shape":
        observation = observation[:59]
    elif invalid == "nonfinite":
        observation[0] = np.nan
    elif invalid == "zero_up":
        observation[41:44] = 0.
    else:
        observation[41:44] = [1., 0., 0.]
    with pytest.raises(ValueError):
        observed_command_angle(observation)


def test_scene_rebuild_and_public_copies_cannot_leave_a_stale_command(physical_motor):
    env = physical_motor
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(20))
    env.reset(seed=88501, options=_RESET_OPTIONS)
    previous_body = env.unwrapped
    env.step(np.zeros(20))
    observation, _ = env.reset(seed=88502, options={"scene_seed": 88502, "physics_seed": 89502})
    assert env.unwrapped is not previous_body
    assert env.unwrapped.data.time == 0
    assert env._effective_heading_offset == 0
    assert env.last_control_frames == []
    _assert_command_observation(env, observation)
    command_before, physical_before = env.next_command_goal, env.physical_observation
    env.next_command_goal[:] = 123.
    env.physical_observation[:] = 123.
    np.testing.assert_array_equal(env.next_command_goal, command_before)
    np.testing.assert_array_equal(env.physical_observation, physical_before)
    env.unwrapped.command_goal[0] += .01
    with pytest.raises(RuntimeError, match="command diverged"):
        env.step(np.zeros(20))
    assert env.unwrapped.data.time == 0


@pytest.mark.parametrize("action", [np.zeros(2), np.zeros(21), np.full(20, np.nan), np.full(20, np.inf)])
def test_invalid_joint_action_cannot_advance_physics(physical_motor, action):
    physical_motor.reset(seed=88501, options=_RESET_OPTIONS)
    with pytest.raises(ValueError, match="twenty finite"):
        physical_motor.step(action)
    assert physical_motor.unwrapped.data.time == 0


@pytest.mark.parametrize("tamper,reason", [("force", "external force"), ("goal", "actual goal")])
def test_physics_invariant_violation_is_detected_on_the_first_2ms_substep(physical_motor, tamper, reason):
    env = physical_motor
    env.reset(seed=88501, options=_RESET_OPTIONS)
    body = env.unwrapped
    original = body._physics_contact_observer

    def faulty_observer(model, data):
        # Inject only inside this test's callback; an end-of-20ms check would
        # miss a violation that was restored before the next control frame.
        previous_goal = body.goal.copy()
        if tamper == "force":
            data.xfrc_applied[body.pelvis_body_id, 0] = 1.
        else:
            body.goal[0] += .01
        try:
            original(model, data)
        finally:
            data.xfrc_applied[:] = 0.
            body.goal[:] = previous_goal

    body._physics_contact_observer = faulty_observer
    with pytest.raises(RuntimeError, match=reason):
        env.step(np.zeros(20))
    assert body.data.time == pytest.approx(.002)
    assert not np.any(body.data.xfrc_applied)


@pytest.mark.parametrize("arguments,reason", [
    ({"command_lookahead_range": (.2, .5)}, "lookahead"),
    ({"command_lookahead_range": (.5, .35)}, "lookahead"),
    ({"command_lookahead_range": (.5, .51)}, "lookahead"),
    ({"command_lookahead_range": (float("nan"), .5)}, "lookahead"),
    ({"command_lookahead_range": (.5,)}, "lookahead"),
    ({"max_heading_radians": .46}, "heading"),
    ({"max_heading_radians": float("nan")}, "heading"),
    ({"heading_slew_radians_per_second": .176}, "slew"),
    ({"reward_scale": 1.}, "reward scale"),
    ({"env_factory": CommandSteeringHumanoidEnv}, "factory"),
])
def test_unsupported_motor_commands_reject_before_building_physics(arguments, reason):
    with pytest.raises(ValueError, match=reason):
        CommandMotorEnv(**arguments, **_PHYSICAL_ARGUMENTS)


def _metadata_policy(env):
    policy = SimpleNamespace(observation_space=env.observation_space, action_space=env.action_space,
        gamma=.99, command_motor_observation_contract=copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT),
        command_motor_control_contract=copy.deepcopy(env.control_contract),
        command_motor_reward_contract=copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT))
    bind_policy_body(policy, env.unwrapped)
    return policy


def test_same_shape_legacy_policy_cannot_silently_change_goal_semantics(physical_motor):
    legacy = _metadata_policy(physical_motor)
    del legacy.command_motor_observation_contract
    del legacy.command_motor_control_contract
    legacy.navigation_observation_contract = copy.deepcopy(NAVIGATION_OBSERVATION_CONTRACT)
    assert_navigation_policy(legacy, physical_motor.navigation_env)
    with pytest.raises(ValueError, match="observation semantics"):
        assert_command_motor_policy(legacy, physical_motor)
    current = _metadata_policy(physical_motor)
    assert_command_motor_policy(current, physical_motor)
    with pytest.raises(ValueError, match="observation contract"):
        assert_navigation_policy(current, physical_motor.navigation_env)
    current.navigation_observation_contract = copy.deepcopy(NAVIGATION_OBSERVATION_CONTRACT)
    with pytest.raises(ValueError, match="incompatible legacy"):
        assert_command_motor_policy(current, physical_motor)


@pytest.mark.parametrize("change,reason", [
    ("semantics", "observation semantics"), ("control", "control contract"),
    ("gamma", "gamma"), ("body", "physical body contract"), ("reward", "reward contract"),
])
def test_mutated_motor_semantics_control_or_body_reject(physical_motor, change, reason):
    policy = _metadata_policy(physical_motor)
    if change == "semantics":
        policy.command_motor_observation_contract["goalFeatures"] = "actual final goal"
    elif change == "control":
        policy.command_motor_control_contract["rewardScale"] = 1.
    elif change == "gamma":
        policy.gamma = .95
    elif change == "reward":
        policy.command_motor_reward_contract["version"] = 1
    else:
        policy.body_model_sha256 = "other-body"
    with pytest.raises(ValueError, match=reason):
        assert_command_motor_policy(policy, physical_motor)


def test_real_ppo_transfer_preserves_actor_scales_only_value_head_and_discards_adam(physical_motor, tmp_path):
    import hashlib
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .command_motor_training import initialize_command_motor

    torch.set_num_threads(1)
    source = PPO("MlpPolicy", physical_motor.navigation_env, seed=42, device="cpu",
        n_steps=8, batch_size=8, n_epochs=1, gamma=.99,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    bind_policy_body(source, physical_motor.unwrapped)
    source.navigation_observation_contract = copy.deepcopy(NAVIGATION_OBSERVATION_CONTRACT)
    # Populate real Adam moments without collecting or claiming training data.
    # The large synthetic counter tests preservation of source bookkeeping.
    source.policy.optimizer.zero_grad()
    loss = source.policy.value_net.weight.square().sum()+source.policy.action_net.weight.square().sum()+source.policy.log_std.sum()
    loss.backward()
    source.policy.optimizer.step()
    assert source.policy.optimizer.state
    source.num_timesteps = 6_381_568
    checkpoint = tmp_path/"legacy-motor.zip"
    source.save(checkpoint)
    before_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    vector = DummyVecEnv([lambda: physical_motor])
    try:
        target, audit = initialize_command_motor(checkpoint, vector, body_env=physical_motor,
            seed=88501, learning_rate=5e-5, n_steps=8, batch_size=8, n_epochs=1)
        assert target.num_timesteps == source.num_timesteps
        assert target.command_motor_physical_control_timesteps == source.num_timesteps
        assert audit["trainingTimestepsAddedByTransfer"] == 0
        assert audit["sourcePhysicalMotorTimesteps"] == source.num_timesteps
        assert audit["sampleCount"] == 32
        assert audit["samplesFromActualTrainResetObservations"] is True
        assert audit["sceneSeedRange"] == [88501, 88532]
        assert audit["physicsSeedRange"] == [89001, 89032]
        assert audit["deterministicActionsMaxAbsErrorBeforeAdaptation"] == 0
        assert audit["criticValueScale"] == .01
        assert not target.policy.optimizer.state
        assert audit["freshAdam"] is True
        assert not hasattr(target, "navigation_observation_contract")
        assert_command_motor_policy(target, physical_motor)
        with pytest.raises(ValueError, match="observation contract"):
            assert_navigation_policy(target, physical_motor.navigation_env)
        for name, original in source.policy.state_dict().items():
            expected = original*.01 if name in {"value_net.weight", "value_net.bias"} else original
            assert torch.equal(target.policy.state_dict()[name], expected), name
        observations = []
        for seed in range(88551, 88583):
            observation, _ = physical_motor.reset(seed=seed, options={
                "scene_seed": seed, "physics_seed": seed+500})
            observations.append(observation)
        observations = np.asarray(observations)
        np.testing.assert_array_equal(source.predict(observations, deterministic=True)[0],
                                      target.predict(observations, deterministic=True)[0])
        with torch.no_grad():
            values = source.policy.predict_values(torch.as_tensor(observations))
            scaled = target.policy.predict_values(torch.as_tensor(observations))
        torch.testing.assert_close(scaled, values*.01, rtol=1e-6, atol=1e-7)
        target.save(tmp_path/"command-motor.zip")
        reloaded = PPO.load(tmp_path/"command-motor.zip", device="cpu")
        assert_command_motor_policy(reloaded, physical_motor)
        assert not reloaded.policy.optimizer.state
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_sha
    finally:
        vector.close()


@pytest.fixture
def reward_v2_policy(physical_motor):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    policy = PPO("MlpPolicy", physical_motor, seed=73, device="cpu",
        n_steps=8, batch_size=8, n_epochs=1, gamma=.99,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    bind_policy_body(policy, physical_motor.unwrapped)
    policy.command_motor_observation_contract = copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT)
    policy.command_motor_control_contract = copy.deepcopy(physical_motor.control_contract)
    policy.command_motor_control_contract["rewardContract"] = copy.deepcopy(COMMAND_MOTOR_REWARD_V2_CONTRACT)
    policy.command_motor_reward_contract = copy.deepcopy(COMMAND_MOTOR_REWARD_V2_CONTRACT)
    # Synthetic counters exercise lineage preservation; no training data or
    # real six-million-step training run is implied by this fixture.
    policy.num_timesteps = 6_586_368
    policy.command_motor_physical_control_timesteps = 6_586_368
    policy.command_motor_reward_v2_initial_physical_controls = 6_483_968
    policy.command_motor_transfer_audit = {"sourcePhysicalMotorTimesteps": 6_381_568,
                                          "trainingTimestepsAddedByTransfer": 0}
    policy.policy.optimizer.zero_grad()
    loss = policy.policy.value_net.weight.square().sum()+policy.policy.action_net.weight.square().sum()+policy.policy.log_std.sum()
    loss.backward()
    policy.policy.optimizer.step()
    assert policy.policy.optimizer.state
    assert torch.count_nonzero(policy.policy.value_net.weight) > 0
    return policy


def test_explicit_reward_v2_to_v3_transfer_keeps_actor_and_lineage_and_zeros_only_value_head(
        physical_motor, reward_v2_policy, tmp_path):
    import hashlib
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .command_motor_training import initialize_command_motor

    source = reward_v2_policy
    checkpoint = tmp_path/"reward-v2.zip"
    source.save(checkpoint)
    source_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="control contract"):
        assert_command_motor_policy(source, physical_motor)
    vector = DummyVecEnv([lambda: physical_motor])
    try:
        target, audit = initialize_command_motor(checkpoint, vector, body_env=physical_motor,
            seed=88701, learning_rate=5e-5, n_steps=8, batch_size=8, n_epochs=1,
            initialize_from_command=True)
        assert target.num_timesteps == target.command_motor_physical_control_timesteps == source.num_timesteps
        assert target.command_motor_reward_v2_initial_physical_controls == source.command_motor_reward_v2_initial_physical_controls
        assert target.command_motor_reward_v3_initial_physical_controls == source.num_timesteps
        assert target.command_motor_reward_contract == COMMAND_MOTOR_REWARD_CONTRACT
        assert not target.policy.optimizer.state
        assert not hasattr(target, "navigation_observation_contract")
        assert audit["sourcePhysicalMotorTimesteps"] == 6_381_568
        assert audit["initialSourceActualPhysicalControlTimesteps"] == 6_586_368
        assert audit["sourceCommandMotorTransferAudit"] == source.command_motor_transfer_audit
        assert audit["sourceRewardContractVersion"] == 2 and audit["targetRewardContractVersion"] == 3
        assert audit["explicitCommandRewardTransfer"] is True
        assert audit["criticValueScale"] == 1.
        assert audit["criticScaledValueMaxAbsError"] == 0
        assert audit["trainingTimestepsAddedByTransfer"] == 0
        assert audit["freshAdam"] is True
        assert audit["sampleCount"] == 32 and audit["samplesFromActualTrainResetObservations"] is True
        assert audit["deterministicActionsMaxAbsErrorBeforeAdaptation"] == 0
        assert_command_motor_policy(target, physical_motor)
        for name, original in source.policy.state_dict().items():
            expected = torch.zeros_like(original) if name in {"value_net.weight", "value_net.bias"} else original
            assert torch.equal(target.policy.state_dict()[name], expected), name
        observations = []
        for seed in range(88751, 88783):
            observation, _ = physical_motor.reset(seed=seed, options={"scene_seed": seed, "physics_seed": seed+500})
            observations.append(observation)
        observations = np.asarray(observations)
        np.testing.assert_array_equal(source.predict(observations, deterministic=True)[0],
                                      target.predict(observations, deterministic=True)[0])
        with torch.no_grad():
            assert torch.count_nonzero(target.policy.predict_values(torch.as_tensor(observations))) == 0
        target.save(tmp_path/"reward-v3.zip")
        reloaded = PPO.load(tmp_path/"reward-v3.zip", device="cpu")
        assert_command_motor_policy(reloaded, physical_motor)
        assert not reloaded.policy.optimizer.state
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_sha
    finally:
        vector.close()


def test_reward_v2_resume_is_rejected_before_any_ppo_learning(physical_motor, reward_v2_policy, tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from .command_motor_training import argument_parser, train

    checkpoint = tmp_path/"reward-v2.zip"
    reward_v2_policy.save(checkpoint)
    args = argument_parser().parse_args(["--resume", str(checkpoint), "--avatar-profile", str(_PROFILE),
        "--output", str(tmp_path/"resume-attempt"), "--envs", "1", "--steps", "1", "--seconds", ".2"])

    def unexpected_learning(*args, **kwargs):
        pytest.fail("reward-v2 checkpoint reached PPO.learn without explicit transfer")

    monkeypatch.setattr(PPO, "learn", unexpected_learning)
    with pytest.raises(ValueError, match="control contract"):
        train(args)
    assert not (tmp_path/"resume-attempt"/"policy.zip").exists()


def test_direct_command_reward_v1_transfer_is_explicitly_unsupported(physical_motor, reward_v2_policy, tmp_path):
    from .command_motor_training import initialize_command_motor

    source = reward_v2_policy
    del source.command_motor_control_contract["rewardContract"]
    del source.command_motor_reward_contract
    checkpoint = tmp_path/"unsupported-reward-v1.zip"
    source.save(checkpoint)
    with pytest.raises(ValueError, match="direct reward-v1 is unsupported"):
        initialize_command_motor(checkpoint, physical_motor, body_env=physical_motor, seed=88701,
            n_steps=8, batch_size=8, n_epochs=1, initialize_from_command=True)


@pytest.mark.parametrize("change", ["float_actual", "bool_actual", "float_steps", "missing_ancestor",
                                     "float_ancestor", "negative_ancestor", "ancestor_after_actual",
                                     "missing_v2_initial", "float_v2_initial", "v2_initial_after_actual",
                                     "v2_initial_before_ancestor"])
def test_explicit_reward_transfer_rejects_invalid_physical_control_lineage(
        physical_motor, reward_v2_policy, tmp_path, change):
    from .command_motor_training import initialize_command_motor

    policy = reward_v2_policy
    if change == "float_actual":
        policy.command_motor_physical_control_timesteps = float(policy.num_timesteps)
    elif change == "bool_actual":
        policy.command_motor_physical_control_timesteps = True
    elif change == "float_steps":
        policy.num_timesteps = float(policy.num_timesteps)
    elif change == "missing_ancestor":
        del policy.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"]
    elif change == "float_ancestor":
        policy.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"] = 6_381_568.
    elif change == "negative_ancestor":
        policy.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"] = -1
    elif change == "ancestor_after_actual":
        policy.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"] = policy.num_timesteps+1
    elif change == "missing_v2_initial":
        del policy.command_motor_reward_v2_initial_physical_controls
    elif change == "float_v2_initial":
        policy.command_motor_reward_v2_initial_physical_controls = float(policy.command_motor_reward_v2_initial_physical_controls)
    elif change == "v2_initial_after_actual":
        policy.command_motor_reward_v2_initial_physical_controls = policy.num_timesteps+1
    else:
        policy.command_motor_reward_v2_initial_physical_controls = policy.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"]-1
    checkpoint = tmp_path/"bad-lineage.zip"
    policy.save(checkpoint)
    with pytest.raises(ValueError, match="lineage"):
        initialize_command_motor(checkpoint, physical_motor, body_env=physical_motor,
            seed=88701, n_steps=8, batch_size=8, n_epochs=1, initialize_from_command=True)


def _counter_metadata():
    return {"command_motor": {"physicalControlsExecuted": 1, "observedPhysicsSubstepsThisControl": 10},
            "navigation": {"split": "train", "sceneSeed": 88501, "physics_seed": 89501,
                           "scene_sha256": "test-scene", "collision_audit_timestep_seconds": .002}}


def _initialized_counter(infos):
    from .command_motor_training import make_counter
    counter = make_counter()
    counter.model = SimpleNamespace(command_motor_physical_control_timesteps=123, num_timesteps=125)
    counter.locals = {"infos": infos}
    return counter


def test_training_counts_one_physical_control_per_worker_and_records_train_scenes():
    infos = [_counter_metadata(), _counter_metadata()]
    infos[1]["navigation"]["sceneSeed"] += 1
    counter = _initialized_counter(infos)
    assert counter._on_step() is True
    assert counter.controls == 2
    assert counter.model.command_motor_physical_control_timesteps == 125
    assert counter.worker_scenes[0][88501]["split"] == "train"
    assert counter.worker_scenes[1][88502]["firstObservedCumulativeTimesteps"] == 125


@pytest.mark.parametrize("name,value", [
    ("physicalControlsExecuted", True), ("physicalControlsExecuted", 1.),
    ("physicalControlsExecuted", None), ("physicalControlsExecuted", 2),
    ("observedPhysicsSubstepsThisControl", True), ("observedPhysicsSubstepsThisControl", 10.),
    ("observedPhysicsSubstepsThisControl", None), ("observedPhysicsSubstepsThisControl", 9),
])
def test_inexact_physics_metadata_cannot_be_counted_as_training(name, value):
    info = _counter_metadata()
    info["command_motor"][name] = value
    counter = _initialized_counter([info])
    with pytest.raises(RuntimeError):
        counter._on_step()
    assert counter.controls == 0
    assert counter.model.command_motor_physical_control_timesteps == 123


@pytest.mark.parametrize("scene_seed", [True, 88501., "88501", -1, 1_000_000])
def test_heldout_or_coerced_scene_seeds_cannot_enter_motor_training(scene_seed):
    info = _counter_metadata()
    info["navigation"]["sceneSeed"] = scene_seed
    counter = _initialized_counter([info])
    with pytest.raises(RuntimeError, match="held-out"):
        counter._on_step()
    assert counter.controls == 0


@pytest.mark.parametrize("entropy", [float("nan"), float("inf"), -.01])
def test_nonfinite_or_negative_entropy_rejects_before_training(entropy):
    from .command_motor_training import _validated_args, argument_parser
    args = argument_parser().parse_args(["--source-policy", "source.zip", "--avatar-profile", "profile.json",
                                         "--output", "out"])
    args.entropy = entropy
    with pytest.raises(ValueError, match="entropy"):
        _validated_args(args)


def test_initialize_only_keeps_log_std_when_explicitly_requested():
    from .command_motor_training import _validated_args, argument_parser
    args = argument_parser().parse_args(["--initialize-from-command", "reward-v2.zip", "--initialize-only",
        "--keep-log-std", "--avatar-profile", "profile.json", "--output", "out"])
    assert args.initialize_only is True
    assert args.initialize_from_command == "reward-v2.zip"
    assert args.reset_log_std is None
    assert _validated_args(args) is args


def test_initialize_only_cannot_resume_or_create_an_output_before_validation(tmp_path):
    from .command_motor_training import argument_parser, train
    output = tmp_path/"invalid-initialization"
    args = argument_parser().parse_args(["--resume", "missing-checkpoint.zip", "--initialize-only",
        "--avatar-profile", "missing-profile.json", "--output", str(output)])
    with pytest.raises(ValueError, match="initialize-only requires an explicit source transfer"):
        train(args)
    assert not output.exists()


def test_keep_log_std_and_explicit_std_reset_are_mutually_exclusive():
    from .command_motor_training import argument_parser
    with pytest.raises(SystemExit):
        argument_parser().parse_args(["--initialize-from-command", "reward-v2.zip", "--keep-log-std",
            "--reset-log-std", "-2.4", "--avatar-profile", "profile.json", "--output", "out"])


def test_evaluate_and_initialize_only_reject_before_loading_a_checkpoint(monkeypatch):
    import sys
    from .command_motor_training import main
    monkeypatch.setattr(sys, "argv", ["command_motor_training", "--initialize-from-command", "missing.zip",
        "--evaluate-only", "--initialize-only", "--avatar-profile", "missing.json", "--output", "out"])
    with pytest.raises(ValueError, match="evaluate-only and initialize-only"):
        main()
