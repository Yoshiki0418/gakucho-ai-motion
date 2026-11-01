"""Actual physics and continuation audits for balanced local-command training."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from .body_contract import bind_policy_body
from .command_motor import (COMMAND_MOTOR_OBSERVATION_CONTRACT,
                            COMMAND_MOTOR_REWARD_CONTRACT,
                            CommandMotorEnv, observed_command_angle)
from .command_motor_curriculum import (LOCAL_COMMAND_BAG_COUNTS,
                                      LOCAL_COMMAND_OFFSETS,
                                      CommandMotorCurriculumEnv)


PROFILE = Path(__file__).resolve().parents[1]/"frontend/public/world-model/gakucho-avatar-profile.json"
ARGUMENTS = {"mode": "turn", "split": "train", "base_seed": 90701,
    "difficulty_range": (.25, .25), "randomize": True,
    "max_episode_seconds": 8., "avatar_profile": PROFILE,
    "command_lookahead_range": (.35, .5), "record_trajectory": True}
OPTIONS = {"scene_seed": 90701, "physics_seed": 91701}


def test_curriculum_changes_only_post_physics_driver_not_initial_body_or_observation():
    original = CommandMotorEnv(**ARGUMENTS)
    sampled = CommandMotorCurriculumEnv(**ARGUMENTS)
    try:
        obs_original, _ = original.reset(seed=90701, options=OPTIONS)
        obs_sampled, info = sampled.reset(seed=90701, options=OPTIONS)
        np.testing.assert_array_equal(obs_original, obs_sampled)
        for name in ("qpos", "qvel", "qacc", "xpos", "geom_xpos", "ctrl",
                     "xfrc_applied", "qfrc_applied"):
            np.testing.assert_array_equal(getattr(original.unwrapped.data, name),
                                          getattr(sampled.unwrapped.data, name), err_msg=name)
        for name in ("body_mass", "body_inertia", "geom_friction"):
            np.testing.assert_array_equal(getattr(original.unwrapped.model, name),
                                          getattr(sampled.unwrapped.model, name), err_msg=name)
        np.testing.assert_array_equal(original.unwrapped.goal, sampled.unwrapped.goal)
        assert original.control_contract == sampled.control_contract
        assert original._lookahead == sampled._lookahead
        assert sampled._effective_heading_offset == 0.
        assert info["command_motor"]["nextCommand"]["requestedHeadingOffsetRadians"] in LOCAL_COMMAND_OFFSETS
        assert sampled.training_command_generator_contract != original.training_command_generator_contract
        assert sampled.training_command_generator_contract["sampleAfterActualPhysicsReset"] is True
    finally:
        original.close()
        sampled.close()


def test_complete_actual_train_reset_bag_is_balanced_reproducible_and_half_saturated():
    first = CommandMotorCurriculumEnv(**ARGUMENTS)
    second = CommandMotorCurriculumEnv(**ARGUMENTS)
    try:
        offsets = []
        for index in range(20):
            options = {"scene_seed": 90701+index, "physics_seed": 91701+index}
            left_obs, left = first.reset(seed=90701+index, options=options)
            right_obs, right = second.reset(seed=90701+index, options=options)
            np.testing.assert_array_equal(left_obs, right_obs)
            assert left["command_motor"]["nextCommand"] == right["command_motor"]["nextCommand"]
            offsets.append(left["command_motor"]["nextCommand"]["requestedHeadingOffsetRadians"])
        counts = [offsets.count(offset) for offset in LOCAL_COMMAND_OFFSETS]
        assert counts == list(LOCAL_COMMAND_BAG_COUNTS)
        assert counts == counts[::-1]
        assert (counts[0]+counts[-1])/len(offsets) == .5
        assert first._curriculum_bag == []
        assert first.training_command_generator_contract["actualConsumedSaturatedControlFractionGuaranteed"] is False
    finally:
        first.close()
        second.close()


def test_actual_100ms_curriculum_is_pure_driver_with_consumed_observed_reward_and_2ms_guards():
    env = CommandMotorCurriculumEnv(**ARGUMENTS)
    shadow = CommandMotorEnv(**ARGUMENTS)
    try:
        observation, _ = env.reset(seed=90701, options=OPTIONS)
        shadow.reset(seed=90701, options=OPTIONS)
        body = env.unwrapped
        goal_object, goal_value = body.goal, body.goal.copy()
        times = []
        original_observer = body._physics_contact_observer

        def observe(model, data):
            assert body.goal is goal_object
            np.testing.assert_array_equal(body.goal, goal_value)
            assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
            times.append(float(data.time))
            original_observer(model, data)

        body._physics_contact_observer = observe
        requested = env._curriculum_requested_offset
        action = np.linspace(-.05, .05, 20)
        for _ in range(5):
            # Advance a second unchanged physical motor with exactly the same
            # consumed command; its later analytic driver does not alter this
            # integration. Compare actual native state, rather than flags.
            shadow._next_command_goal = env.next_command_goal
            shadow.unwrapped.command_goal = env.next_command_goal
            before = env._effective_heading_offset
            observed_angle, observed_distance = observed_command_angle(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            _, shadow_reward, shadow_done, shadow_truncated, shadow_info = shadow.step(action)
            assert (terminated, truncated) == (shadow_done, shadow_truncated) == (False, False)
            assert reward == shadow_reward
            assert info["command_motor"]["trackingReward"] == shadow_info["command_motor"]["trackingReward"]
            assert env._curriculum_requested_offset == requested
            assert abs(env._effective_heading_offset-before) <= .175*.02+1e-12
            tracking = info["command_motor"]["trackingReward"]
            assert tracking["consumedCommandRelativeHeadingRadians"] == pytest.approx(observed_angle, abs=2e-6)
            assert tracking["measurements"]["commandDistanceMeters"] == pytest.approx(observed_distance, abs=2e-6)
            assert not any("requested" in key.lower() for key in tracking["measurements"])
            assert info["command_motor"]["observedPhysicsSubstepsThisControl"] == 10
            assert info["command_motor"]["rewardContract"] == COMMAND_MOTOR_REWARD_CONTRACT
            for name in ("qpos", "qvel", "qacc", "ctrl", "actuator_force", "geom_xpos", "xpos"):
                np.testing.assert_array_equal(getattr(body.data, name), getattr(shadow.unwrapped.data, name), err_msg=name)
            snapshot_state = {name: getattr(body.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
            before_count = env._episode_observed_substeps
            env.snapshot()
            env.snapshot()
            assert env._episode_observed_substeps == before_count
            for name, values in snapshot_state.items():
                np.testing.assert_array_equal(getattr(body.data, name), values)
        np.testing.assert_allclose(times, np.arange(1, 51)*.002, rtol=0, atol=1e-15)
        assert body.data.time == pytest.approx(.1)
        # Updating a requested command alone has no physical or final-goal
        # mutation, even though the short reference command changes.
        state = {name: getattr(body.data, name).copy() for name in ("qpos", "qvel", "ctrl", "qacc")}
        env._update_next_command(advance=True)
        for name, values in state.items():
            np.testing.assert_array_equal(getattr(body.data, name), values)
        assert body.goal is goal_object
        np.testing.assert_array_equal(body.goal, goal_value)
    finally:
        env.close()
        shadow.close()


@pytest.mark.parametrize("change,reason", [
    ({"split": "heldout"}, "actual train scenes"),
    ({"mode": "obstacle"}, "empty turn rooms"),
    ({"max_episode_seconds": 5.}, "6..8"),
    ({"max_episode_seconds": 9.}, "6..8"),
    ({"heading_slew_radians_per_second": .5}, "existing .175"),
    ({"max_heading_radians": .3}, "existing .45"),
])
def test_curriculum_rejects_hidden_control_changes_or_heldout_use(change, reason):
    with pytest.raises(ValueError, match=reason):
        CommandMotorCurriculumEnv(**{**ARGUMENTS, **change})


def _checkpoint_with_real_adam(env, path):
    """Synthetic lineage fixture; real optimizer moments, no claimed RL data."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    policy = PPO("MlpPolicy", env, seed=71, device="cpu", gamma=.99,
        n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}})
    policy.policy.optimizer.zero_grad()
    sum(parameter.square().sum()+parameter.sum() for parameter in policy.policy.parameters()).backward()
    policy.policy.optimizer.step()
    policy.command_motor_observation_contract = copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT)
    policy.command_motor_control_contract = copy.deepcopy(env.control_contract)
    policy.command_motor_reward_contract = copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT)
    policy.command_motor_transfer_audit = {"sourcePhysicalMotorTimesteps": 0}
    policy.num_timesteps = policy.command_motor_physical_control_timesteps = 1024
    policy.command_motor_reward_v3_initial_physical_controls = 512
    policy._n_updates = 3
    bind_policy_body(policy, env.unwrapped)
    policy.save(path)
    return policy


def test_real_ppo_resume_keeps_full_actor_critic_std_adam_epochs_and_counts(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .command_motor_training import resume_command_motor, _state_sha256
    env = CommandMotorCurriculumEnv(**ARGUMENTS)
    env.reset(seed=90701, options=OPTIONS)
    checkpoint = tmp_path/"source.zip"
    source = _checkpoint_with_real_adam(env, checkpoint)
    zip_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    vector = DummyVecEnv([lambda: env])
    try:
        target, audit = resume_command_motor(checkpoint, vector, body_env=env, seed=90701)
        assert _state_sha256(target.policy.state_dict()) == _state_sha256(source.policy.state_dict())
        assert _state_sha256(target.policy.optimizer.state_dict()) == _state_sha256(source.policy.optimizer.state_dict())
        assert target.num_timesteps == target.command_motor_physical_control_timesteps == 1024
        assert target._n_updates == 3
        assert target.command_motor_reward_v3_initial_physical_controls == 512
        assert audit["sampleCount"] == 32
        assert audit["deterministicActionsMaxAbsError"] == audit["criticValuesMaxAbsError"] == 0
        assert audit["physicalControlsAddedByResumeAudit"] == audit["ppoUpdateEpochsAddedByResumeAudit"] == 0
        assert audit["allAdamStatePreservedExactly"] is True
        assert audit["freshAdam"] is audit["criticReset"] is False
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == zip_sha
    finally:
        vector.close()


@pytest.mark.parametrize("extra,reason", [
    ([], "keep-log-std"),
    (["--keep-log-std", "--seconds", "30"], "6..8"),
    (["--keep-log-std", "--seconds", "8", "--evaluate-only"], "held-out evaluation"),
])
def test_curriculum_cli_rejects_undeclared_std_or_eval_changes(extra, reason):
    from .command_motor_training import argument_parser, _validated_args
    args = argument_parser().parse_args(["--resume", "source.zip", "--local-command-curriculum",
        "--seconds", "8", "--avatar-profile", str(PROFILE), "--output", "unused", *extra])
    with pytest.raises(ValueError, match=reason):
        _validated_args(args)


def test_curriculum_cli_explicit_resume_preserves_std_and_pins_extra_source():
    from .command_motor_training import argument_parser, _validated_args, _source_hashes
    args = argument_parser().parse_args(["--resume", "source.zip", "--local-command-curriculum",
        "--keep-log-std", "--seconds", "8", "--lookahead-min", ".35", "--lookahead-max", ".5",
        "--steps", "512000", "--seed", "90701", "--resume-audit-only",
        "--avatar-profile", str(PROFILE), "--output", "unused"])
    assert _validated_args(args) is args
    assert args.reset_log_std is None
    sources = _source_hashes()
    assert len(sources) == 22
    path = Path(__file__).parent/"command_motor_curriculum.py"
    assert sources[path.name] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_counter_distinguishes_requested_episodes_from_real_consumed_commands():
    from types import SimpleNamespace
    from .command_motor_training import make_counter
    from .test_command_motor import _counter_metadata
    counter = make_counter()
    counter.model = SimpleNamespace(command_motor_physical_control_timesteps=100, num_timesteps=102)
    infos = []
    for requested, actual, episode in ((.45, 0., 1), (.45, .45, 1), (-.45, -.45, 2)):
        info = _counter_metadata()
        info["command_motor"].update({
            "nextCommand": {"requestedHeadingOffsetRadians": requested, "curriculumEpisodeIndex": episode},
            "trackingReward": {"consumedCommandRelativeHeadingRadians": actual,
                "measurements": {"commandDistanceMeters": .4}}})
        infos.append(info)
    for info in infos:
        counter.locals = {"infos": [info]}
        counter._on_step()
    coverage = counter.curriculum_coverage
    assert coverage["observedEpisodeCount"] == 2
    assert coverage["requestedOffsetEpisodeCounts"] == {"+0.45": 1, "-0.45": 1}
    assert coverage["actualConsumedControlCount"] == 3
    assert coverage["actualConsumedZeroControls"] == 1
    assert coverage["actualConsumedSaturatedNegativeControls"] == coverage["actualConsumedSaturatedPositiveControls"] == 1
    assert counter.controls == 3
