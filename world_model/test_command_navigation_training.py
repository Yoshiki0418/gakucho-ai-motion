"""Audit actual PPO accounting/resumption, split boundaries and trace sampling."""
import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .body_contract import bind_policy_body
from .command_navigation import COMMAND_OBSERVATION_CONTRACT, assert_command_policy
from .command_navigation_evaluation import _scene_seeds, append_control_frames
from .command_navigation_training import (_validated_args, argument_parser,
    make_control_counter, resume_command)


class CountedCommandEnv(gym.Env):
    def __init__(self, terminal_every=3):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (112,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
        self.avatar_profile = {"avatarId": "test-avatar"}
        self.body_model_sha256 = "test-body"
        self.control_contract = {"motorCheckpointSha256": "test-motor", "decisionControlSteps": 5,
            "rewardDiscountPerDecision": .99, "rewardScale": .01,
            "minWalkingLookaheadMeters": .35, "stopLookaheadActionThreshold": -.8}
        self.terminal_every = terminal_every
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(112, dtype=np.float32), {}

    def step(self, action):
        self.index += 1
        terminal = self.index >= self.terminal_every
        return np.zeros(112, dtype=np.float32), 1., terminal, False, {
            "command_control_steps": 1 if terminal else 5}


def command_policy(env, *, seed=12):
    torch.set_num_threads(1)
    policy = PPO("MlpPolicy", env, seed=seed, n_steps=8, batch_size=8, n_epochs=1, device="cpu",
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
    body = env.envs[0] if isinstance(env, DummyVecEnv) else env
    bind_policy_body(policy, body)
    policy.command_observation_contract = copy.deepcopy(COMMAND_OBSERVATION_CONTRACT)
    policy.command_control_contract = copy.deepcopy(body.control_contract)
    policy.command_physical_control_timesteps = 0
    return policy


def test_real_ppo_counts_short_terminal_decisions_and_each_worker_exactly():
    vector = DummyVecEnv([lambda: CountedCommandEnv(1), lambda: CountedCommandEnv(3)])
    try:
        policy = command_policy(vector)
        counter = make_control_counter(5)
        policy.learn(total_timesteps=16, callback=counter)
        assert policy.num_timesteps == 16
        # Worker 1 executes [1]*8; worker 2 [5,5,1,5,5,1,5,5].
        assert policy.command_physical_control_timesteps == 40
        assert counter.run_physical_control_steps == 40
        assert policy.command_physical_control_timesteps < policy.num_timesteps*5
    finally:
        vector.close()


@pytest.mark.parametrize("bad_count", [None, 0, 6, True, 1.5])
def test_physical_accounting_rejects_missing_inexact_or_impossible_counts(bad_count):
    class Invalid(CountedCommandEnv):
        def step(self, action):
            obs, reward, done, trunc, info = super().step(action)
            info["command_control_steps"] = bad_count
            return obs, reward, done, trunc, info
    vector = DummyVecEnv([Invalid])
    try:
        policy = command_policy(vector)
        with pytest.raises(ValueError, match="exact executed motor ticks"):
            policy.learn(total_timesteps=8, callback=make_control_counter(5))
    finally:
        vector.close()


def test_resume_restores_optimizer_counts_and_reseeds_workers(tmp_path):
    original = command_policy(CountedCommandEnv())
    original.learn(total_timesteps=8, callback=make_control_counter(5))
    original.save(tmp_path/"command")
    vector = DummyVecEnv([CountedCommandEnv, CountedCommandEnv])
    try:
        loaded = resume_command(tmp_path/"command.zip", vector, body_env=CountedCommandEnv(),
            seed=8731, learning_rate=3e-5, entropy=.0005)
        assert loaded.command_physical_control_timesteps == original.command_physical_control_timesteps
        assert loaded.num_timesteps == original.num_timesteps
        assert vector._seeds == [8731, 8732]
        assert loaded.policy.optimizer.state
        for name, value in original.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[name])
        for old, new in zip(original.policy.optimizer.state.values(), loaded.policy.optimizer.state.values()):
            assert torch.equal(old["exp_avg"], new["exp_avg"])
            assert torch.equal(old["exp_avg_sq"], new["exp_avg_sq"])
        vector.reset()
        assert [env.np_random_seed for env in vector.envs] == [8731, 8732]
    finally:
        vector.close()


@pytest.mark.parametrize("change", [{"motorCheckpointSha256": "other"}, {"decisionControlSteps": 4}])
def test_motor_or_decision_cadence_changes_reject_resumption(change):
    env = CountedCommandEnv()
    policy = command_policy(env)
    env.control_contract.update(change)
    with pytest.raises(ValueError, match="motor or control parameters differ"):
        assert_command_policy(policy, env)


def test_cli_defaults_match_stable_motor_probe_bounds():
    args = argument_parser().parse_args(["--mode", "obstacle", "--motor-policy", "motor.zip",
        "--avatar-profile", "profile.json", "--output", "out"])
    assert _validated_args(args) is args
    assert args.max_lookahead_m == .5 and args.decision_steps == 5
    assert args.heading_slew_radians_per_second == .175


@pytest.mark.parametrize("changes,reason", [
    ({"seed": 999999, "envs": 8}, "held-out"),
    ({"max_lookahead_m": .7}, "lookahead"),
    ({"max_lookahead_m": .25}, "lookahead"),
    ({"heading_slew_radians_per_second": float("nan")}, "heading-slew"),
    ({"decision_steps": 11}, "decision-steps"),
    ({"difficulty_min": .4, "difficulty_max": .2}, "difficulty"),
])
def test_training_rejects_scene_leakage_and_unstable_control_parameters(changes, reason):
    args = argument_parser().parse_args(["--mode", "obstacle", "--motor-policy", "motor.zip",
        "--avatar-profile", "profile.json", "--output", "out"])
    vars(args).update(changes)
    with pytest.raises(ValueError, match=reason):
        _validated_args(args)


def test_evaluation_checks_all_heldout_seeds_before_constructing_physics():
    assert _scene_seeds([1_100_000, 1_100_001]) == [1_100_000, 1_100_001]
    for seeds in ([999999], [2_000_000], [1_100_000, 1_100_000], []):
        with pytest.raises(ValueError):
            _scene_seeds(seeds)


def test_actual_twenty_ms_frames_are_retained_across_decision_boundaries():
    trace = {"frames": [{"time": 0.}]}
    frames = [{"time": value} for value in (.02, .04, .06, .08, .10)]
    append_control_frames(trace, frames, expected_count=5, control_dt=.02)
    append_control_frames(trace, [{"time": .12}], expected_count=1, control_dt=.02)
    assert len(trace["frames"]) == 7
    assert trace["frames"][1:] == frames+[{"time": .12}]


@pytest.mark.parametrize("frames,count", [
    ([{"time": .1}], 5), ([{"time": .1}], 1),
    ([{"time": .02}, {"time": .02}], 2),
])
def test_downsampled_or_duplicate_frames_cannot_hide_brief_support_losses(frames, count):
    trace = {"frames": [{"time": 0.}]}
    with pytest.raises(ValueError):
        append_control_frames(trace, frames, expected_count=count, control_dt=.02)
    assert trace["frames"] == [{"time": 0.}]


def make_physical_command_stub(monkeypatch, tmp_path, *, reason=None, ticks=None,
                               base_per_tick=0., terminal_amount=0., motor_hip_yaw_action_limit=None):
    from . import command_navigation as commands

    class PhysicalTerminalEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (110,), np.float32)
            self.action_space = gym.spaces.Box(-1., 1., (20,), np.float32)
            self.dt = .02
            self.goal = np.array([0., -2., 0.])
            self.pelvis_body_id = 0
            self.data = SimpleNamespace(qpos=np.array([0., 0., 1.]),
                xmat=np.eye(3).reshape(1, 9), xfrc_applied=np.zeros((1, 6)), qfrc_applied=np.zeros(20))
            self.index = 0

        def reset(self, *, seed=None, options=None):
            self.index = 0
            return np.zeros(110, dtype=np.float32), {}

        def step(self, action):
            self.index += 1
            done = self.index == ticks
            terminal = terminal_amount if done else 0.
            info = {"fall": done and reason in ("fall", "collision_and_fall"),
                "navigation": {"terminal_reason": reason if done else None,
                    "reward": {"terminal_cost": max(-terminal, 0.),
                        "success_bonus": max(terminal, 0.)}}}
            return np.zeros(110, dtype=np.float32), base_per_tick+terminal, done and reason != "time_limit", done and reason == "time_limit", info

    physical = PhysicalTerminalEnv()
    motor = SimpleNamespace(policy=SimpleNamespace(set_training_mode=lambda mode: None,
        parameters=lambda: iter(())),
        predict=lambda observation, deterministic: (np.zeros(20, dtype=np.float32), None))
    monkeypatch.setattr(commands, "_CommandTaskNavigationEnv", lambda **kwargs: physical)
    monkeypatch.setattr(commands, "assert_navigation_policy", lambda *args: None)
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: motor)
    checkpoint = tmp_path/"motor.zip"
    checkpoint.write_bytes(b"test motor contract bytes")
    return commands.CommandNavigationEnv(motor_checkpoint=checkpoint, decision_steps=5,
        motor_hip_yaw_action_limit=motor_hip_yaw_action_limit)


@pytest.mark.parametrize("limit", [None, .15])
def test_only_explicit_versioned_motor_limit_changes_two_actual_joint_commands(monkeypatch, tmp_path, limit):
    from .humanoid import JOINT_NAMES
    env = make_physical_command_stub(monkeypatch, tmp_path, motor_hip_yaw_action_limit=limit)
    raw = np.linspace(-.8, .8, 20, dtype=np.float32)
    raw_before = raw.copy()
    applied = []
    physical_step = env.env.step

    def observe_action(action):
        applied.append(action.copy())
        return physical_step(action)

    env.env.step = observe_action
    env._motor.predict = lambda observation, deterministic: (raw, None)
    try:
        env.reset(seed=18)
        original_goal = env.unwrapped.goal
        goal_bytes = original_goal.tobytes()
        env.step(np.array([.5, 1.], dtype=np.float32))
        expected = raw.copy()
        yaw = [JOINT_NAMES.index(f"{side}_hip_yaw") for side in ("left", "right")]
        if limit is not None:
            expected[yaw] = np.clip(expected[yaw], -limit, limit)
            assert env.control_contract["version"] == 2
            assert env.control_contract["motorHipYawResidualActionLimit"] == limit
        else:
            assert env.control_contract["version"] == 1
            assert "motorHipYawResidualActionLimit" not in env.control_contract
        assert len(applied) == 5
        assert all(action.tobytes() == expected.tobytes() for action in applied)
        assert raw.tobytes() == raw_before.tobytes()
        assert env.unwrapped.goal is original_goal and original_goal.tobytes() == goal_bytes
        assert not np.any(env.unwrapped.data.xfrc_applied)
        assert not np.any(env.unwrapped.data.qfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize("limit", [0., -1., 1.01, float("nan"), float("inf")])
def test_training_rejects_invalid_motor_limit_before_starting_physics(limit):
    args = argument_parser().parse_args(["--mode", "turn", "--motor-policy", "motor.zip",
        "--avatar-profile", "profile.json", "--output", "out"])
    args.motor_hip_yaw_action_limit = limit
    with pytest.raises(ValueError, match="motor-hip-yaw"):
        _validated_args(args)


@pytest.mark.parametrize("reason,ticks,base_per_tick,terminal_amount,expected", [
    ("fall", 1, -8., -1500., -75.08),
    ("collision", 3, -8., -1500., -75.24),
    ("collision_and_fall", 5, -8., -1500., -75.40),
    ("success", 2, 8., 1500., 75.16),
    ("time_limit", 2, -8., 0., -.16),
])
def test_terminal_reward_preserves_ticks_and_scales_only_the_terminal_incentive(
        monkeypatch, tmp_path, reason, ticks, base_per_tick, terminal_amount, expected):
    """A shorter failing decision must not reduce or duplicate the final cost."""
    env = make_physical_command_stub(monkeypatch, tmp_path, reason=reason, ticks=ticks,
        base_per_tick=base_per_tick, terminal_amount=terminal_amount)
    try:
        env.reset(seed=18)
        _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert reward == pytest.approx(expected, abs=1e-10)
        assert info["command_control_steps"] == ticks
        assert terminated is (reason != "time_limit") and truncated is (reason == "time_limit")
        # A decision ending after one tick pays the same terminal incentive as
        # one ending after all five; ordinary tick rewards are summed exactly.
        actual_terminal = reward-base_per_tick*ticks*.01
        assert actual_terminal == pytest.approx(terminal_amount*5*.01, abs=1e-10)
        audit = info["command_navigation"]["reward"]
        assert audit["physicalControlRewardSum"] == base_per_tick*ticks+terminal_amount
        assert audit["terminalRewardAdjustment"] == terminal_amount*4
        assert audit["commandDecisionReward"] == reward
        assert audit["unscaledCommandDecisionReward"] == base_per_tick*ticks+terminal_amount*5
        assert audit["rewardScale"] == .01
        assert audit["executedPhysicalControlSteps"] == ticks
        assert audit["unscaledTerminalFailureCost"] == 7500.
        assert audit["terminalFailureCost"] == 75.
        assert audit["unscaledTerminalSuccessBonus"] == 7500.
        assert audit["terminalSuccessBonus"] == 75.
        assert audit["unscaledNegativeContinuingRewardLowerBound"] == pytest.approx(-4000., abs=1e-9)
        assert audit["negativeContinuingRewardLowerBound"] == pytest.approx(-40., abs=1e-9)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(2, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize("split,scene_seed,passes", [("train", 8731, True),
    ("heldout", 1_100_000, False), ("train", 1_100_000, False)])
def test_actual_worker_scene_audit_records_training_episodes_and_rejects_heldout_leakage(split, scene_seed, passes):
    class Audited(CountedCommandEnv):
        def __init__(self):
            super().__init__()
            self.episode = -1

        def reset(self, **kwargs):
            self.episode += 1
            return super().reset(**kwargs)

        def step(self, action):
            obs, reward, done, trunc, info = super().step(action)
            info["navigation"] = {"split": split, "scene_seed": scene_seed,
                "physics_seed": 8741+self.episode, "episode_index": self.episode,
                "scene_sha256": "a"*64, "difficulty": .1}
            return obs, reward, done, trunc, info

    vector = DummyVecEnv([Audited])
    try:
        policy = command_policy(vector)
        counter = make_control_counter(5, audit_scenes=True)
        if not passes:
            with pytest.raises(ValueError, match="train split|does not belong"):
                policy.learn(total_timesteps=8, callback=counter)
        else:
            policy.learn(total_timesteps=8, callback=counter)
            episodes = counter.worker_scenes[0]
            assert len(episodes) == 3
            assert [entry["episodeIndex"] for entry in episodes] == [0, 1, 2]
            assert all(entry["sceneSeed"] == scene_seed and entry["split"] == "train" for entry in episodes)
            assert [entry["physicsSeed"] for entry in episodes] == [8741, 8742, 8743]
    finally:
        vector.close()


@pytest.mark.parametrize("action,requested,effective", [
    (-.8, .35, .35), (1., .5, .5), (-.81, 0., 0.),
])
def test_motion_commands_use_verified_walking_range_and_explicit_stop(
        monkeypatch, tmp_path, action, requested, effective):
    env = make_physical_command_stub(monkeypatch, tmp_path)
    try:
        env.reset(seed=18)
        original_goal = env.unwrapped.goal.copy()
        obs, _, terminated, truncated, info = env.step(np.array([0., action]))
        command = info["command_navigation"]["command"]
        assert command["lookaheadMeters"] == pytest.approx(requested)
        assert command["effectiveLookaheadMeters"] == pytest.approx(effective)
        assert not terminated and not truncated
        assert obs[-1] == pytest.approx(action)
        assert np.array_equal(original_goal, env.unwrapped.goal)
    finally:
        env.close()


def test_moving_command_still_caps_distance_at_the_actual_final_destination(monkeypatch, tmp_path):
    env = make_physical_command_stub(monkeypatch, tmp_path)
    try:
        env.reset(seed=18)
        # Test-only fixture initialization places the task goal 0.1m ahead.
        env.unwrapped.goal[:] = [0., -.1, 0.]
        _, _, _, _, info = env.step(np.array([0., 1.]))
        command = info["command_navigation"]["command"]
        assert command["lookaheadMeters"] == .5
        assert command["effectiveLookaheadMeters"] == pytest.approx(.1)
        assert np.array_equal(env.unwrapped.goal, [0., -.1, 0.])
    finally:
        env.close()


def test_discount_horizon_and_changed_pace_semantics_are_rejected():
    env = CountedCommandEnv()
    policy = command_policy(env)
    policy.gamma = .98
    with pytest.raises(ValueError, match="discount horizon"):
        assert_command_policy(policy, env)
    policy.gamma = .99
    policy.command_control_contract.pop("minWalkingLookaheadMeters")
    with pytest.raises(ValueError, match="motor or control parameters"):
        assert_command_policy(policy, env)


@pytest.mark.parametrize("swings,supported,expected", [
    ([0, 2], True, False), ([1, 3], True, False), ([2, 2], True, True),
    ([3, 2], False, False),
])
def test_arrival_gate_requires_actual_sustained_swings_on_both_legs(monkeypatch, swings, supported, expected):
    from .command_navigation import _CommandTaskNavigationEnv
    from .navigation import EpisodeNavigationEnv

    # Existing navigation tests audit support, distance and 0.5s dwell. Here
    # isolate the added body-counter condition without fabricating a gait.
    monkeypatch.setattr(EpisodeNavigationEnv, "_metrics", lambda self, info: {"instant_success": supported})
    body = CountedCommandEnv()
    body._swing_counts = np.asarray(swings, dtype=int)
    wrapper = _CommandTaskNavigationEnv.__new__(_CommandTaskNavigationEnv)
    gym.Wrapper.__init__(wrapper, body)
    metrics = wrapper._metrics({})
    assert metrics["instant_success"] is expected
    assert metrics["actual_gait_swing_counts"] == swings
    assert metrics["minimum_required_swing_count_per_foot"] == 2
