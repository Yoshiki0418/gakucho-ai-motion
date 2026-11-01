"""Real SB3 audits for the explicit, same-body 59-to-110 policy transfer."""
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from .body_contract import bind_policy_body
from .navigation_training import (NAVIGATION_OBSERVATION_CONTRACT, _validated_args,
                                  argument_parser, assert_navigation_policy,
                                  audit_transfer, initialize_from59, resume_from110)


class ContractEnv(gym.Env):
    def __init__(self, observation_count=110, *, body="same-body"):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_count,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (20,), np.float32)
        self.avatar_profile = {"avatarId": "same-avatar", "physical": {"rootHeight": 1.}}
        self.body_model_sha256 = body

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0., False, False, {}


@pytest.fixture
def source():
    torch.set_num_threads(1)
    env = ContractEnv(59)
    policy = PPO("MlpPolicy", env, seed=391, n_steps=8, batch_size=8,
                 policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, device="cpu")
    bind_policy_body(policy, env)
    policy.num_timesteps = 4096
    # Create nonempty real Adam moments without an environment learning run.
    policy.policy.optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in policy.policy.parameters()).backward()
    policy.policy.optimizer.step()
    return policy


def transfer(source, env=None):
    return initialize_from59(source, env or ContractEnv(), n_steps=8, batch_size=8)


def test_transfer_copies_all_parameters_and_zeros_only_new_sensor_columns(source):
    target, report = transfer(source)
    old = source.policy.state_dict()
    new = target.policy.state_dict()
    first_layers = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}
    for name, value in old.items():
        if name in first_layers:
            assert torch.equal(new[name][:, :59], value)
            assert torch.count_nonzero(new[name][:, 59:]).item() == 0
        else:
            assert torch.equal(new[name], value)
    assert target.num_timesteps == source.num_timesteps
    assert report["verified"]
    assert target.navigation_observation_contract == NAVIGATION_OBSERVATION_CONTRACT


def test_actor_critic_and_actions_are_preserved_for_arbitrary_sensor_inputs(source):
    target, _ = transfer(source)
    report = audit_transfer(source, target, seed=234, sample_count=64)
    assert report["verified"] and report["arbitrarySensorValues"]
    assert max(report[key] for key in report if key.endswith("MaxAbsError")) <= 1e-6
    with torch.no_grad():
        target.policy.mlp_extractor.policy_net[0].weight[0, 59] = .5
    with pytest.raises(ValueError, match="changed the source outputs"):
        audit_transfer(source, target)


def test_optimizer_is_rebuilt_with_new_parameters_and_no_old_adam_moments(source):
    assert source.policy.optimizer.state
    target, report = transfer(source)
    assert not target.policy.optimizer.state
    old_ids = {id(parameter) for group in source.policy.optimizer.param_groups for parameter in group["params"]}
    new_ids = {id(parameter) for group in target.policy.optimizer.param_groups for parameter in group["params"]}
    assert not old_ids & new_ids
    assert not report["optimizerStateReused"]


def test_body_profile_and_body_geometry_must_both_match(source):
    with pytest.raises(ValueError, match="physical body contract"):
        transfer(source, ContractEnv(body="different-body"))
    env = ContractEnv()
    env.avatar_profile = {"avatarId": "different-avatar"}
    with pytest.raises(ValueError, match="body profile"):
        transfer(source, env)


def test_wrong_input_action_bounds_and_architecture_are_rejected(source):
    source.observation_space = gym.spaces.Box(-np.inf, np.inf, (110,), np.float32)
    with pytest.raises(ValueError, match="59 observations"):
        transfer(source)
    source.observation_space = gym.spaces.Box(-np.inf, np.inf, (59,), np.float32)
    env = ContractEnv()
    env.action_space = gym.spaces.Box(-.5, .5, (20,), np.float32)
    with pytest.raises(ValueError, match="action bounds"):
        transfer(source, env)
    source.policy_kwargs["net_arch"] = {"pi": [64, 64], "vf": [64, 64]}
    with pytest.raises(ValueError, match="128x128"):
        transfer(source)


def test_saved_checkpoint_records_contract_and_rejects_missing_semantics(source, tmp_path):
    env = ContractEnv()
    target, _ = transfer(source, env)
    target.save(tmp_path/"navigation")
    loaded = PPO.load(tmp_path/"navigation.zip", device="cpu")
    assert_navigation_policy(loaded, env)
    assert loaded.navigation_transfer_audit["verified"]
    loaded.navigation_observation_contract = {"version": 0}
    with pytest.raises(ValueError, match="observation contract"):
        assert_navigation_policy(loaded, env)


def test_navigation_resume_rejects_sensor_bounds_change(source):
    env = ContractEnv()
    target, _ = transfer(source, env)
    env.observation_space = gym.spaces.Box(-2., 2., (110,), np.float32)
    with pytest.raises(ValueError, match="observation bounds"):
        assert_navigation_policy(target, env)


def test_resume_restores_real_sb3_optimizer_but_uses_the_requested_worker_seeds(source, tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    target, _ = transfer(source)
    target.policy.optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in target.policy.parameters()).backward()
    target.policy.optimizer.step()
    target.save(tmp_path/"navigation")
    vector = DummyVecEnv([ContractEnv, ContractEnv, ContractEnv])
    requested_seed = 8507
    assert target.seed != requested_seed
    try:
        loaded = resume_from110(tmp_path/"navigation.zip", vector, body_env=ContractEnv(),
                                seed=requested_seed, learning_rate=8e-5, entropy=.0003)
        assert loaded.seed == requested_seed
        assert vector._seeds == [requested_seed, requested_seed+1, requested_seed+2]
        assert loaded.num_timesteps == target.num_timesteps
        assert loaded.policy.optimizer.state
        for name, value in target.policy.state_dict().items():
            assert torch.equal(value, loaded.policy.state_dict()[name])
        for original, restored in zip(target.policy.optimizer.state.values(), loaded.policy.optimizer.state.values()):
            assert torch.equal(original["exp_avg"], restored["exp_avg"])
            assert torch.equal(original["exp_avg_sq"], restored["exp_avg_sq"])
        vector.reset()
        # The first real reset consumes exactly the newly requested seed for
        # each worker, rather than the seed embedded in the saved checkpoint.
        assert [env.np_random_seed for env in vector.envs] == [requested_seed, requested_seed+1, requested_seed+2]
    finally:
        vector.close()


def test_zero_residual_baseline_needs_no_policy_contract():
    assert_navigation_policy(None, ContractEnv())


@pytest.mark.parametrize("changes,reason", [
    ({"steps": 0}, "steps"), ({"seconds": float("nan")}, "seconds"),
    ({"seed": 999999, "envs": 8}, "held-out"),
    ({"difficulty_min": .7, "difficulty_max": .2}, "difficulty"),
    ({"learning_rate": 0.}, "learning-rate"),
    ({"entropy": float("inf")}, "entropy"),
    ({"resume110": "also.zip"}, "exactly one"),
])
def test_cli_rejects_invalid_training_contract(changes, reason):
    args = SimpleNamespace(steps=4096, seconds=30., envs=8, checkpoint_every=100000,
                           seed=8201, difficulty_min=0., difficulty_max=.2,
                           learning_rate=1e-4, entropy=.001, reset_log_std=None,
                           initialize_from59="source.zip", resume110=None, mode="turn")
    vars(args).update(changes)
    with pytest.raises(ValueError, match=reason):
        _validated_args(args)


def test_cli_requires_an_explicit_exclusive_initialization():
    parser = argument_parser()
    base = ["--mode", "turn", "--avatar-profile", "profile.json", "--output", "out"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(SystemExit):
        parser.parse_args(base+["--initialize-from59", "a.zip", "--resume110", "b.zip"])


def test_actual_measured_avatar_walking_checkpoint_transfer():
    # This is a checkpoint/network audit, not a training or physics rollout.
    root = Path(__file__).resolve().parents[1]
    checkpoint = root/".runtime/world-model/stage1-gakucho-walking-refined/policy.zip"
    profile = root/"frontend/public/world-model/gakucho-avatar-profile.json"
    if not checkpoint.is_file():
        pytest.skip("local measured-avatar phase1 checkpoint is unavailable")
    from .locomotion import FootstepHumanoidEnv
    from .sensors import SceneSensorWrapper
    torch.set_num_threads(1)
    env = SceneSensorWrapper(FootstepHumanoidEnv(task="walk", avatar_profile=profile,
                                               reference_start=False, strict_gait_reward=True))
    try:
        old = PPO.load(checkpoint, device="cpu")
        target, report = initialize_from59(old, env, n_steps=8, batch_size=8)
        assert_navigation_policy(target, env)
        assert report["verified"]
        assert target.avatar_profile_sha256 == old.avatar_profile_sha256
        assert target.body_model_sha256 == old.body_model_sha256
        assert target.num_timesteps == 6_176_768
    finally:
        env.close()
