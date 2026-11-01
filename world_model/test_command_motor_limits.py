"""Only yaw actions may change; navigator weights/counters never train here."""
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from .body_contract import bind_policy_body
from .command_navigation import COMMAND_OBSERVATION_CONTRACT, assert_command_policy
from .command_motor_limits import (HIP_YAW_INDICES, audit_command_policy_copy,
    bounded_motor_actions, initialize_bounded_command_policy)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_only_two_yaw_actions_change_and_source_eighteen_values_keep_exact_bits(dtype):
    action = np.linspace(-.91, .87, 20, dtype=dtype)
    action[list(HIP_YAW_INDICES)] = [-.8, .7]
    original = action.copy()
    result = bounded_motor_actions(action, .15)
    assert result.dtype == action.dtype and not np.shares_memory(result, action)
    assert np.array_equal(action, original)
    other = [index for index in range(20) if index not in HIP_YAW_INDICES]
    assert result[other].tobytes() == action[other].tobytes()
    assert result[list(HIP_YAW_INDICES)] == pytest.approx([-.15, .15])


def test_tensor_input_autograd_and_network_parameters_are_untouched():
    action = torch.linspace(-1., 1., 20, requires_grad=True)
    original = action.detach().clone()
    bounded_motor_actions(action, .15)
    assert torch.equal(action.detach(), original) and action.requires_grad and action.grad is None


@pytest.mark.parametrize("limit", [0., -1., 1.01, float("nan"), float("inf"), True])
def test_invalid_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        bounded_motor_actions(np.zeros(20), limit)


@pytest.mark.parametrize("action", [np.zeros(19), np.zeros((1, 20)),
    np.full(20, np.nan), np.full(20, np.inf), ["bad"]*20])
def test_invalid_action_vectors_are_rejected(action):
    with pytest.raises(ValueError, match="twenty finite"):
        bounded_motor_actions(action, .15)


class ContractEnv(gym.Env):
    def __init__(self, *, limit=None, body="same-body"):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (112,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.avatar_profile = {"avatarId": "same-avatar"}
        self.body_model_sha256 = body
        self.motor_checkpoint_sha256 = "same-motor"
        self.control_contract = {"version": 1, "motorCheckpointSha256": "same-motor",
            "rewardDiscountPerDecision": .99, "decisionControlSteps": 5, "maxLookaheadMeters": .5}
        if limit is not None:
            self.control_contract.update(version=2, motorHipYawResidualActionLimit=limit)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(112, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(112, dtype=np.float32), 0., False, False, {}


@pytest.fixture
def source():
    torch.set_num_threads(1)
    env = ContractEnv()
    policy = PPO("MlpPolicy", env, seed=172, n_steps=8, batch_size=8,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}}, device="cpu")
    bind_policy_body(policy, env)
    policy.command_observation_contract = copy.deepcopy(COMMAND_OBSERVATION_CONTRACT)
    policy.command_control_contract = copy.deepcopy(env.control_contract)
    policy.command_physical_control_timesteps = 12340
    policy.num_timesteps = 2468
    policy.policy.optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in policy.policy.parameters()).backward()
    policy.policy.optimizer.step()
    return policy


def transfer(source, target=None):
    return initialize_bounded_command_policy(source, target or ContractEnv(limit=.15),
        source_env=ContractEnv(), seed=192, n_steps=8, batch_size=8)


def test_real_network_transfer_keeps_every_tensor_and_both_counters_without_training(source):
    before = {name: value.clone() for name, value in source.policy.state_dict().items()}
    target, audit = transfer(source)
    assert source.policy.optimizer.state and not target.policy.optimizer.state
    for name, value in before.items():
        assert torch.equal(value, source.policy.state_dict()[name])
        assert torch.equal(value, target.policy.state_dict()[name])
    assert target.num_timesteps == source.num_timesteps == 2468
    assert target.command_physical_control_timesteps == source.command_physical_control_timesteps == 12340
    assert audit["additionalTrainingTimesteps"] == 0 and audit["verified"]
    assert audit["allActorCriticAndLogStdTensorsExactlyEqual"]


def test_saved_transferred_checkpoint_enforces_new_contract_and_has_empty_optimizer(source, tmp_path):
    env = ContractEnv(limit=.15)
    target, _ = transfer(source, env)
    target.save(tmp_path/"bounded")
    loaded = PPO.load(tmp_path/"bounded.zip", device="cpu")
    assert_command_policy(loaded, env)
    assert not loaded.policy.optimizer.state
    with pytest.raises(ValueError, match="motor or control parameters"):
        assert_command_policy(loaded, ContractEnv())


@pytest.mark.parametrize("changes", [{"maxLookaheadMeters": .4}, {"motorCheckpointSha256": "other"},
    {"decisionControlSteps": 3}, {"rewardDiscountPerDecision": .98}])
def test_no_other_control_contract_change_is_silently_allowed(source, changes):
    env = ContractEnv(limit=.15)
    env.control_contract.update(changes)
    with pytest.raises(ValueError, match="only version"):
        transfer(source, env)


def test_target_body_or_observation_bounds_change_rejects_transfer(source):
    with pytest.raises(ValueError, match="body contract"):
        transfer(source, ContractEnv(limit=.15, body="other"))
    env = ContractEnv(limit=.15)
    env.observation_space = gym.spaces.Box(-2., 2., (112,), np.float32)
    with pytest.raises(ValueError, match="spaces exactly"):
        transfer(source, env)


def test_input_contract_or_missing_physical_counter_rejects_transfer(source):
    source.command_observation_contract = {"version": 0}
    with pytest.raises(ValueError, match="observation semantics"):
        transfer(source)
    source.command_observation_contract = copy.deepcopy(COMMAND_OBSERVATION_CONTRACT)
    del source.command_physical_control_timesteps
    with pytest.raises(ValueError, match="physical-control"):
        transfer(source)


def test_copy_audit_rejects_even_one_actor_parameter_change(source):
    target, _ = transfer(source)
    with torch.no_grad():
        target.policy.mlp_extractor.policy_net[0].weight[0, 0] += .01
    with pytest.raises(ValueError, match="tensor"):
        audit_command_policy_copy(source, target)
