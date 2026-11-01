"""Training safeguards and real supervised optimizer behavior on synthetic data.

These CPU tests perform no simulator controls or PPO learning. CUDA execution
is separately proven by the actual training run, rather than mocked here.
"""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from . import visual_training as training
from .visual_policy import visual_policy_kwargs


def _arguments(tmp_path):
    return training.argument_parser().parse_args([
        '--motor-policy', str(tmp_path/'missing-motor.zip'),
        '--avatar-profile', str(tmp_path/'missing-profile.json'),
        '--output', str(tmp_path/'new-run'),
    ])


def test_cuda_unavailable_is_rejected_before_checkpoint_or_output_io(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='actual CUDA'):
        training.train(_arguments(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('changes', [
    {'envs': 0}, {'envs': 33}, {'steps': 0}, {'bc_episodes': -1},
    {'seconds': float('nan')}, {'seconds': 0.}, {'bc_epochs': 0},
    {'difficulty_min': -.1}, {'difficulty_max': 1.1},
    {'difficulty_min': .5, 'difficulty_max': .2},
    {'n_steps': 1}, {'batch_size': 255}, {'checkpoint_every': 0},
    {'seed': 1_000_000}, {'seed': -1}, {'eval_seed': 88711},
    {'eval_seed': 1_800_000}, {'eval_seed': 1_799_999, 'eval_episodes': 2},
])
def test_invalid_training_domain_and_counts_fail_before_world_or_output_io(tmp_path, monkeypatch, changes):
    # Validation is CPU-testable without pretending to run a CUDA operation.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    args = _arguments(tmp_path)
    for name, value in changes.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        training.train(args)
    assert list(tmp_path.iterdir()) == []


def test_completed_or_started_run_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    args = _arguments(tmp_path)
    output = tmp_path/'new-run'
    output.mkdir()
    checkpoint = output/'policy.zip'
    checkpoint.write_bytes(b'completed-checkpoint-must-survive')
    with pytest.raises(ValueError, match='fresh output'):
        training.train(args)
    assert checkpoint.read_bytes() == b'completed-checkpoint-must-survive'
    assert sorted(path.name for path in output.iterdir()) == ['policy.zip']


class _SpaceOnlyEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        'depth': gym.spaces.Box(0., 1., (3, 48, 64), dtype=np.float32),
        'state': gym.spaces.Box(-5., 5., (11,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)

    def reset(self, **kwargs):
        raise AssertionError('supervised fitting must not reset or advance any world')

    def step(self, action):
        raise AssertionError('supervised fitting must not execute physical controls')


def test_supervision_updates_actor_cnn_and_resets_ppo_adam_without_ppo_counts():
    torch.set_num_threads(1)
    model = PPO('MultiInputPolicy', _SpaceOnlyEnv(), device='cpu', seed=88711,
                n_steps=2, batch_size=2, n_epochs=1,
                policy_kwargs=visual_policy_kwargs())
    try:
        # Give the initial PPO optimizer real state, without invoking learn().
        model.policy.optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().mean() for parameter in model.policy.parameters()).backward()
        model.policy.optimizer.step()
        previous_optimizer = model.policy.optimizer
        assert previous_optimizer.state
        model.num_timesteps, model._n_updates = 17, 9
        before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        rng = np.random.default_rng(89711)
        observations = {'depth': rng.random((8, 3, 48, 64), dtype=np.float32),
                        'state': rng.normal(size=(8, 11)).astype(np.float32)}
        actions = np.asarray([[.8, -.6]]*8, dtype=np.float32)
        original_observations, original_actions = copy.deepcopy(observations), actions.copy()
        report = training.fit_teacher(model, observations, actions, epochs=3, batch_size=4, seed=88711)
        after = model.policy.state_dict()
        assert any(not torch.equal(before[key], after[key]) for key in before if 'depth_encoder' in key)
        assert not torch.equal(before['action_net.weight'], after['action_net.weight'])
        for key in before:
            if key.startswith('mlp_extractor.value_net.'):
                assert torch.equal(before[key], after[key])
        assert torch.count_nonzero(model.policy.value_net.weight) == 0
        assert torch.count_nonzero(model.policy.value_net.bias) == 0
        torch.testing.assert_close(model.policy.log_std, torch.tensor([-1.5, -3.]))
        assert model.policy.optimizer is not previous_optimizer
        assert not model.policy.optimizer.state
        assert (model.num_timesteps, model._n_updates) == (17, 9)
        assert report['examples'] == 8 and report['optimizerSteps'] == 6
        assert report['ppoDecisionsAdded'] == 0
        assert report['ppoAdamResetAfterSupervision'] is True
        assert report['geometryPrivilegedLabelsTrainOnly'] is True
        assert report['labelsAreNotClaimsOfPhysicallyVerifiedTeacherSuccess'] is True
        assert report['device'] == 'cpu'
        assert np.isfinite(report['lossMse']).all()
        for key in observations:
            np.testing.assert_array_equal(observations[key], original_observations[key])
        np.testing.assert_array_equal(actions, original_actions)
    finally:
        model.env.close()
