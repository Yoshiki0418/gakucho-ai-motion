"""Resume serialization tests on CPU; these provide no CUDA training evidence."""
from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from .command_motor_training import _state_sha256
from .visual_policy import visual_policy_kwargs
from .visual_training import resume_visual_policy


class _UnsteppedEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        'depth': gym.spaces.Box(0., 1., (3, 48, 64), dtype=np.float32),
        'state': gym.spaces.Box(-5., 5., (11,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)
    observation_contract = {'testOnlySemantics': 'exactly preserved'}
    control_contract = {'testOnlyFrozenMotor': 'unchanged'}
    avatar_profile = None
    body_model_sha256 = None

    def reset(self, **kwargs):
        raise AssertionError('checkpoint resume must not reset or step physics')

    def step(self, action):
        raise AssertionError('checkpoint resume must not reset or step physics')


def _checkpoint(tmp_path):
    torch.set_num_threads(1)
    env = _UnsteppedEnv()
    model = PPO('MultiInputPolicy', env, device='cpu', seed=98711,
                n_steps=2, batch_size=2, n_epochs=1, gamma=.99,
                policy_kwargs=visual_policy_kwargs())
    model.visual_observation_contract = copy.deepcopy(env.observation_contract)
    model.visual_control_contract = copy.deepcopy(env.control_contract)
    model.avatar_profile_sha256 = None
    model.body_model_sha256 = None
    # Actual Adam moments, including critic and std, rather than empty state.
    model.policy.optimizer.zero_grad(set_to_none=True)
    sum(parameter.square().mean() for parameter in model.policy.parameters()).backward()
    model.policy.optimizer.step()
    model.num_timesteps, model._n_updates = 16, 3
    model.visual_physical_control_timesteps = 77
    model.visual_physics_substeps = 770
    path = tmp_path/'checkpoint.zip'
    model.save(path)
    return path, model, env


def _cpu_load_only(monkeypatch):
    original = PPO.load
    calls = []

    def cpu_load(*args, **kwargs):
        calls.append(kwargs.get('device'))
        # Explicit CPU unit-test adaptation, never passed off as CUDA evidence.
        kwargs['device'] = 'cpu'
        return original(*args, **kwargs)

    monkeypatch.setattr(PPO, 'load', cpu_load)
    return calls


def test_resume_preserves_nonzero_actor_critic_std_adam_and_native_counts(tmp_path, monkeypatch):
    path, source, env = _checkpoint(tmp_path)
    calls = _cpu_load_only(monkeypatch)
    try:
        resumed, audit = resume_visual_policy(path, source.env, body_env=env, seed=99711)
        assert calls == ['cpu', 'cuda']
        assert resumed.device.type == 'cpu'
        assert _state_sha256(resumed.policy.state_dict()) == _state_sha256(source.policy.state_dict())
        assert _state_sha256(resumed.policy.optimizer.state_dict()) == _state_sha256(source.policy.optimizer.state_dict())
        assert resumed.policy.optimizer.state
        assert torch.count_nonzero(resumed.policy.value_net.weight) > 0
        assert (resumed.num_timesteps, resumed._n_updates,
                resumed.visual_physical_control_timesteps, resumed.visual_physics_substeps) == (16, 3, 77, 770)
        assert audit['sourcePpoDecisions'] == 16
        assert audit['sourcePpoUpdateEpochs'] == 3
        assert audit['sourcePhysicalControls'] == 77
        assert audit['sourcePhysicsSubsteps'] == 770
        assert audit['actorCriticLogStdAndAdamPreservedExactly']
        assert not audit['criticReset'] and not audit['freshAdam']
        assert not audit['teacherFittingRepeated']
    finally:
        source.env.close()


@pytest.mark.parametrize('attribute,value', [
    ('visual_physical_control_timesteps', -1),
    ('visual_physical_control_timesteps', True),
    ('visual_physics_substeps', 769),
    ('visual_physics_substeps', 770.),
    ('num_timesteps', -1),
    ('_n_updates', -1),
])
def test_resume_rejects_corrupt_native_accounting_before_gpu_load(tmp_path, monkeypatch, attribute, value):
    path, source, env = _checkpoint(tmp_path)
    setattr(source, attribute, value)
    source.save(path)
    calls = _cpu_load_only(monkeypatch)
    try:
        with pytest.raises(ValueError, match='exact native'):
            resume_visual_policy(path, source.env, body_env=env, seed=99711)
        assert calls == ['cpu']
    finally:
        source.env.close()


def test_real_camera_contract_survives_sb3_json_roundtrip_and_accepts_r1_checkpoint(tmp_path):
    """Real pinhole/body contracts must work after metadata JSON serialization."""
    from .avatar_profile import load_avatar_profile
    from .body_contract import bind_policy_body
    from .visual_navigation import assert_visual_policy
    from .visual_training import make_env

    root = Path(__file__).resolve().parents[1]
    motor = root/'.runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip'
    r1 = root/'.runtime/world-model/vision-navigation/cuda-depth-ppo-r1/before-ppo-policy.zip'
    if not motor.exists() or not r1.exists():
        pytest.skip('exact frozen motor and pre-interruption visual checkpoint are not installed')
    torch.set_num_threads(1)
    env = make_env(motor, load_avatar_profile(root/'frontend/public/world-model/gakucho-avatar-profile.json'),
                   90801, 1., (0., .5))
    try:
        before = PPO.load(r1, device='cpu')
        assert_visual_policy(before, env)
        assert type(env.control_contract['depthCameraConfig']['mount_offset_local']) is list
        model = PPO('MultiInputPolicy', env, device='cpu', seed=98711,
                    n_steps=2, batch_size=2, n_epochs=1, gamma=.99,
                    policy_kwargs=visual_policy_kwargs())
        bind_policy_body(model, env.unwrapped)
        model.visual_observation_contract = copy.deepcopy(env.observation_contract)
        model.visual_control_contract = copy.deepcopy(env.control_contract)
        assert_visual_policy(model, env)
        checkpoint = tmp_path/'real-contract-roundtrip.zip'
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, device='cpu')
        assert loaded.visual_control_contract == env.control_contract
        assert loaded.visual_observation_contract == env.observation_contract
        assert_visual_policy(loaded, env)
    finally:
        env.close()
