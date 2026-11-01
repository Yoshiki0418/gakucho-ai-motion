"""CPU checks for conservative transfer; these are not CUDA navigation evidence."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from .command_motor_training import _state_sha256
from .visual_policy import visual_policy_kwargs
from .visual_refinement import configure_conservative_refinement


class _TestEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "depth": gym.spaces.Box(0.,1.,(3,48,64),dtype=np.float32),
        "state": gym.spaces.Box(-5.,5.,(11,),dtype=np.float32)})
    action_space = gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)


def _model():
    torch.set_num_threads(1)
    model = PPO("MultiInputPolicy", _TestEnv(), device="cpu", seed=77201,
        n_steps=2,batch_size=2,n_epochs=1,policy_kwargs=visual_policy_kwargs())
    # Establish actual nonempty Adam moments before checking their transfer.
    sum(p.square().mean() for p in model.policy.parameters()).backward()
    model.policy.optimizer.step()
    model.policy.optimizer.zero_grad(set_to_none=True)
    model.num_timesteps, model._n_updates = 16, 3
    model.visual_physical_control_timesteps, model.visual_physics_substeps = 77, 770
    return model


def test_refinement_preserves_all_weights_adam_and_native_counts():
    model = _model()
    weights, adam = _state_sha256(model.policy.state_dict()), _state_sha256(model.policy.optimizer.state_dict())
    audit = configure_conservative_refinement(model)
    assert _state_sha256(model.policy.state_dict()) == weights
    assert _state_sha256(model.policy.optimizer.state_dict()) == adam
    assert (model.num_timesteps,model._n_updates,model.visual_physical_control_timesteps,
            model.visual_physics_substeps) == (16,3,77,770)
    assert all(not p.requires_grad for p in model.policy.features_extractor.parameters())
    assert not model.policy.log_std.requires_grad
    assert audit["weightsPreservedExactly"] and audit["adamMomentsPreservedExactly"]
    assert not audit["optimizerReset"] and not audit["criticReset"]
    assert model.lr_schedule(.5) == 1e-5 and model.target_kl == .01
    assert model.policy.action_net.weight.requires_grad
    assert model.policy.value_net.weight.requires_grad


def test_actual_optimizer_step_changes_actor_and_critic_but_preserves_pixels_and_noise():
    model = _model()
    audit = configure_conservative_refinement(model)
    before = {k:v.clone() for k,v in model.policy.state_dict().items()}
    obs = {"depth":torch.rand(4,3,48,64), "state":torch.randn(4,11)}
    model.policy.optimizer.zero_grad(set_to_none=True)
    loss = (model.policy.get_distribution(obs).distribution.mean-.7).square().mean()
    loss += .5*(model.policy.predict_values(obs)+20.).square().mean()
    loss.backward()
    assert model.policy.action_net.weight.grad.abs().sum() > 0
    assert model.policy.value_net.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.policy.features_extractor.parameters())
    assert model.policy.log_std.grad is None
    model.policy.optimizer.step()
    for name in audit["frozenParameterNames"]:
        assert torch.equal(before[name],model.policy.state_dict()[name])
    assert not torch.equal(before["action_net.weight"],model.policy.action_net.weight)
    assert not torch.equal(before["value_net.weight"],model.policy.value_net.weight)


@pytest.mark.parametrize("kwargs",[
    {"learning_rate":float("nan")},{"learning_rate":0.},{"learning_rate":1e-4},
    {"target_kl":0.},{"target_kl":float("inf")},{"target_kl":.03}])
def test_bad_trust_region_configuration_does_not_mutate_policy(kwargs):
    model = _model()
    before = _state_sha256(model.policy.state_dict())
    with pytest.raises(ValueError,match="finite and bounded"):
        configure_conservative_refinement(model,**kwargs)
    assert _state_sha256(model.policy.state_dict()) == before
    assert all(p.requires_grad for p in model.policy.parameters())


@pytest.mark.parametrize("name,value",[("visual_physical_control_timesteps",True),
    ("visual_physics_substeps",769),("num_timesteps",-1),("_n_updates",3.)])
def test_corrupt_native_counts_rejected(name,value):
    model = _model()
    setattr(model,name,value)
    with pytest.raises(ValueError,match="exact native"):
        configure_conservative_refinement(model)
