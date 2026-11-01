from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from .motor_expert_mixture import ExpertMixtureVisualNavigationEnv,right_expert_weight


def _observation(angle,distance=.4):
    observation=np.zeros(110,dtype=np.float32)
    observation[41:44]=[0.,0.,1.]
    observation[50:53]=[distance*np.sin(angle),-distance*np.cos(angle),-1.]
    return observation


def test_gate_is_smooth_bounded_local_command_only_and_zero_on_stop():
    assert right_expert_weight(_observation(-.45))==0.
    assert right_expert_weight(_observation(0.))==0.
    assert np.isclose(right_expert_weight(_observation(.1)),.5)
    assert right_expert_weight(_observation(.45))==1.
    assert right_expert_weight(_observation(.45,0.))==0.
    assert right_expert_weight(_observation(1e-4))<1e-5
    with pytest.raises(ValueError):
        right_expert_weight(_observation(.1),0.)


def test_gate_preserves_exact_expert_endpoints_and_bounded_convex_output():
    env=object.__new__(ExpertMixtureVisualNavigationEnv)
    env._mixture_ready=False
    env.full_weight_heading_radians=.2
    left=np.full(20,-1.,dtype=np.float32);right=np.ones(20,dtype=np.float32)
    env._source_motor=SimpleNamespace(predict=lambda *a,**k:(left.copy(),None))
    env._right_predict=lambda *a,**k:(right.copy(),None)
    np.testing.assert_array_equal(env._blend_predict(_observation(-.2),deterministic=True)[0],left)
    np.testing.assert_array_equal(env._blend_predict(_observation(.3),deterministic=True)[0],right)
    blended=env._blend_predict(_observation(.1),deterministic=True)[0]
    assert np.max(np.abs(blended))<1e-5
    with pytest.raises(ValueError):
        env._blend_predict(_observation(.1),deterministic=False)


def test_actual_mixture_uses_unchanged_physics_and_freezes_both_experts():
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'.runtime/world-model/vision-navigation/motor-cycle-tracking-r5/checkpoint-40960.zip'
    if not checkpoint.exists():
        pytest.skip('qualified diagnostic checkpoint unavailable')
    env=ExpertMixtureVisualNavigationEnv(motor_checkpoint=checkpoint,mode='turn',split='train',base_seed=90301,
        difficulty_range=(0.,0.),randomize=False,max_episode_seconds=8.,record_trajectory=True)
    try:
        observation,_=env.reset(seed=91301,options={'scene_seed':90301,'physics_seed':91301})
        body=env.unwrapped;identity=body.goal;value=body.goal.copy()
        _,_,done,truncated,info=env.step(np.asarray([1.,-.6],dtype=np.float32))
        assert not done and not truncated
        assert info['command_control_steps']==5 and info['command_physics_substeps']==50
        assert len(env.last_control_frames)==5
        assert body.goal is identity and np.array_equal(body.goal,value)
        assert not np.any(body.data.qfrc_applied) and not np.any(body.data.xfrc_applied)
        assert env.control_contract['expertMixture']['extraTrainingControls']==0
        assert 0.<=env._last_expert_weight<=1.
        env.assert_frozen_motor(check_checkpoint=True)
        saved_gate=env.full_weight_heading_radians
        env.full_weight_heading_radians=.1
        with pytest.raises(RuntimeError,match='gate contract'):
            env._blend_predict(env.motor_env._observation(),deterministic=True)
        env.full_weight_heading_radians=saved_gate
        saved_count=env._source_motor.num_timesteps
        env._source_motor.num_timesteps=saved_count+.5
        with pytest.raises(RuntimeError,match='counters'):
            env.assert_frozen_motor()
        env._source_motor.num_timesteps=saved_count
        env.assert_frozen_motor(check_checkpoint=True)
    finally:
        env.close()
