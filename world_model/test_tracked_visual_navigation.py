"""CPU transfer/physics checks for v4/v5 adapters; fixtures add zero training."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .command_motor_training import _state_sha256
from .motor_tracking_training import MOTOR_TRACKING_REWARD_CONTRACT
from .tracked_visual_navigation import (TrackedVisualNavigationEnv, _tracking_validation, make_visual_env,
                                       transfer_visual_policy)
from .visual_navigation import (FROZEN_MOTOR_CHECKPOINT_SHA256,
                                VisualNavigationEnv, assert_visual_policy)


_ROOT=Path(__file__).resolve().parents[1]
_MOTOR=_ROOT/'.runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip'
_HIGH=_ROOT/'.runtime/world-model/vision-navigation/cuda-depth-conservative-r4/policy.zip'
_PROFILE=_ROOT/'frontend/public/world-model/gakucho-avatar-profile.json'


@pytest.fixture(scope='module',params=(4,5),ids=('instant-v4','cycle-v5'))
def transferred_motor(tmp_path_factory,request):
    if not _MOTOR.exists() or not _HIGH.exists():
        pytest.skip('local immutable checkpoints unavailable')
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model=PPO.load(_MOTOR,device='cpu')
    version=request.param
    contract=MOTOR_TRACKING_REWARD_CONTRACT
    if version==5:
        from .motor_cycle_tracking_training import MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
        contract=MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
        model.motor_cycle_tracking_reward_contract=copy.deepcopy(contract)
    model.command_motor_reward_contract=copy.deepcopy(contract)
    model.motor_tracking_reward_contract=copy.deepcopy(contract)
    model.command_motor_control_contract['rewardContract']=copy.deepcopy(contract)
    model.motor_tracking_initial_physical_controls=model.num_timesteps
    model.motor_tracking_transfer_audit={
        'sourceCheckpointSha256':FROZEN_MOTOR_CHECKPOINT_SHA256,'sourceRewardVersion':3,'targetRewardVersion':version,
        'sourceV3BodyObservationActionRewardAndControlContractsValidated':True,
        'allActorCriticLogStdTensorsPreservedExactly':True,'allAdamStatePreservedExactly':True,
        'initialActualPhysicalControls':model.num_timesteps}
    directory=tmp_path_factory.mktemp('tracked-visual')
    checkpoint=directory/f'v{version}-transfer-zero-added-training.zip'
    model.save(checkpoint)
    return checkpoint,version


def _kwargs():
    return {'mode':'turn','split':'train','base_seed':94701,'randomize':False,
        'difficulty_range':(0.,0.),'max_episode_seconds':1.,'avatar_profile':_PROFILE,'record_trajectory':True}


def test_original_rejects_v4_and_explicit_adapter_preserves_real_input_and_stepping(transferred_motor):
    checkpoint,version=transferred_motor
    with pytest.raises(ValueError,match='immutable trained 5aa'):
        VisualNavigationEnv(motor_checkpoint=checkpoint,**_kwargs())
    old=make_visual_env(motor_checkpoint=_MOTOR,**_kwargs())
    new=make_visual_env(motor_checkpoint=checkpoint,**_kwargs())
    try:
        assert type(old) is VisualNavigationEnv and isinstance(new,TrackedVisualNavigationEnv)
        options={'scene_seed':94701,'physics_seed':95701}
        original,_=old.reset(seed=95701,options=options)
        adapted,_=new.reset(seed=95701,options=options)
        for key in original:
            np.testing.assert_array_equal(original[key],adapted[key])
        body=new.unwrapped; goal_object,goal=body.goal,body.goal.copy()
        tensor_sha=_state_sha256(new._motor.policy.state_dict())
        obs,reward,done,truncated,info=new.step(np.asarray([.1,-.6],np.float32))
        assert info['command_control_steps']==5 and info['command_physics_substeps']==50
        assert len(new.last_control_frames)==5 and np.isclose(body.data.time,.1)
        assert body.goal is goal_object and np.array_equal(body.goal,goal)
        assert not np.any(body.data.qfrc_applied) and not np.any(body.data.xfrc_applied)
        assert _state_sha256(new._motor.policy.state_dict())==tensor_sha
        new.assert_frozen_motor(check_checkpoint=True)
        assert new.control_contract['sourceMotorTrainingRewardContract']['version']==version
        assert new.control_contract['sourceMotorTrainingRewardVersion']==version
        assert 'tracking-v4' not in new.control_contract['role']
        assert new.control_contract['inferenceLowIntrinsicRewardContract']['version']==3
        assert not new.control_contract['inferenceLowIntrinsicRewardUsedByHighController']
        assert new._last_reward_summary['lowIntrinsicTrackingRewardUsed'] is False
    finally:
        old.close();new.close()


def test_high_transfer_preserves_actor_critic_std_adam_native_counts_and_actual_action(transferred_motor,tmp_path):
    from stable_baselines3 import PPO
    checkpoint,version=transferred_motor
    env=TrackedVisualNavigationEnv(motor_checkpoint=checkpoint,**_kwargs())
    try:
        observation,_=env.reset(seed=95701,options={'scene_seed':94701,'physics_seed':95701})
        source=PPO.load(_HIGH,device='cpu')
        with pytest.raises(ValueError,match='control contract differs'):
            assert_visual_policy(source,env)
        model,audit=transfer_visual_policy(_HIGH,env,output=tmp_path/'high.zip',actual_observation=observation)
        assert _state_sha256(source.policy.state_dict())==_state_sha256(model.policy.state_dict())
        assert _state_sha256(source.policy.optimizer.state_dict())==_state_sha256(model.policy.optimizer.state_dict())
        np.testing.assert_array_equal(source.predict(observation,deterministic=True)[0],model.predict(observation,deterministic=True)[0])
        assert audit['actorCriticLogStdAdamAndAllNativeCountsPreservedExactly']
        assert audit['actualObservationActionPreservedExactly'] and audit['ppoDecisionsAdded']==0
        loaded=PPO.load(tmp_path/'high.zip',device='cpu')
        assert_visual_policy(loaded,env)
        assert loaded.visual_motor_transfer_audit==audit
        with pytest.raises(ValueError,match='preserve prior policy'):
            transfer_visual_policy(_HIGH,env,output=tmp_path/'high.zip')
    finally:
        env.close()


def test_frozen_adapter_rejects_provenance_tampering(transferred_motor):
    checkpoint,version=transferred_motor
    env=TrackedVisualNavigationEnv(motor_checkpoint=checkpoint,**_kwargs())
    try:
        env._motor.motor_tracking_transfer_audit['initialActualPhysicalControls']+=1
        with pytest.raises(RuntimeError,match='provenance'):
            env.assert_frozen_motor()
    finally:
        env.close()


def test_v5_cycle_specific_contract_is_frozen(transferred_motor):
    checkpoint,version=transferred_motor
    if version!=5:
        pytest.skip('cycle-specific metadata belongs to v5')
    env=TrackedVisualNavigationEnv(motor_checkpoint=checkpoint,**_kwargs())
    try:
        env._motor.motor_cycle_tracking_reward_contract['cycleHistory']='changed'
        with pytest.raises(RuntimeError,match='provenance'):
            env.assert_frozen_motor()
    finally:
        env.close()


def test_v3_cannot_enter_explicit_v4_adapter():
    if not _MOTOR.exists():
        pytest.skip('local checkpoint unavailable')
    with pytest.raises(ValueError,match='unchanged 5aa'):
        TrackedVisualNavigationEnv(motor_checkpoint=_MOTOR,**_kwargs())


@pytest.mark.parametrize('reward',[{'version':4},{'version':5},{'version':6}])
def test_declared_version_cannot_replace_exact_registered_training_contract(reward):
    with pytest.raises(ValueError,match='exact registered v4 or v5'):
        _tracking_validation(SimpleNamespace(command_motor_reward_contract=reward),None)
