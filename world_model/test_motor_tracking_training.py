import copy
from pathlib import Path

import numpy as np
import pytest

from .command_motor import command_tracking_reward,observed_command_angle
from .motor_tracking_training import (MOTOR_TRACKING_REWARD_CONTRACT,MotorTrackingEnv,
                                     assert_motor_tracking_policy,tracking_reward_v4)


def _saved_measurement():
    import json
    probe=Path(__file__).resolve().parents[1]/'.runtime/world-model/vision-navigation/cuda-depth-conservative-r4/after/stage2-visual-obstacle-1180100-failed.json'
    if not probe.exists():
        pytest.skip('physical recording not available')
    frames=json.loads(probe.read_text())['frames']
    return next(frame['stats']['command_motor']['trackingReward']['measurements'] for frame in frames[1:])


def test_reward_changes_only_measured_yaw_term_and_retains_scale_free_dense_floor():
    measurement=_saved_measurement()
    old=command_tracking_reward(measurement)
    new=tracking_reward_v4(measurement)
    for term in old['components']:
        if term!='commandYawRate':
            assert new['components'][term]==old['components'][term]
    assert -4.<=new['components']['commandYawRate']<=0.
    assert new['components']['commandYawRate']<=old['components']['commandYawRate']
    assert new['denseRawReward']==max(-8.,sum(new['components'].values()))
    assert measurement==_saved_measurement()


def test_perfect_yaw_tracking_has_no_penalty():
    measurement=_saved_measurement()
    measurement['rootHeadingRadians']=measurement['rootPreHeadingRadians']
    measurement['commandBearingRadians']=measurement['rootPreHeadingRadians']
    result=tracking_reward_v4(measurement)
    assert result['components']['commandYawRate']==0.


def test_stop_curriculum_never_teleports_or_changes_final_goal_and_is_observable():
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'.runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip'
    if not checkpoint.exists():
        pytest.skip('immutable local motor not available')
    env=MotorTrackingEnv(base_seed=94101,stop_probability=1.,avatar_profile=root/'frontend/public/world-model/gakucho-avatar-profile.json')
    policy=PPO.load(checkpoint,device='cpu')
    try:
        observation,_=env.reset(seed=94201,options={'scene_seed':94101,'physics_seed':94201})
        body=env.unwrapped;identity=body.goal;value=body.goal.copy()
        seen_stop=False;controls=0
        for _ in range(245):
            action=policy.predict(observation,deterministic=True)[0]
            observation,reward,done,truncated,info=env.step(action)
            controls+=1
            assert body.goal is identity and np.array_equal(body.goal,value)
            assert np.all(body.data.xfrc_applied==0.) and np.all(body.data.qfrc_applied==0.)
            assert info['command_motor']['observedPhysicsSubstepsThisControl']==10
            assert info['command_motor']['rewardVersion']==4
            assert np.isfinite(reward)
            if env._stop_active:
                seen_stop=True
                assert observed_command_angle(observation)[1]<1e-5
            if done or truncated:
                break
        assert seen_stop
        assert np.isclose(body.data.time,controls*.02)
        assert env.reward_control_contract['rewardContract']==MOTOR_TRACKING_REWARD_CONTRACT
        assert env.control_contract['rewardContract']['version']==3
        with pytest.raises(ValueError,match='reward-v4'):
            assert_motor_tracking_policy(policy,env)
    finally:
        env.close()
