import math
from pathlib import Path

import numpy as np
import pytest

from .motor_cycle_tracking_training import CycleYawTracker


def test_one_gait_cycle_oscillation_does_not_become_false_turn_error():
    tracker=CycleYawTracker(1.12)
    heading=0.
    for index in range(56):
        next_heading=.1*math.sin(2.*math.pi*(index+1)/56.)+.135*(index+1)*.02
        result=tracker.update(.02,next_heading-heading,.135)
        heading=next_heading
    assert np.isclose(result['actualMeanYawRateRadiansPerSecond'],.135)
    assert np.isclose(result['consumedTargetMeanYawRateRadiansPerSecond'],.135)
    assert abs(result['boundedPenaltyRaw'])<1e-10


def test_wrong_way_persistent_turn_keeps_penalty_despite_gait_oscillation():
    tracker=CycleYawTracker(1.12)
    previous=0.
    for index in range(56):
        heading=.1*math.sin(2.*math.pi*(index+1)/56.)-.135*(index+1)*.02
        result=tracker.update(.02,heading-previous,.135)
        previous=heading
    assert result['boundedPenaltyRaw']< -3.
    assert result['boundedPenaltyRaw']>= -4.


def test_partial_oldest_interval_is_trimmed_exactly_without_future_samples():
    tracker=CycleYawTracker(.03)
    tracker.update(.02,.002,.1)
    result=tracker.update(.02,.006,.3)
    assert np.isclose(result['coveredSeconds'],.03)
    assert np.isclose(result['actualMeanYawRateRadiansPerSecond'],(.001+.006)/.03)
    assert np.isclose(result['consumedTargetMeanYawRateRadiansPerSecond'],(.001+.006)/.03)
    assert abs(result['boundedPenaltyRaw'])<1e-10


def test_startup_penalty_is_coverage_bounded_and_rejects_nonphysical_inputs():
    tracker=CycleYawTracker(1.)
    result=tracker.update(.02,0.,.135)
    assert -.08<=result['boundedPenaltyRaw']<=0.
    with pytest.raises(ValueError):
        tracker.update(0.,0.,0.)
    with pytest.raises(ValueError):
        tracker.update(.02,float('nan'),0.)


def test_actual_physics_reward_preserves_v3_terms_and_reports_v5_truthfully():
    import torch
    from stable_baselines3 import PPO
    from .command_motor import command_tracking_reward
    from .motor_cycle_tracking_training import MotorCycleTrackingEnv,MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
    root=Path(__file__).resolve().parents[1]
    checkpoint=root/'.runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip'
    if not checkpoint.exists():
        pytest.skip('immutable source unavailable')
    torch.set_num_threads(1)
    env=MotorCycleTrackingEnv(base_seed=96101,avatar_profile=root/'frontend/public/world-model/gakucho-avatar-profile.json')
    policy=PPO.load(checkpoint,device='cpu')
    try:
        observation,_=env.reset(seed=96201,options={'scene_seed':96101,'physics_seed':96201})
        body=env.unwrapped;identity=body.goal;value=body.goal.copy()
        for _ in range(3):
            observation,reward,done,truncated,info=env.step(policy.predict(observation,deterministic=True)[0])
            cm=info['command_motor'];tracking=cm['trackingReward']
            baseline=command_tracking_reward(tracking['measurements'])
            assert all(tracking['components'][key]==val for key,val in baseline['components'].items())
            assert cm['rewardContract']==MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
            assert cm['rewardVersion']==5
            assert cm['observedPhysicsSubstepsThisControl']==10
            assert body.goal is identity and np.array_equal(body.goal,value)
            assert not np.any(body.data.qfrc_applied) and not np.any(body.data.xfrc_applied)
            assert np.isfinite(reward) and not done and not truncated
        assert np.isclose(tracking['cycleYawTracking']['coveredSeconds'],.06)
        assert np.isclose(body.data.time,.06)
        env.reset(seed=96201,options={'scene_seed':96101,'physics_seed':96201})
        assert len(env._cycle_tracker.intervals)==0
    finally:
        env.close()
