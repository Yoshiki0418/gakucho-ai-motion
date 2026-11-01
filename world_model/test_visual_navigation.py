"""Check the visual interface and one real 100 ms frozen-motor transition."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from . import visual_navigation as visual
from .command_motor import CommandMotorEnv


_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = _ROOT/'frontend/public/world-model/gakucho-avatar-profile.json'
_MOTOR = _ROOT/'.runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip'
_MOTOR_SHA = '5aa9536b35defac5d715f98c75f558f6f5cdaf9a26344672d02e7e4d4358252b'
_PHYSICS = dict(mode='turn', split='train', base_seed=88711,
                difficulty_range=(.1,.1), randomize=False,
                max_episode_seconds=1., avatar_profile=_PROFILE)
_RESET = {'scene_seed':88711,'physics_seed':89711}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key],right[key])
    elif isinstance(left, (tuple,list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a,b in zip(left,right):
            _assert_nested_equal(a,b)
    else:
        assert left == right


@pytest.mark.parametrize('changes', [
    {'decision_steps':4}, {'decision_steps':6}, {'decision_steps':5.0},
    {'depth_stack':1}, {'depth_stack':4}, {'depth_stack':True}, {'depth_stack':2.0},
    {'max_heading_radians':0.}, {'max_heading_radians':.46}, {'max_heading_radians':float('nan')},
    {'max_lookahead_m':.34}, {'max_lookahead_m':.51}, {'max_lookahead_m':float('inf')},
    {'heading_slew_radians_per_second':0.}, {'heading_slew_radians_per_second':.5},
])
def test_invalid_visual_configuration_is_rejected_before_loading_a_checkpoint(tmp_path, changes):
    with pytest.raises(ValueError):
        visual.VisualNavigationEnv(motor_checkpoint=tmp_path/'missing-policy.zip', **changes, **_PHYSICS)


@pytest.fixture(scope='module')
def real_visual_smoke():
    if not _MOTOR.is_file():
        pytest.skip('the exact frozen 5aa integration checkpoint is not installed')
    assert _sha(_MOTOR) == _MOTOR_SHA
    torch.set_num_threads(1)
    patch = pytest.MonkeyPatch()
    captures = []
    scratch = np.zeros((1,48,64),dtype=np.float32)

    def capture(body, config):
        assert (config.height,config.width) == (48,64)
        captures.append(float(body.data.time))
        scratch.fill(len(captures)/10.)
        return scratch, {'physicalTimeSeconds':float(body.data.time),'captureIndex':len(captures),
                         'worldStateWritten':False,'pixelRayCount':48*64}

    patch.setattr(visual,'capture_depth',capture)
    env = visual.VisualNavigationEnv(motor_checkpoint=_MOTOR, record_trajectory=True, **_PHYSICS)
    try:
        assert type(env.motor_env).step is CommandMotorEnv.step
        assert type(env.motor_env).reset is CommandMotorEnv.reset
        first,first_info = env.reset(seed=89711,options=_RESET)
        assert set(first) == {'depth','state'}
        np.testing.assert_array_equal(first['depth'],np.full((3,48,64),.1,dtype=np.float32))
        second,second_info = env.reset(seed=89711,options=_RESET)
        np.testing.assert_array_equal(second['depth'],np.full((3,48,64),.2,dtype=np.float32))
        assert captures == [0.,0.]
        # A caller changing an observation must not change stack history.
        second['depth'][0].fill(.99)
        np.testing.assert_array_equal(second['depth'][1],np.full((48,64),.2,dtype=np.float32))
        np.testing.assert_array_equal(env.latest_depth,np.full((1,48,64),.2,dtype=np.float32))
        body = env.motor_env.unwrapped
        expected_final_goal = (body.data.xmat[body.pelvis_body_id].reshape(3,3).T
                               @(body.goal-body.data.qpos[:3])).astype(np.float32)
        np.testing.assert_array_equal(first['state'][:3],expected_final_goal)
        assert not np.allclose(first['state'][:3],env.motor_env._observation()[50:53])
        goal_object,goal_values = body.goal,body.goal.copy()
        weights = copy.deepcopy(env._motor.policy.state_dict())
        adam = copy.deepcopy(env._motor.policy.optimizer.state_dict())
        counts = (env._motor.num_timesteps,env._motor.command_motor_physical_control_timesteps,env._motor._n_updates)
        observed_substeps = []
        observer = body._physics_contact_observer

        def observe(model,data):
            assert body.goal is goal_object
            np.testing.assert_array_equal(body.goal,goal_values)
            assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
            observed_substeps.append(float(data.time))
            observer(model,data)

        body._physics_contact_observer = observe
        observation,reward,done,truncated,info = env.step(np.asarray([.65,.25],dtype=np.float32))
        assert not done and not truncated
        assert info['command_control_steps'] == 5
        assert info['command_physics_substeps'] == 50
        assert len(env.last_control_frames) == 5
        np.testing.assert_allclose(observed_substeps,np.arange(1,51)*.002,atol=1e-15,rtol=0)
        assert body.data.time == pytest.approx(.1)
        assert captures == [0.,0.,pytest.approx(.1)]
        np.testing.assert_array_equal(observation['depth'],np.stack([
            np.full((48,64),.2,dtype=np.float32),np.full((48,64),.2,dtype=np.float32),
            np.full((48,64),.3,dtype=np.float32)]))
        assert set(env.observation_space.spaces) == {'depth','state'}
        assert env.observation_space.contains(observation)
        assert observation['depth'].dtype == observation['state'].dtype == np.float32
        assert observation['state'].shape == (11,)
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3,3)
        np.testing.assert_array_equal(observation['state'][:3],(rotation.T@(body.goal-body.data.qpos[:3])).astype(np.float32))
        np.testing.assert_array_equal(observation['state'][3:6],(rotation.T@body.data.qvel[:3]).astype(np.float32))
        np.testing.assert_array_equal(observation['state'][6:9],(rotation.T@[0.,0.,1.]).astype(np.float32))
        assert observation['state'][9] == pytest.approx(env.motor_env._effective_heading_offset/.45)
        assert observation['state'][10] == pytest.approx(.25)
        physical_reward = sum(frame['stats']['command_motor']['inheritedNavigationReward'] for frame in env.last_control_frames)
        terminal_adjustment = 4*sum(frame['stats']['command_motor']['successBonusRaw']
                                  -frame['stats']['command_motor']['terminalCostRawOutsideDenseClamp']
                                  for frame in env.last_control_frames)
        assert reward == pytest.approx(.01*(physical_reward+terminal_adjustment))
        assert all(not p.requires_grad for p in env._motor.policy.parameters())
        env.assert_frozen_motor()
        _assert_nested_equal(weights,env._motor.policy.state_dict())
        _assert_nested_equal(adam,env._motor.policy.optimizer.state_dict())
        assert counts == (env._motor.num_timesteps,env._motor.command_motor_physical_control_timesteps,env._motor._n_updates)
        assert _sha(_MOTOR) == _MOTOR_SHA
        yield env,observation,info,captures
    finally:
        env.close()
        patch.undo()


def test_depth_and_local_final_goal_are_the_entire_high_level_observation(real_visual_smoke):
    env,observation,info,captures = real_visual_smoke
    assert observation['depth'].shape == (3,48,64)
    assert observation['state'].shape == (11,)
    assert set(observation) == {'depth','state'}
    assert env.observation_contract['version'] >= 1
    assert 'visual_navigation' in info


def test_snapshots_and_frozen_checks_do_not_capture_depth_or_advance_physics(real_visual_smoke):
    env,_,_,captures = real_visual_smoke
    body = env.motor_env.unwrapped
    arrays = {name:getattr(body.data,name).copy() for name in
              ('qpos','qvel','qacc','ctrl','xfrc_applied','qfrc_applied','xpos','geom_xpos','actuator_force')}
    before = (len(captures),float(body.data.time),body._step_count,env.motor_env._episode_observed_substeps)
    first,second = env.snapshot(),env.snapshot()
    assert json.dumps(first,sort_keys=True) == json.dumps(second,sort_keys=True)
    env.assert_frozen_motor()
    assert before == (len(captures),float(body.data.time),body._step_count,env.motor_env._episode_observed_substeps)
    for name,expected in arrays.items():
        np.testing.assert_array_equal(getattr(body.data,name),expected,err_msg=name)


def _visual_command_stub(*, terminal_after=1, outcome='failure', goal_distance=2.):
    """Exercise the real high-level loop with no model, simulator or PPO updates."""
    class Motor(gym.Env):
        def __init__(self):
            self.observation_space=gym.spaces.Box(-np.inf,np.inf,(110,),np.float32)
            self.action_space=gym.spaces.Box(-1.,1.,(20,),np.float32)
            self.data=SimpleNamespace(time=0.,qpos=np.asarray([0.,0.,1.]),
                xmat=np.eye(3).reshape(1,9),qfrc_applied=np.zeros(26),xfrc_applied=np.zeros((1,6)))
            self.goal=np.asarray([0.,-goal_distance,0.])
            self.pelvis_body_id=0
            self.dt=.02
            self.requested_heading_offset=0.
            self.requested_lookahead_m=0.
            self.previous_pace_action=-1.
            self._effective_heading_offset=0.
            self.heading_slew_radians_per_second=.175
            self.index=0
            self.last_control_frames=[]
            self.consumed_commands=[]

        def _update_next_command(self,*,advance):
            visual._ExternalCommandMotorEnv._update_next_command(self,advance=advance)

        def _observation(self):
            return np.zeros(110,dtype=np.float32)

        def step(self,action):
            np.testing.assert_array_equal(action,np.zeros(20,dtype=np.float32))
            self.consumed_commands.append(copy.deepcopy(self._command))
            self.index+=1
            self.data.time=self.index*.02
            terminal=self.index==terminal_after
            bonus=1500. if terminal and outcome=='success' else 0.
            cost=1500. if terminal and outcome=='failure' else 0.
            raw=2.+bonus-cost
            info={'command_motor':{'observedPhysicsSubstepsThisControl':10,'inheritedNavigationReward':raw},
                  'navigation':{'reward':{'success_bonus':bonus,'terminal_cost':cost,
                    'unclamped_reward':2.+bonus,'reward_clamp_adjustment':0.}}}
            self._update_next_command(advance=True)
            self.last_control_frames=[{'time':self.data.time,'stats':copy.deepcopy(info)}]
            # Deliberately unrelated intrinsic motor reward must be ignored.
            return self._observation(),987654.,terminal and outcome!='timeout',terminal and outcome=='timeout',info

    motor=Motor()
    env=visual.VisualNavigationEnv.__new__(visual.VisualNavigationEnv)
    gym.Wrapper.__init__(env,motor)
    env.motor_env=motor
    env.decision_steps=5
    env.max_heading_radians=.45
    env.max_lookahead_m=.5
    env.stop_action_threshold=-.8
    env.reward_scale=.01
    env.record_trajectory=True
    env._has_reset=True
    env._done=False
    env.observation_contract=copy.deepcopy(visual.VISUAL_OBSERVATION_CONTRACT)
    env.control_contract={'decisionControlSteps':5,'rewardScale':.01}
    env._depth_metadata={'sampleIndex':1}
    env._last_reward_summary=None
    env._last_control_steps=env._last_physics_substeps=0
    env.last_control_frames=[]
    env._motor=SimpleNamespace(predict=lambda obs,deterministic:(np.zeros(20,dtype=np.float32),None))
    captures=[]
    env._capture_depth=lambda:captures.append(motor.data.time)
    env.assert_frozen_motor=lambda:None
    env._observation=lambda:{'depth':np.zeros((3,48,64),dtype=np.float32),'state':np.zeros(11,dtype=np.float32)}
    return env,motor,captures


@pytest.mark.parametrize('pace,expected', [
    (-1.,0.), (np.nextafter(-.8,-np.inf),0.), (-.8,.35),
    (np.float32(-.8),0.), (0.,.35+(.8/1.8)*.15), (1.,.5), (2.,.5),
])
def test_command_pace_mapping_and_strict_stop_boundary_without_physics(pace,expected):
    env,motor,captures=_visual_command_stub()
    _,_,done,truncated,info=env.step([2.,pace])
    assert done and not truncated
    assert motor.requested_heading_offset==.45
    assert motor.requested_lookahead_m==pytest.approx(expected)
    assert motor.consumed_commands[0]['effectiveLookaheadMeters']==pytest.approx(expected)
    assert info['command_control_steps']==1 and info['command_physics_substeps']==10
    assert captures==[.02]


def test_short_command_is_capped_by_actual_goal_distance_without_mutating_goal():
    env,motor,_=_visual_command_stub(goal_distance=.1)
    original_goal=motor.goal
    env.step([-2.,1.])
    assert motor.goal is original_goal
    np.testing.assert_array_equal(motor.goal,[0.,-.1,0.])
    assert motor.requested_heading_offset==-.45
    assert motor.requested_lookahead_m==.5
    assert motor.consumed_commands[0]['effectiveLookaheadMeters']==pytest.approx(.1)


@pytest.mark.parametrize('outcome,expected', [('failure',.04-75.),('success',.04+75.),('timeout',.04)])
def test_early_terminal_uses_exact_executed_counts_and_decision_terminal_scale(outcome,expected):
    env,motor,captures=_visual_command_stub(terminal_after=2,outcome=outcome)
    _,reward,done,truncated,info=env.step([0.,0.])
    assert (done,truncated)==(outcome!='timeout',outcome=='timeout')
    assert reward==pytest.approx(expected)
    assert info['command_control_steps']==2 and info['command_physics_substeps']==20
    assert [frame['time'] for frame in env.last_control_frames]==[.02,.04]
    assert captures==[.04]
    summary=info['visual_navigation']['reward']
    assert summary['executedPhysicalControlSteps']==2
    assert summary['lowIntrinsicTrackingRewardUsed'] is False
    assert summary['commandDecisionReward']==pytest.approx(expected)
    with pytest.raises(RuntimeError,match='reset'):
        env.step([0.,0.])


@pytest.mark.parametrize('action', [[0.],[0.,0.,0.],[float('nan'),0.],[0.,float('inf')]])
def test_invalid_command_does_not_reach_motor_or_camera(action):
    env,motor,captures=_visual_command_stub()
    with pytest.raises(ValueError,match='two finite actions'):
        env.step(action)
    assert motor.index==0 and captures==[]
