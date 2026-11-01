"""Pure-stub evaluation accounting and depth-ablation tests; no physics runs."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import visual_evaluation as evaluation
from . import visual_navigation as navigation


def test_visual_evaluation_validates_the_entire_cohort_before_checkpoint_io():
    with pytest.raises(ValueError,match='held.?out'):
        evaluation.evaluate(None,motor_checkpoint='/missing-policy.zip',scene_seeds=[1_170_200,88711])
    with pytest.raises(ValueError,match='distinct'):
        evaluation.evaluate(None,motor_checkpoint='/missing-policy.zip',scene_seeds=[1_170_200]*2)


@pytest.mark.parametrize('blank_depth',[False,True])
def test_short_terminal_keeps_every_native_frame_and_separate_actual_ablation_depth(tmp_path,monkeypatch,blank_depth):
    instances,actor_inputs,audit_inputs = [],[],[]

    class FakeVisual:
        def __init__(self,**kwargs):
            instances.append(self)
            self.kwargs = kwargs
            self.body = SimpleNamespace(data=SimpleNamespace(time=0.,qpos=np.zeros(27),qfrc_applied=np.zeros(26),xfrc_applied=np.zeros((1,6))),
                model=SimpleNamespace(opt=SimpleNamespace(gravity=np.asarray([0.,0.,-9.81]))),
                goal=np.asarray([0.,-2.,0.]),dt=.02,avatar_profile_sha256='same-avatar',body_model_sha256='same-body')
            self.motor_env = SimpleNamespace(unwrapped=self.body)
            self._motor = SimpleNamespace(num_timesteps=7200768)
            self.scene = {'seed':kwargs['base_seed']}
            self.observation_contract = {'version':1,'depthShape':[3,48,64]}
            self.control_contract = {'decisionControlSteps':5,'motorFrozen':True}
            self.depth_metadata = {'sampleIndex':1,'actualSimTimeSeconds':0.}
            self.last_control_frames = []
            self.checks = 0
            self.closed = False
            self.depth = np.full((3,48,64),.2,dtype=np.float32)

        def reset(self,**kwargs):
            return {'depth':self.depth.copy(),'state':np.zeros(11,dtype=np.float32)},{}

        def snapshot(self):
            return {'time':self.body.data.time,'stats':{'fall':False,'goal_distance':1.,
                    'command_motor':{'observedPhysicsSubstepsThisEpisode':round(self.body.data.time/.002)}}}

        def step(self,action):
            for timestamp in (.02,.04):
                self.body.data.time=timestamp
                self.last_control_frames.append(self.snapshot())
            return {'depth':self.depth.copy(),'state':np.zeros(11,dtype=np.float32)},-15.,True,False,{
                'command_control_steps':2,'command_physics_substeps':20}

        def assert_frozen_motor(self,**kwargs):
            assert kwargs == {'check_checkpoint':True}
            self.checks += 1

        def close(self):
            self.closed = True

    def predict(observation,**kwargs):
        assert kwargs == {'deterministic':True}
        assert set(observation) == {'depth','state'}
        actor_inputs.append(observation)
        return np.zeros(2,dtype=np.float32),None

    def quality(trace):
        audit_inputs.append(trace)
        gates={'survived':True,'collisionFree':False,'goalReached':False,'stopped':True,
               'bothFeetSupport':True,'sustainedStop':False,'physicalRepeatedGait':False}
        return {'navigationVerified':False,'gates':gates,'failedGates':['collisionFree','goalReached','sustainedStop','physicalRepeatedGait']}

    monkeypatch.setattr(navigation,'VisualNavigationEnv',FakeVisual)
    monkeypatch.setattr(navigation,'assert_visual_policy',lambda policy,env:None)
    monkeypatch.setattr(evaluation,'trace_body_metadata',lambda body:{'avatarProfileSha256':'same-avatar','bodyModelSha256':'same-body'})
    monkeypatch.setattr(evaluation,'audit_navigation',quality)
    motor=tmp_path/'motor.zip';motor.write_bytes(b'fake-stub-motor')
    out=tmp_path/'evaluation'
    result=evaluation.evaluate(SimpleNamespace(predict=predict),motor_checkpoint=motor,
        scene_seeds=[1_170_200],physics_seed_base=3_170_200,output=out,blank_depth=blank_depth)
    assert len(instances)==len(actor_inputs)==len(audit_inputs)==1
    assert instances[0].kwargs['split']=='heldout' and instances[0].closed and instances[0].checks==2
    np.testing.assert_array_equal(instances[0].depth,np.full((3,48,64),.2,dtype=np.float32))
    np.testing.assert_array_equal(actor_inputs[0]['depth'],np.full((3,48,64),1. if blank_depth else .2,dtype=np.float32))
    assert [frame['time'] for frame in audit_inputs[0]['frames']]==[0.,.02,.04]
    assert result['totalDecisionSteps']==1 and result['totalPhysicalControls']==2 and result['totalPhysicsSubsteps']==20
    assert result['episodes'][0]['physicalControlFrameCount']==2
    assert result['proficiencyPassed'] is False and result['privilegedTeacherUsed'] is False
    recording=result['recordings'][0]
    assert (out/recording['filename']).is_file()
    with np.load(out/recording['depthFilename'],allow_pickle=False) as samples:
        assert samples['actualDepthStack'].shape==(1,3,48,64)
        np.testing.assert_array_equal(samples['actualDepthStack'][0],np.full((3,48,64),.2,dtype=np.float32))
        np.testing.assert_array_equal(samples['policyDepthStack'][0],np.full((3,48,64),1. if blank_depth else .2,dtype=np.float32))
        np.testing.assert_array_equal(samples['physicalTimeSeconds'],[0.])
