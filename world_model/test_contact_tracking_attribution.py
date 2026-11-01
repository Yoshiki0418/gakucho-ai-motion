import json
import pytest

from . import contact_tracking_attribution as tracking
from . import tracked_visual_navigation as adapter


def test_explicit_motor_factory_rejects_a_changed_actual_initial_state(tmp_path, monkeypatch):
    baseline=tmp_path/'before'/'geometry-1180100'
    baseline.mkdir(parents=True)
    (baseline/'episode.json').write_text(json.dumps({'initialPhysicalState':{'qpos':[0.], 'qvel':[0.]}}))
    received=[]

    def rollout(controller,**kwargs):
        received.append((controller,kwargs))
        return {'initialPhysicalState':{'qpos':[.001], 'qvel':[0.]}}

    monkeypatch.setattr(tracking,'rollout_controller',rollout)
    with pytest.raises(RuntimeError,match='different scene or actual qpos/qvel'):
        tracking.evaluate_tracking_motor(motor_checkpoint='new-tracking-motor.zip',avatar_profile='avatar.json',
            baseline=baseline.parent,output=tmp_path/'after',seeds=(1180100,))
    controller, kwargs=received[0]
    assert controller=='geometry'
    assert kwargs['env_factory'] is adapter.make_visual_env
    assert kwargs['scene_seed']==1180100 and kwargs['physics_seed']==3180100


def test_reject_completed_tracking_diagnostic_before_opening_old_data(tmp_path):
    with pytest.raises(ValueError,match='preserve completed'):
        tracking.evaluate_tracking_motor(motor_checkpoint='missing.zip',avatar_profile='missing.json',
            baseline='missing',output=tmp_path)
