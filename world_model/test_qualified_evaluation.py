"""Match guards and independent contact gates without costly model training."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from .command_motor_training import _dump, _sha256
from . import qualified_evaluation as evaluation


def test_initial_checksum_includes_every_value_dtype_and_shape():
    base = np.arange(6, dtype=np.float64)
    assert evaluation.array_sha256(base) != evaluation.array_sha256(base.astype(np.float32))
    assert evaluation.array_sha256(base) != evaluation.array_sha256(base.reshape(2,3))
    changed=base.copy(); changed[-1] += .0000001
    assert evaluation.array_sha256(base) != evaluation.array_sha256(changed)
    assert evaluation.array_sha256(base)==evaluation.array_sha256(base.copy())


@pytest.mark.parametrize("kwargs",[
    {"scene_seeds":[1182000,1182000]}, {"scene_seeds":[1182000,1800000]},
    {"scene_seeds":[88701]}, {"physics_seed_base":True}, {"physics_seed_base":2**32-1},
    {"seconds":float("nan")},{"seconds":True},{"envs":0},{"envs":True},{"envs":17}])
def test_bad_cohorts_rejected_before_any_policy_io(tmp_path,kwargs):
    with pytest.raises(ValueError):
        evaluation.evaluate_cohort("missing.zip",tmp_path/"out",motor_checkpoint="missing-motor.zip",**kwargs)
    assert not (tmp_path/"out").exists()


def _cohort():
    return {"motorCheckpointSha256":"motor","avatarProfileSha256":"avatar","bodyModelSha256":"body",
        "difficultyRange":[0.,.5],"seconds":30.,"verified":0,"device":"cpu",
        "records":[{"episode":{"sceneSeed":1182000,"navigationVerified":False},
            "initialState":{"physicsSeed":3182000, "sceneSha256":"scene", "qposSha256":"qpos",
            "qvelSha256":"qvel", "depthSha256":"depth","stateSha256":"state","goalSha256":"goal"}}]}


@pytest.mark.parametrize("key",["physicsSeed","sceneSha256","qposSha256","qvelSha256","depthSha256","stateSha256","goalSha256"])
def test_comparison_rejects_different_actual_initial_state(key):
    before=_cohort();after=copy.deepcopy(before)
    after["records"][0]["initialState"][key]="changed"
    with pytest.raises(ValueError,match=key):
        evaluation.compare_cohorts(before,after)


def test_comparison_records_gains_only_after_exact_match():
    before=_cohort();after=copy.deepcopy(before)
    after["verified"]=1
    after["records"][0]["episode"]["navigationVerified"]=True
    result=evaluation.compare_cohorts(before,after)
    assert result["gainedSceneSeeds"]==[1182000]
    assert result["regressedSceneSeeds"]==[]
    assert result["exactInitialQposQvelGoalDepthStateAndSceneMatched"]


def test_whole_controller_transfer_requires_explicit_flag_and_preserves_all_initial_guards():
    before=_cohort();after=copy.deepcopy(before)
    after["motorCheckpointSha256"]="new-v4-low"
    with pytest.raises(ValueError,match="explicit whole-controller"):
        evaluation.compare_cohorts(before,after)
    result=evaluation.compare_cohorts(before,after,allow_motor_transfer=True)
    assert result["motorDynamicsChanged"] and result["explicitMotorTransferComparison"]
    assert result["beforeMotorCheckpointSha256"]=="motor"
    assert result["afterMotorCheckpointSha256"]=="new-v4-low"
    assert "cannot isolate" in result["attributionScope"]
    after["records"][0]["initialState"]["qvelSha256"]="different"
    with pytest.raises(ValueError,match="qvelSha256"):
        evaluation.compare_cohorts(before,after,allow_motor_transfer=True)


def test_different_inference_device_never_passes_matched_comparison():
    before=_cohort();after=copy.deepcopy(before);after["device"]="cuda:0"
    with pytest.raises(ValueError,match="device"):
        evaluation.compare_cohorts(before,after,allow_motor_transfer=True)


def _stored_trace(tmp_path,monkeypatch,collision=False):
    from . import navigation_evaluation
    monkeypatch.setattr(navigation_evaluation,"audit_gait",lambda trace:{"walkingVerified":True})
    stats={"fall":False,"root_position":[0.,0.,1.],"root_velocity":[0.,0.,0.],"goal_distance":.1,
        "upright":1.,"navigation_floor_support":[True,True],"navigation_success_seconds":.5,
        "navigation":{"metrics":{"floor_support_forces_newtons":[100.,100.],
        "goal_floor_matches":True,"root_height_error_m":0.}}}
    frames=[{"time":i*.02,"stats":copy.deepcopy(stats)} for i in range(26)]
    if collision:
        frames[1]["stats"]["collision_force_newtons"]=30.
    scene={"geoms":[],"spawn":[0.,0.,0.],"goals":[{"position":[0.,-.1,0.]}]}
    trace={"frames":frames,"dt":.02,"navigation":{"mode":"turn"},"scene":scene,
        "avatarProfileSha256":"avatar","bodyModelSha256":"body"}
    quality=navigation_evaluation.audit_navigation(trace)
    trace["navigationQuality"]=quality
    _dump(tmp_path/"trace.json",trace)
    arrays={"qpos":np.asarray([0.,0.,1.]+[0.]*24),"qvel":np.zeros(26),
        "depth":np.ones((3,48,64),np.float32),"state":np.zeros(11,np.float32),"goal":np.asarray([0.,-.1,0.])}
    np.savez_compressed(tmp_path/"initial-state.npz",**arrays)
    np.savez_compressed(tmp_path/"depth.npz",actualDepthStack=arrays["depth"][None],state11=arrays["state"][None])
    initial={key+"Sha256":evaluation.array_sha256(value) for key,value in arrays.items()}
    initial.update(sceneSha256=evaluation.scene_sha256(scene),avatarProfileSha256="avatar",bodyModelSha256="body")
    _dump(tmp_path/"initial-state.json",initial)
    record={"episodes":[{"quality":quality,"physicalControlSteps":25,"physical2msSubsteps":250,
        "sceneSha256":evaluation.scene_sha256(scene)}],
        "recordings":[{"filename":"trace.json","traceSha256":_sha256(tmp_path/"trace.json"),"depthFilename":"depth.npz"}],
        "avatarProfileSha256":"avatar","bodyModelSha256":"body"}
    return record


def test_independent_audit_preserves_early_contact_after_successful_final_stop(tmp_path,monkeypatch):
    record=_stored_trace(tmp_path,monkeypatch,collision=True)
    result=evaluation.independently_audit(record,tmp_path)
    assert result["allNativeFramesReaudited"]==26
    assert result["quality"]["gates"]["goalReached"]
    assert result["quality"]["gates"]["sustainedStop"]
    assert result["quality"]["failedGates"]==["collisionFree"]
    assert not result["quality"]["navigationVerified"]


def test_terminal_success_info_cannot_override_independent_contact_failure(tmp_path,monkeypatch):
    record=_stored_trace(tmp_path,monkeypatch,collision=True)
    record["episodes"][0]["quality"]=copy.deepcopy(record["episodes"][0]["quality"])
    record["episodes"][0]["quality"]["navigationVerified"]=True
    with pytest.raises(RuntimeError,match="independent native audit disagrees"):
        evaluation.independently_audit(record,tmp_path)


def test_missing_native_frame_and_changed_initial_joint_rejected(tmp_path,monkeypatch):
    record=_stored_trace(tmp_path,monkeypatch)
    trace=json.loads((tmp_path/"trace.json").read_text())
    trace["frames"].pop(1)
    _dump(tmp_path/"trace.json",trace)
    record["recordings"][0]["traceSha256"]=_sha256(tmp_path/"trace.json")
    with pytest.raises(RuntimeError,match="native audit disagrees|20 ms"):
        evaluation.independently_audit(record,tmp_path)
    record=_stored_trace(tmp_path,monkeypatch)
    with np.load(tmp_path/"initial-state.npz",allow_pickle=False) as original:
        arrays={key:original[key].copy() for key in original.files}
    arrays["qpos"][-1]=.1
    np.savez_compressed(tmp_path/"initial-state.npz",**arrays)
    with pytest.raises(RuntimeError,match="initial qpos checksum"):
        evaluation.independently_audit(record,tmp_path)


def test_first_actor_action_must_match_actual_consumed_native_command(tmp_path,monkeypatch):
    record=_stored_trace(tmp_path,monkeypatch)
    trace=json.loads((tmp_path/"trace.json").read_text())
    action=np.asarray([.4,-.6],np.float32)
    trace['navigation']['visualControlContract']={'maxHeadingRadians':.45}
    trace['frames'][1]['stats']['command_motor']={'nextCommand':{
        'requestedHeadingOffsetRadians':float(action[0])*.45}}
    trace['frames'][1]['stats']['visual_navigation']={'previousLookaheadAction':float(action[1])}
    _dump(tmp_path/'trace.json',trace)
    record['recordings'][0]['traceSha256']=_sha256(tmp_path/'trace.json')
    depth=np.ones((3,48,64),np.float32);state=np.zeros(11,np.float32)
    np.savez_compressed(tmp_path/'depth.npz',actualDepthStack=depth[None],policyDepthStack=depth[None],state11=state[None])
    def store_actor():
        arrays={'action':action,'depth':depth,'state':state}
        np.savez_compressed(tmp_path/'first-actor-input-action.npz',**arrays)
        _dump(tmp_path/'first-actor-input-action.json',{
            key+'Sha256':evaluation.array_sha256(value) for key,value in arrays.items()})
    store_actor()
    assert evaluation.independently_audit(record,tmp_path)['quality']['navigationVerified']
    # Even a consistent replacement of the NPZ and its hash cannot rewrite the
    # command that the first recorded physical control actually consumed.
    action[0]=-.4
    store_actor()
    with pytest.raises(RuntimeError,match='actual consumed physical command'):
        evaluation.independently_audit(record,tmp_path)
