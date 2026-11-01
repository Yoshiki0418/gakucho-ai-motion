from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from .contact_attribution import actual_body_clearance, rollout_controller, summarize_measurements


def test_actual_surface_clearance_uses_collision_geometry_and_preserves_state():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="furniture" type="box" pos="0 0 1" size=".5 .5 .5"/>
      <body pos="1.5 0 1"><freejoint/><geom name="actor" type="sphere" size=".25"/>
        <geom name="decoration" type="sphere" size=".9" contype="0" conaffinity="0" mass="0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = SimpleNamespace(model=model, data=data)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    result = actual_body_clearance(body, {"id":"furniture", "position":[0,0,1], "size":[1,1,1]})
    assert result["actualCollisionGeomSurfaceClearanceMeters"] == pytest.approx(.75)
    assert result["rootPointClearanceMeters"] == pytest.approx(1.)
    assert result["closestAvatarCollisionGeom"] == "actor"
    np.testing.assert_array_equal(qpos,data.qpos)
    np.testing.assert_array_equal(qvel,data.qvel)
    np.testing.assert_array_equal(ctrl,data.ctrl)


def test_attribution_excludes_postcontact_yaw_and_keeps_sign_error():
    samples = []
    for time,actual in ((.02,-.1),(.04,.2),(.06,10.)):
        samples.append({"timeSeconds":time,"actualYawRateRadiansPerSecond":actual,
            "consumedCommandRelativeHeadingRadians":.3,
            "requestedHeadingOffsetRadians":.45,"desiredYawRateRadiansPerSecond":.1,
            "rootPointClearanceMeters":.4})
    report = summarize_measurements(samples,.04)
    assert report["precontactTurningControlCount"] == 2
    assert report["precontactTurningMeanActualYawRate"] == pytest.approx(.05)
    assert report["precontactTurningWrongSignFraction"] == .5
    assert report["firstRequestedTurnOverPoint2RadiansSeconds"] == .02


def test_heldout_camera_inputs_are_rejected_as_training_data_before_checkpoint_io(tmp_path):
    with pytest.raises(ValueError,match="held-out diagnostic observations"):
        rollout_controller("geometry",scene_seed=1180100,physics_seed=3180100,
            motor_checkpoint="missing.zip",avatar_profile="missing.json",output=tmp_path,
            collect_observations=True)
