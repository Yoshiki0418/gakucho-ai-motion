"""Measured dimensions checked through compiled FK and actual contact physics."""

import copy
import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .avatar_profile import avatar_profile_sha256, load_avatar_profile
from .environment import HumanoidRoomEnv
from .humanoid import JOINT_NAMES, TORQUE_LIMITS, build_humanoid_xml
from .imitation import ReferenceHumanoidEnv
from .locomotion import FootstepHumanoidEnv, solve_leg
from .references import load_motion_reference


def measured_test_profile():
    """An independent asymmetric measured-body fixture, not the generic rig."""
    root = .79
    hips = [0., 0., root]
    return {"version": 1, "avatar": "synthetic-measured-body", "physical": {
        "coordinateSystem": "metres, Z-up, -Y-forward", "rootHeight": root, "height": 1.52,
        "hipOffsets": {"left": [.075, .008, -.035], "right": [-.078, .008, -.035]},
        "legs": {side: {"thighLength": .335, "shankLength": .350} for side in ("left", "right")},
        "arms": {side: {"upperArmLength": .245, "forearmLength": .220} for side in ("left", "right")},
        "foot": {side: {"halfSize": [.048, .113, .031], "centerOffset": [.003, -.050, -.035]}
                 for side in ("left", "right")},
        "joints": {"hips": {"position": hips}, "spine": {"position": [0., -.015, .86]},
                   "spine4": {"position": [0., -.020, 1.12]}, "head": {"position": [0., -.010, 1.36]},
                   "leftUpperArm": {"position": [.168, -.013, 1.12]},
                   "rightUpperArm": {"position": [-.171, -.013, 1.12]}},
        "head": {"radius": .105, "centerOffset": [0., -.015, .015]}}}


def test_geometry_uses_measured_joint_positions_lengths_and_actual_shoe_bounds():
    profile = measured_test_profile()
    env = HumanoidRoomEnv(avatar_profile=profile)
    rest = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, rest)
    for physical, body in (("hips", "pelvis"), ("spine", "torso"), ("head", "head"),
                           ("leftUpperArm", "left_upper_arm"), ("rightUpperArm", "right_upper_arm")):
        np.testing.assert_allclose(rest.xpos[env.model.body(body).id], profile["physical"]["joints"][physical]["position"], atol=1e-10)
    for side in ("left", "right"):
        physical = profile["physical"]
        for child, length in (("shank", "thighLength"), ("foot", "shankLength")):
            assert np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_{child}").id]) == pytest.approx(physical["legs"][side][length])
        assert np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_forearm").id]) == pytest.approx(physical["arms"][side]["upperArmLength"])
        geom = env.model.geom(f"humanoid_{side}_foot").id
        np.testing.assert_allclose(env.model.geom_size[geom], physical["foot"][side]["halfSize"])
        np.testing.assert_allclose(env.model.geom_pos[geom], physical["foot"][side]["centerOffset"])
    assert env.model.body_mass.sum() == pytest.approx(64.7)
    assert not env.physical_assumptions["massWasMeasured"]
    assert not env.physical_assumptions["jointStrengthWasMeasured"]
    assert not np.any(env.model.actuator_trnid[:, 0] == 0)
    np.testing.assert_allclose(env.model.actuator_forcerange[:, 1], TORQUE_LIMITS)
    env.close()


def test_standing_calibration_places_actual_shoe_corners_above_the_floor_and_snapshots_bodies():
    env = HumanoidRoomEnv(avatar_profile=measured_test_profile())
    corners = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for geom in env._foot_geom_ids:
        actual_corners = (corners*env.model.geom_size[geom]) @ env.data.geom_xmat[geom].reshape(3, 3).T + env.data.geom_xpos[geom]
        assert actual_corners[:, 2].min() == pytest.approx(.0005, abs=1e-8)
    frames = env.snapshot()["bodies"]
    frozen = copy.deepcopy(frames)
    for body in frames:
        body_id = env.model.body(body["name"]).id
        np.testing.assert_allclose(body["position"], env.data.xpos[body_id])
        np.testing.assert_allclose(body["quaternion"], env.data.xquat[body_id])
    env.step(np.zeros(20))
    assert frames == frozen
    assert len(env.physics_rest_bodies) == len(frames)
    env.close()


def test_body_contract_is_scene_independent_and_profile_is_owned(tmp_path):
    profile = measured_test_profile()
    path = tmp_path / "avatar.json"
    path.write_text(json.dumps(profile))
    plain = HumanoidRoomEnv(avatar_profile=path)
    scene = {"spawn": [0, 0, 0], "geoms": [{"id": "other_floor", "position": [0, 0, -.2], "size": [12, 8, .4]}]}
    other = HumanoidRoomEnv(scene=scene, avatar_profile=profile)
    assert plain.body_model_sha256 == other.body_model_sha256
    assert plain.avatar_profile_sha256 == other.avatar_profile_sha256
    profile["physical"]["legs"]["left"]["thighLength"] = .5
    assert other.avatar_profile["physical"]["legs"]["left"]["thighLength"] == .335
    assert avatar_profile_sha256({**other.avatar_profile, "profileSha256": "untrusted-self-hash"}) == other.avatar_profile_sha256
    generic = HumanoidRoomEnv()
    assert generic.body_model_sha256 != other.body_model_sha256
    generic.close()
    plain.close()
    other.close()


@pytest.mark.parametrize("field,value", [("rootHeight", float("nan")), ("rootHeight", .1),
                                        ("coordinateSystem", "Y-up"), ("height", -1)])
def test_invalid_profile_is_rejected_before_physics_compilation(field, value):
    profile = measured_test_profile()
    profile["physical"][field] = value
    with pytest.raises(ValueError):
        build_humanoid_xml(avatar_profile=profile)


def test_reference_grounds_the_profile_body_and_is_not_reused_across_different_dimensions():
    profile = measured_test_profile()
    reference = load_motion_reference("walk", avatar_profile=profile)
    generic = load_motion_reference("walk")
    env = ReferenceHumanoidEnv(task="walk", reference_start=False, avatar_profile=profile)
    assert reference.avatar_profile_sha256 == env.avatar_profile_sha256
    assert reference.body_model_sha256 == env.body_model_sha256
    assert reference.scale/generic.scale == pytest.approx(1.52/1.8)
    assert reference.diagnostics["sourceSha256"] == generic.diagnostics["sourceSha256"]
    assert reference.duration == generic.duration
    assert reference.diagnostics["normalizationTargetHeightMeters"] == 1.52
    assert np.max(np.abs(reference.root_positions[:, 2]-generic.root_positions[:, 2])) > .1
    data = mujoco.MjData(env.model)
    for time in (0., .4, .8, 1.12):
        sample = reference.sample(time)
        data.qpos[:3] = sample.root_position
        data.qpos[3:7] = sample.root_quaternion
        data.qpos[env._qpos_indices] = sample.joint_targets
        mujoco.mj_forward(env.model, data)
        lowest = [data.geom_xpos[geom, 2]-np.abs(data.geom_xmat[geom].reshape(3, 3)[2]).dot(env.model.geom_size[geom])
                  for geom in env._foot_geom_ids]
        assert min(lowest) == pytest.approx(.003, abs=.001)
    np.testing.assert_array_equal(env.data.qpos[env._qpos_indices], env.standing_pose)
    env.close()


def test_measured_unequal_leg_ik_matches_independent_mujoco_fk_and_cannot_move_the_root():
    env = FootstepHumanoidEnv(avatar_profile=measured_test_profile())
    root = np.asarray([.2, -.3, env.standing_root_height-.03*env._body_scale])
    horizontal = Rotation.from_euler("z", .7).as_matrix()
    root_rotation = horizontal @ Rotation.from_euler("XY", [.03, -.04]).as_matrix()
    data = mujoco.MjData(env.model)
    for side in ("left", "right"):
        hip = env.model.body_pos[env.model.body(f"{side}_thigh").id]
        upper = np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_shank").id])
        lower = np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_foot").id])
        geom = env.model.geom(f"humanoid_{side}_foot").id
        for longitudinal in (-.08, 0., .08):
            centre = horizontal @ [hip[0]+env.model.geom_pos[geom, 0], hip[1]+env.model.geom_pos[geom, 1]+longitudinal, 0]
            centre[:2] += root[:2]
            centre[2] = env.model.geom_size[geom, 2]+.04
            ankle = centre-horizontal @ env.model.geom_pos[geom]
            vector = root_rotation.T @ (ankle-root)-hip
            angles = solve_leg(vector, root_rotation, horizontal, upper=upper, lower=lower)
            data.qpos[:] = env.model.qpos0
            data.qpos[:3] = root
            mujoco.mju_mat2Quat(data.qpos[3:7], root_rotation.ravel())
            for suffix, value in zip(("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"), angles):
                data.qpos[env.model.jnt_qposadr[env.model.joint(f"{side}_{suffix}").id]] = value
            mujoco.mj_forward(env.model, data)
            np.testing.assert_allclose(data.xpos[env.model.body(f"{side}_foot").id], ankle, atol=1e-10)
            # The two-axis ankle has no yaw, so the small lateral fit residual
            # is bounded while the sole stays horizontal through the full FK.
            assert np.linalg.norm(data.geom_xpos[geom]-centre) < .004
            assert data.geom_xmat[geom].reshape(3, 3)[2, 2] > .999999
    before = env.data.qpos.copy()
    env.data.time = 1.4
    assert np.isfinite(env._reference_pose()).all()
    np.testing.assert_array_equal(env.data.qpos, before)
    env.close()


def test_actual_gakucho_rest_joint_chain_matches_measured_major_bones_and_shoe_contact():
    from pathlib import Path
    profile_path = Path(__file__).resolve().parents[1] / "frontend/public/world-model/gakucho-avatar-profile.json"
    profile = load_avatar_profile(profile_path)
    env = HumanoidRoomEnv(avatar_profile=profile, max_episode_seconds=6)
    data = mujoco.MjData(env.model)
    # Bring the compact rig's down-bound arms into the measured display T-pose
    # through real shoulder hinges; no segment stretch or root placement fit.
    for side, sign in (("left", -1), ("right", 1)):
        data.qpos[env.model.jnt_qposadr[env.model.joint(f"{side}_shoulder_roll").id]] = sign*np.pi/2
    mujoco.mj_forward(env.model, data)
    mappings = {"hips": "pelvis", "spine": "torso", "head": "head"}
    for side in ("left", "right"):
        mappings.update({side+"UpLeg": side+"_thigh", side+"LowerLeg": side+"_shank",
                         side+"Foot": side+"_foot", side+"UpperArm": side+"_upper_arm",
                         side+"LowerArm": side+"_forearm"})
    for canonical, body in mappings.items():
        np.testing.assert_allclose(data.xpos[env.model.body(body).id],
                                   profile["physical"]["joints"][canonical]["position"], atol=5e-6)
    # The hand sphere represents fingers as well; compare the wrist endpoint
    # obtained independently from the forearm body and measured length.
    for side in ("left", "right"):
        body = env.model.body(f"{side}_forearm").id
        endpoint = data.xpos[body] + data.xmat[body].reshape(3, 3) @ [0, 0, -profile["physical"]["arms"][side]["forearmLength"]]
        np.testing.assert_allclose(endpoint, profile["physical"]["joints"][side+"Hand"]["position"], atol=5e-6)
    for _ in range(300):
        _, _, terminated, _, info = env.step(np.zeros(20))
        assert not terminated
    assert info["success"] and info["upright"] > .99
    assert info["left_foot_force"] > 200 and info["right_foot_force"] > 200
    assert info["left_foot_force"]+info["right_foot_force"] == pytest.approx(64.7*9.81, abs=1)
    env.close()


def test_actual_seated_pelvis_bottom_and_support_match_measured_skin_envelope():
    """An initialized physical seat-contact test, not a learned sit claim."""
    from pathlib import Path
    from .humanoid import PD_KP, PD_KD
    profile_path = Path(__file__).resolve().parents[1] / "frontend/public/world-model/gakucho-avatar-profile.json"
    profile = load_avatar_profile(profile_path)
    seat_top = .443916
    scene = {"geoms": [{"id": "floor", "position": [0, 0, -.1], "size": [10, 10, .2]},
                       {"id": "seat", "position": [0, 0, seat_top-.025], "size": [.6, .6, .05], "semantic": "seat"}],
             "goals": [{"kind": "seat", "position": [0, 0, seat_top]}]}
    env = HumanoidRoomEnv(task="sit", scene=scene, avatar_profile=profile)
    assert env.seat_pelvis_offset == pytest.approx(profile["physical"]["pelvis"]["seatOffset"], abs=1e-9)
    assert env.goal[2] == pytest.approx(seat_top+env.seat_pelvis_offset)
    # Set a test initial condition with 90-degree hips/knees, matching the
    # independent skin measurement. Subsequent motion is real gravity/contact.
    env.data.qpos[:3] = [0, 0, seat_top+env.seat_pelvis_offset-.0003]
    env.data.qpos[env._qpos_indices] = 0
    for side in ("left", "right"):
        env.data.qpos[env.model.jnt_qposadr[env.model.joint(side+"_hip_pitch").id]] = -np.pi/2
        env.data.qpos[env.model.jnt_qposadr[env.model.joint(side+"_knee").id]] = np.pi/2
    mujoco.mj_forward(env.model, env.data)
    geom = env.model.geom("humanoid_pelvis").id
    initial_bottom = env.data.geom_xpos[geom, 2]-env.model.geom_size[geom, 0]
    assert initial_bottom == pytest.approx(seat_top-.0003, abs=1e-9)
    target = env.data.qpos[env._qpos_indices].copy()
    for _ in range(100):
        env.data.ctrl[:] = np.clip(PD_KP*(target-env.data.qpos[env._qpos_indices])-PD_KD*env.data.qvel[env._qvel_indices], -TORQUE_LIMITS, TORQUE_LIMITS)
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    info = env._info()
    assert not info["fall"] and info["upright"] > .98
    assert info["pelvis_support_contact"]
    assert info["pelvis_support_force"] > .3*64.7*9.81
    assert abs(info["root_height"]-env.goal[2]) < .005
    np.testing.assert_array_equal(env.data.xfrc_applied, np.zeros_like(env.data.xfrc_applied))
    np.testing.assert_array_equal(env.data.qfrc_applied, np.zeros_like(env.data.qfrc_applied))
    env.close()
