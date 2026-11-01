"""Validate motion provenance, physical bounds, calibration and loop timing."""

import json

import mujoco
import numpy as np
import pytest

from .humanoid import JOINT_LIMITS, JOINT_NAMES, build_humanoid_xml
from .references import _BASIS, _parse_bvh, load_motion_reference


@pytest.fixture(scope="module")
def walk():
    return load_motion_reference("walk")


@pytest.fixture(scope="module")
def sit():
    return load_motion_reference("sit")


def test_bvh_intrinsic_channel_order_and_fk():
    source = """HIERARCHY
ROOT Hips {
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Tip {
  OFFSET 0 1 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site { OFFSET 0 1 0 }
 }
}
MOTION
Frames: 2
Frame Time: 0.1
0 0 0 0 0 0 0 0 0
1 2 3 90 0 0 0 0 0
"""
    bvh = _parse_bvh(source, target_height=2)
    np.testing.assert_allclose(bvh.world_positions[1, 1], [0, 2, 3], atol=1e-12)
    np.testing.assert_allclose(bvh.world_positions[1, 2], [-1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(_BASIS @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(_BASIS @ [0, 0, 1], [0, -1, 0], atol=1e-12)
    assert np.linalg.det(_BASIS) == pytest.approx(1)


def test_verified_walk_native_trajectory_and_speed_preserved(walk):
    assert walk.source_frame_count == 769
    assert walk.diagnostics["sourceManifestHashMatches"]
    assert walk.diagnostics["provenance"] == "cached_t2m_generated"
    assert walk.source_frame_range == (260, 372)
    assert walk.scale == pytest.approx(1.1306532663316582)
    assert walk.duration == pytest.approx(1.12)
    assert np.linalg.norm(walk.stride) == pytest.approx(.6483920017138907, abs=2e-7)
    assert walk.native_heading == pytest.approx(.09220043721550454, abs=2e-7)
    assert walk.source_root_positions[-1, 1] < -3.7
    first, after = walk.sample(0), walk.sample(walk.duration)
    np.testing.assert_allclose(after.root_position[:2] - first.root_position[:2], walk.stride, atol=1e-12)
    np.testing.assert_allclose(after.joint_targets, first.joint_targets, atol=1e-12)
    assert walk.sample(.56).root_velocity[1] < -.25
    assert np.ptp(walk.joint_targets[:, JOINT_NAMES.index("left_knee")]) > .1


@pytest.mark.parametrize("kind", ["walk", "sit"])
def test_physical_reference_bounds_and_actual_foot_box_calibration(kind, walk, sit):
    reference = walk if kind == "walk" else sit
    model = mujoco.MjModel.from_xml_string(build_humanoid_xml())
    data = mujoco.MjData(model)
    indices = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    for time in [0, reference.duration * .37, reference.duration]:
        sample = reference.sample(time)
        assert np.isfinite(sample.joint_targets).all()
        assert np.all(sample.joint_targets >= JOINT_LIMITS[:, 0] - 1e-12)
        assert np.all(sample.joint_targets <= JOINT_LIMITS[:, 1] + 1e-12)
        data.qpos[:] = model.qpos0
        data.qpos[:3] = sample.root_position
        data.qpos[3:7] = sample.root_quaternion
        data.qpos[indices] = sample.joint_targets
        mujoco.mj_forward(model, data)
        corners = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        lowest = []
        for side in ("left", "right"):
            geom = model.geom(f"humanoid_{side}_foot").id
            world_corners = (corners * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            lowest.append(world_corners[:, 2].min())
        # Linear interpolation of calibrated angles/height can leave sub-mm
        # residuals; this checks actual eight corners, not the calibration helper.
        assert min(lowest) == pytest.approx(.003, abs=.001)


def test_sit_has_finite_reference_end_and_explicit_loss_diagnostics(sit):
    final = sit.sample(sit.duration)
    held = sit.sample(sit.duration + 10)
    np.testing.assert_allclose(final.joint_targets, held.joint_targets)
    np.testing.assert_array_equal(held.root_velocity, np.zeros(3))
    assert sit.diagnostics["referenceOnly"]
    assert not sit.diagnostics["isPhysicalRollout"]
    assert not sit.diagnostics["rootPositionUsedAsControl"]
    assert set(sit.diagnostics["jointLimits"]) == set(JOINT_NAMES)
    assert "LeftFoot" in sit.diagnostics["positionFitMeters"]
    json.dumps(sit.diagnostics, allow_nan=False)


def test_sampling_never_changes_an_environment_root(walk):
    from .environment import HumanoidRoomEnv
    env = HumanoidRoomEnv()
    before = env.data.qpos.copy()
    time = env.data.time
    for t in np.arange(0, 4, .02):
        json.dumps(walk.sample(float(t)).to_dict(), allow_nan=False)
    np.testing.assert_array_equal(env.data.qpos, before)
    assert env.data.time == time
    env.close()
