"""Independent MuJoCo FK and control contracts for the adapted footstep prior."""

import ast
import inspect
import math
import textwrap
import time

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .environment import HumanoidRoomEnv
from .humanoid import JOINT_NAMES, TORQUE_LIMITS, build_humanoid_xml
from .imitation import ReferenceHumanoidEnv
from .locomotion import FootstepHumanoidEnv, solve_leg


def _leg_oracle(model, data, side, yaw, pitch, roll, y, lift):
    """Evaluate solver output through the compiled MuJoCo joint chain."""
    sign = 1 if side == "left" else -1
    horizontal = Rotation.from_euler("z", yaw).as_matrix()
    root_rotation = horizontal @ Rotation.from_euler("XY", [pitch, roll]).as_matrix()
    root = np.asarray([0., 0., .90])
    foot_id = model.geom(f"humanoid_{side}_foot").id
    hip_id = model.body(f"{side}_thigh").id
    centre = horizontal @ np.asarray([sign * .10, y, 0.]) + [0, 0, lift + model.geom_size[foot_id, 2]]
    # Recover the desired ankle origin from the actual compiled geometry's
    # offset; this independently checks the controller's centre/ankle mapping.
    ankle = centre - horizontal @ model.geom_pos[foot_id]
    vector = root_rotation.T @ (ankle - root) - model.body_pos[hip_id]
    angles = solve_leg(vector, root_rotation, horizontal)
    names = [f"{side}_{name}" for name in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")]
    data.qpos[:] = model.qpos0
    data.qpos[:3] = root
    mujoco.mju_mat2Quat(data.qpos[3:7], root_rotation.ravel())
    for name, value in zip(names, angles):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    actual_rotation = data.geom_xmat[foot_id].reshape(3, 3)
    joint_ranges = [model.jnt_range[model.joint(name).id] for name in names]
    reach = np.linalg.norm(model.body_pos[model.body(f"{side}_shank").id]) + np.linalg.norm(model.body_pos[model.body(f"{side}_foot").id])
    return {
        "yaw": yaw, "rootPitch": pitch, "rootRoll": roll, "side": side,
        "footY": y, "soleHeight": lift, "targetDistance": float(np.linalg.norm(vector)),
        "unclampedReach": float(np.linalg.norm(vector)) <= reach - .002 + 1e-10,
        "physicalReach": float(np.linalg.norm(vector)) <= reach + 1e-10,
        "footCenterErrorMeters": float(np.linalg.norm(data.geom_xpos[foot_id] - centre)),
        "ankleErrorMeters": float(np.linalg.norm(data.xpos[model.body(f"{side}_foot").id] - ankle)),
        "footUpZ": float(actual_rotation[2, 2]),
        "actualSoleHeight": float(data.geom_xpos[foot_id, 2] - np.abs(actual_rotation[2]).dot(model.geom_size[foot_id])),
        "withinJointLimits": all(low - 1e-12 <= value <= high + 1e-12
                                 for value, (low, high) in zip(angles, joint_ranges)),
        "angles": list(angles),
        "actualFootCenter": data.geom_xpos[foot_id].copy().tolist(),
    }


def collect_leg_oracle_results():
    model = mujoco.MjModel.from_xml_string(build_humanoid_xml())
    data = mujoco.MjData(model)
    return [_leg_oracle(model, data, side, yaw, pitch, roll, y, lift)
            for yaw in (0., .7, math.pi / 2)
            for pitch, roll in ((0., 0.), (.06, .06), (-.06, -.06), (.10, -.08))
            for side in ("left", "right") for y in (-.15, 0., .15) for lift in (0., .035, .075)]


def test_compiled_mujoco_fk_matches_reachable_foot_targets_and_flat_soles():
    rows = collect_leg_oracle_results()
    reachable = [row for row in rows if row["unclampedReach"]]
    assert len(rows) == 216 and len(reachable) > 190
    assert max(row["ankleErrorMeters"] for row in reachable) < 1e-10
    assert max(row["footCenterErrorMeters"] for row in reachable) < .005
    assert min(row["footUpZ"] for row in reachable) > .999999
    assert max(abs(row["actualSoleHeight"] - row["soleHeight"]) for row in reachable) < 1e-10
    assert all(row["withinJointLimits"] for row in rows)
    assert min(row["angles"][2] for row in rows) > 0  # No knee hyperextension.


def test_ik_targets_and_actual_geometry_are_covariant_under_room_yaw():
    model = mujoco.MjModel.from_xml_string(build_humanoid_xml())
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        base = _leg_oracle(model, data, side, 0, .08, -.05, -.12, .055)
        for yaw in (-math.pi / 2, .7, math.pi):
            turned = _leg_oracle(model, data, side, yaw, .08, -.05, -.12, .055)
            np.testing.assert_allclose(base["angles"], turned["angles"], atol=1e-12)
            expected = Rotation.from_euler("z", yaw).apply(base["actualFootCenter"])
            np.testing.assert_allclose(expected, turned["actualFootCenter"], atol=1e-12)


def test_standing_start_native_t2m_cycle_and_adaptation_metadata():
    env = FootstepHumanoidEnv(randomize=False)
    observation, _ = env.reset(seed=18623)
    np.testing.assert_array_equal(env.data.qvel, np.zeros(env.model.nv))
    np.testing.assert_array_equal(env.data.qpos[env._qpos_indices], env.standing_pose)
    assert not env.reference_start
    assert env.reference.duration == pytest.approx(1.12)
    before = env.data.qpos.copy()
    env.data.time = 1.4
    first = env._reference_pose()
    env.data.time += env.reference.duration
    second = env._reference_pose()
    np.testing.assert_allclose(first, second, atol=1e-10)
    np.testing.assert_array_equal(before, env.data.qpos)
    env.reset(seed=18623)
    _, _, _, _, info = env.step(np.zeros(20))
    assert env.data.time == pytest.approx(.02, abs=1e-12)
    assert observation.shape == (59,)
    assert "two-link joint IK" in info["physical_gait_prior"]
    assert "bounded torque PD" in info["physical_gait_prior"]
    assert not info["reference_start"]
    env.close()


def test_actual_motor_forces_are_bounded_and_floating_root_has_no_actuator():
    env = FootstepHumanoidEnv(randomize=False)
    env.reset(seed=133)
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert 0 not in env.model.actuator_trnid[:, 0]
    for sign in (1, -1):
        env.step(np.full(20, sign * 100.))
        assert np.all(np.abs(env.data.ctrl) <= TORQUE_LIMITS + 1e-10)
        assert np.all(np.abs(env.data.actuator_force) <= TORQUE_LIMITS + 1e-10)
    env.close()


def test_reference_and_step_never_assign_floating_pose_or_mocap_state():
    for cls in (HumanoidRoomEnv, ReferenceHumanoidEnv, FootstepHumanoidEnv):
        for method in (cls._reference_pose, cls.step):
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                    targets = [node.target]
                else:
                    continue
                for target in targets:
                    assert not any(isinstance(part, ast.Attribute) and part.attr in ("qpos", "mocap_pos", "mocap_quat")
                                   for part in ast.walk(target))


def test_close_goal_retains_the_same_tilt_feedback_as_standing_balance():
    env = FootstepHumanoidEnv(randomize=False)
    env.reset(seed=6133)
    env.data.time = 2.0
    rotation = Rotation.from_euler("XY", [.10, .07]).as_matrix()
    mujoco.mju_mat2Quat(env.data.qpos[3:7], rotation.ravel())
    env.data.qvel[3:5] = [.20, -.15]
    env.goal = env.data.qpos[:3] + [0, -.05, 0]
    mujoco.mj_forward(env.model, env.data)
    before = env.data.qpos.copy()
    reference = env._reference_pose()
    # Existing gravity balance is a separate implementation, rather than a
    # copy of the footstep blend. Gait amplitude zero must retain its feedback.
    baseline = HumanoidRoomEnv._reference_pose(env)
    np.testing.assert_allclose(reference, baseline, atol=1e-12)
    assert abs(reference[JOINT_NAMES.index("left_ankle_pitch")] - env.standing_pose[JOINT_NAMES.index("left_ankle_pitch")]) > .20
    np.testing.assert_array_equal(before, env.data.qpos)
    env.close()


def test_overshooting_goal_does_not_restart_forward_analytic_steps():
    env = FootstepHumanoidEnv(randomize=False)
    env.reset(seed=4713)
    env.data.time = 1.4
    env.goal = env.spawn + [0, -.50, 0]
    moving = env._reference_pose()
    assert np.linalg.norm(moving - env.standing_pose) > .1
    env.goal = env.spawn + [0, .50, 0]  # Behind a body facing -Y.
    stopped = env._reference_pose()
    np.testing.assert_allclose(stopped, HumanoidRoomEnv._reference_pose(env), atol=1e-12)
    env.close()


def test_zero_residual_stopped_body_keeps_physical_balance_for_six_seconds():
    env = FootstepHumanoidEnv(randomize=False, max_episode_seconds=6)
    env.reset(seed=51031)
    env.goal = env.spawn + [0, -.05, 0]
    for _ in range(300):
        _, _, terminated, _, info = env.step(np.zeros(20))
        assert not terminated
    assert info["time"] == pytest.approx(6, abs=1e-10)
    assert info["upright"] > .99 and .90 < info["root_height"] < .96
    assert info["left_foot_force"] > 100 and info["right_foot_force"] > 100
    env.close()


def test_delayed_control_call_still_integrates_only_ten_substeps(monkeypatch):
    env = FootstepHumanoidEnv(randomize=False)
    env.reset(seed=31003)
    original = mujoco.mj_step
    calls = []

    def counted(model, data):
        calls.append(float(data.time))
        original(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted)
    time.sleep(.04)
    env.step(np.zeros(20))
    assert len(calls) == 10
    assert env.data.time == pytest.approx(.02, abs=1e-12)
    np.testing.assert_allclose(np.diff(calls), np.full(9, .002), atol=1e-12)
    env.close()
