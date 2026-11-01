"""Steering contracts checked against the compiled floating body and contacts."""
import ast
import inspect
import math
import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from .environment import HumanoidRoomEnv
from .humanoid import JOINT_LIMITS, JOINT_NAMES, TORQUE_LIMITS
from .locomotion import FootstepHumanoidEnv
from .steering import SteeringHumanoidEnv, _ObservedPhysicsEnv, solve_leg_with_yaw


PROFILE = Path(__file__).resolve().parents[1] / "frontend/public/world-model/gakucho-avatar-profile.json"


def _target(env, angle, distance=2.):
    env.goal = env.data.qpos[:3] + [distance*math.sin(angle), -distance*math.cos(angle), 0.]


def test_phase_one_joint_targets_remain_identical_without_heading_error():
    plain = FootstepHumanoidEnv(randomize=False, avatar_profile=PROFILE)
    turning = SteeringHumanoidEnv(randomize=False, avatar_profile=PROFILE)
    for time in (0., .30, .80, 1.40, 2.):
        plain.data.time = turning.data.time = time
        np.testing.assert_array_equal(plain._reference_pose(), turning._reference_pose())
    assert plain.body_model_sha256 == turning.body_model_sha256
    assert plain.avatar_profile_sha256 == turning.avatar_profile_sha256
    assert turning.observation_space.shape == (59,)
    assert turning.strict_gait_reward
    plain.close()
    turning.close()


def test_disabled_steering_preserves_prior_with_off_axis_goal():
    env = SteeringHumanoidEnv(randomize=False, steering_enabled=False)
    env.data.time = 1.40
    _target(env, .5)
    np.testing.assert_array_equal(env._reference_pose(), FootstepHumanoidEnv._reference_pose(env))
    env.close()


def test_mirrored_goals_produce_mirrored_joint_twists_with_unchanged_physics():
    env = SteeringHumanoidEnv(randomize=False)
    env.data.time = 1.40
    before_pos, before_vel = env.data.qpos.copy(), env.data.qvel.copy()
    _target(env, .25)
    positive = env._reference_pose()
    positive_error = env._last_steering_error
    _target(env, -.25)
    negative = env._reference_pose()
    assert positive_error == pytest.approx(.25)
    assert env._last_steering_error == pytest.approx(-.25)
    for side in ("left", "right"):
        index = JOINT_NAMES.index(f"{side}_hip_yaw")
        assert positive[index] == pytest.approx(-negative[index])
        assert abs(positive[index]) <= env.max_turn_radians
    np.testing.assert_array_equal(before_pos, env.data.qpos)
    np.testing.assert_array_equal(before_vel, env.data.qvel)
    np.testing.assert_array_equal(env.data.xfrc_applied, np.zeros_like(env.data.xfrc_applied))
    np.testing.assert_array_equal(env.data.qfrc_applied, np.zeros_like(env.data.qfrc_applied))
    env.close()


def test_heading_error_uses_physical_root_rotation_and_wraps_across_pi():
    env = SteeringHumanoidEnv(randomize=False)
    env.data.time = 1.40
    yaw = math.pi-.05
    env.data.qpos[3:7] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    mujoco.mj_forward(env.model, env.data)
    _target(env, -math.pi+.05)
    error, command = env._steering_command()
    assert error == pytest.approx(.10)
    assert 0. < command <= env.max_turn_radians
    env.close()


def test_near_waypoint_and_initial_standing_start_retain_unmodified_balance():
    env = SteeringHumanoidEnv(randomize=False, avatar_profile=PROFILE)
    _target(env, 1.)
    np.testing.assert_array_equal(env._reference_pose(), FootstepHumanoidEnv._reference_pose(env))
    env.data.time = 1.40
    env.data.qvel[5] = .8
    _target(env, 1., .1)
    np.testing.assert_array_equal(env._reference_pose(), FootstepHumanoidEnv._reference_pose(env))
    env.close()


def test_waypoint_changes_do_not_reset_gait_phase_or_root_state():
    env = SteeringHumanoidEnv(randomize=False)
    env.data.time = 2.25
    before = env.data.qpos.copy()
    phase = env.reference.sample(env.data.time+env.phase_offset).phase
    _target(env, .3)
    env._reference_pose()
    _target(env, -.3)
    env._reference_pose()
    assert env.data.time == 2.25
    assert env.reference.sample(env.data.time+env.phase_offset).phase == phase
    np.testing.assert_array_equal(env.data.qpos, before)
    env.close()


def test_actual_turning_motor_forces_remain_bounded_without_root_actuator():
    env = SteeringHumanoidEnv(randomize=False, avatar_profile=PROFILE)
    _target(env, .4)
    assert 0 not in env.model.actuator_trnid[:, 0]
    for index in range(20):
        action = np.full(20, 100. if index % 2 else -100.)
        _, _, terminated, _, info = env.step(action)
        assert np.all(abs(env.data.ctrl) <= TORQUE_LIMITS+1e-10)
        assert np.all(abs(env.data.actuator_force) <= TORQUE_LIMITS+1e-10)
        assert np.isfinite(env.data.qpos).all()
        assert np.max(abs(env.data.xfrc_applied)) == 0.
        assert np.max(abs(env.data.qfrc_applied)) == 0.
        if terminated:
            break
    assert info["steering_enabled"]
    assert abs(info["steering_joint_target_radians"]) <= env.max_turn_radians
    assert info["strict_gait_reward"]
    env.close()


def test_control_does_not_assign_floating_pose_velocity_or_external_forces():
    for method in (SteeringHumanoidEnv._reference_pose,
                   SteeringHumanoidEnv._steering_command, SteeringHumanoidEnv.step,
                   _ObservedPhysicsEnv.step):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
            for target in targets:
                assert not any(isinstance(part, ast.Attribute) and part.attr in (
                    "qpos", "qvel", "qacc", "mocap_pos", "mocap_quat", "xfrc_applied", "qfrc_applied")
                    for part in ast.walk(target))


def test_substep_observer_is_the_only_change_to_the_original_physics_step():
    original = ast.parse(textwrap.dedent(inspect.getsource(HumanoidRoomEnv.step)))
    observed = ast.parse(textwrap.dedent(inspect.getsource(_ObservedPhysicsEnv.step)))
    loop = next(node for node in ast.walk(observed) if isinstance(node, ast.For))
    observer_assignment = loop.body[-2]
    observer_call = loop.body[-1]
    assert isinstance(observer_assignment, ast.Assign)
    assert observer_assignment.targets[0].id == "observer"
    assert isinstance(observer_call, ast.If)
    del loop.body[-2:]
    assert ast.dump(original) == ast.dump(observed)


def test_read_only_substep_observation_preserves_full_rollout_bit_for_bit():
    plain = FootstepHumanoidEnv(randomize=True, avatar_profile=PROFILE,
                                strict_gait_reward=True, max_episode_seconds=6)
    observed = SteeringHumanoidEnv(randomize=True, avatar_profile=PROFILE,
                                   steering_enabled=False, max_episode_seconds=6)
    plain.reset(seed=82020)
    observed.reset(seed=82020)
    for env in (plain, observed):
        env.goal = env.spawn + [0., -.05, 0.]
    contacts = []

    def observe(model, data):
        forces = []
        for index in range(data.ncon):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            forces.append(float(wrench[0]))
        contacts.append((float(data.time), forces))

    observed._physics_contact_observer = observe
    for _ in range(300):
        first = plain.step(np.zeros(20))
        second = observed.step(np.zeros(20))
        np.testing.assert_array_equal(plain.data.qpos, observed.data.qpos)
        np.testing.assert_array_equal(plain.data.qvel, observed.data.qvel)
        np.testing.assert_array_equal(plain.data.ctrl, observed.data.ctrl)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1:4] == second[1:4]
        if first[2] or first[3]:
            break
    assert observed.data.time == pytest.approx(6.)
    assert len(contacts) == 3000
    np.testing.assert_allclose(np.diff([time for time, _ in contacts]), .002, atol=1e-12)
    assert any(any(force > 1 for force in forces) for _, forces in contacts)
    plain.close()
    observed.close()


@pytest.mark.parametrize("kwargs", [
    {"max_turn_radians": 0}, {"max_turn_radians": .5},
    {"max_turn_radians": float("nan")}, {"steering_gain": -1.},
    {"steering_gain": 3.}, {"steering_damping": -.1},
    {"steering_damping": float("inf")}, {"steering_damping": 1.},
])
def test_invalid_steering_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SteeringHumanoidEnv(**kwargs)


def _oracle_state(env, pose, root_rotation):
    """Compiled MuJoCo FK, independent of the controller's matrix formulas."""
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.model.qpos0
    data.qpos[:3] = [0., 0., env.standing_root_height]
    mujoco.mju_mat2Quat(data.qpos[3:7], root_rotation.ravel())
    data.qpos[env._qpos_indices] = pose
    mujoco.mj_forward(env.model, data)
    return data


@pytest.mark.parametrize("profile", [None, PROFILE])
def test_prescribed_hip_yaw_keeps_real_ankle_endpoints_and_flat_sole_normals(profile):
    from scipy.spatial.transform import Rotation

    env = SteeringHumanoidEnv(randomize=False, avatar_profile=profile)
    pose = env.standing_pose.copy()
    for side in ("left", "right"):
        pose[JOINT_NAMES.index(f"{side}_hip_pitch")] = -.30
        pose[JOINT_NAMES.index(f"{side}_hip_roll")] = .10 if side == "left" else -.10
        pose[JOINT_NAMES.index(f"{side}_knee")] = .60
    worst_error, worst_normal = 0., 0.
    for root_yaw in (0., .70):
        for tilt in ((0., 0.), (.06, -.06), (-.06, .06)):
            root_rotation = Rotation.from_euler("z", root_yaw).as_matrix() @ Rotation.from_euler("XY", tilt).as_matrix()
            original = _oracle_state(env, pose, root_rotation)
            for side in ("left", "right"):
                thigh = env.model.body(f"{side}_thigh").id
                foot = env.model.body(f"{side}_foot").id
                ankle = original.xpos[foot].copy()
                vector = root_rotation.T @ (ankle-original.xpos[thigh])
                upper = np.linalg.norm(env.model.body_pos[env.model.body(f"{side}_shank").id])
                lower = np.linalg.norm(env.model.body_pos[foot])
                suffixes = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
                indices = [JOINT_NAMES.index(f"{side}_{suffix}") for suffix in suffixes]
                for yaw in (-.18, -.12, -.06, .06, .12, .18):
                    targets = solve_leg_with_yaw(vector, root_rotation, [0., 0., 1.], yaw,
                                                  upper=upper, lower=lower)
                    modified = pose.copy()
                    modified[indices] = targets
                    actual = _oracle_state(env, modified, root_rotation)
                    error = np.linalg.norm(actual.xpos[foot]-ankle)
                    normal_error = np.linalg.norm(actual.xmat[foot].reshape(3, 3)[:, 2]-[0., 0., 1.])
                    worst_error = max(worst_error, error)
                    worst_normal = max(worst_normal, normal_error)
                    assert np.all(np.asarray(targets) >= JOINT_LIMITS[indices, 0]-1e-12)
                    assert np.all(np.asarray(targets) <= JOINT_LIMITS[indices, 1]+1e-12)
    assert worst_error < 1e-10
    assert worst_normal < 1e-10
    env.close()


def test_steering_retargets_both_legs_while_retaining_original_ankle_balance():
    from scipy.spatial.transform import Rotation

    env = SteeringHumanoidEnv(randomize=False, avatar_profile=PROFILE,
                               max_turn_radians=.18, steering_gain=.65)
    worst_ankle_error, worst_normal_error = 0., 0.
    for tilt in ((0., 0.), (.06, -.06), (-.06, .06)):
        root_rotation = Rotation.from_euler("z", .70).as_matrix() @ Rotation.from_euler("XY", tilt).as_matrix()
        mujoco.mju_mat2Quat(env.data.qpos[3:7], root_rotation.ravel())
        env.data.qvel[3:5] = [.10, -.10]
        mujoco.mj_forward(env.model, env.data)
        for elapsed in (1.12, 1.35, 1.57, 1.79, 2.01, 2.23):
            env.data.time = elapsed
            for difference in (-.55, .55):
                _target(env, .70+difference)
                before_pos, before_vel = env.data.qpos.copy(), env.data.qvel.copy()
                original_pose = FootstepHumanoidEnv._reference_pose(env)
                updated_pose = env._reference_pose()
                original = _oracle_state(env, original_pose, root_rotation)
                updated = _oracle_state(env, updated_pose, root_rotation)
                for side in ("left", "right"):
                    foot = env.model.body(f"{side}_foot").id
                    ankle_error = np.linalg.norm(original.xpos[foot]-updated.xpos[foot])
                    normal_error = np.linalg.norm(original.xmat[foot].reshape(3, 3)[:, 2]-updated.xmat[foot].reshape(3, 3)[:, 2])
                    worst_ankle_error = max(worst_ankle_error, ankle_error)
                    worst_normal_error = max(worst_normal_error, normal_error)
                np.testing.assert_array_equal(env.data.qpos, before_pos)
                np.testing.assert_array_equal(env.data.qvel, before_vel)
    assert worst_ankle_error < 1e-10
    assert worst_normal_error < 1e-10
    env.close()


def test_yaw_without_leg_resolve_would_change_the_real_endpoint_and_sole_normal():
    """Regression witness for the previous post-IK yaw addition."""
    env = SteeringHumanoidEnv(randomize=False, avatar_profile=PROFILE)
    env.data.time = 1.35
    pose = FootstepHumanoidEnv._reference_pose(env)
    original = _oracle_state(env, pose, np.eye(3))
    pose[JOINT_NAMES.index("left_hip_yaw")] = .18
    previous = _oracle_state(env, pose, np.eye(3))
    foot = env.model.body("left_foot").id
    assert np.linalg.norm(previous.xpos[foot]-original.xpos[foot]) > .005
    assert np.linalg.norm(previous.xmat[foot].reshape(3, 3)[:, 2]-original.xmat[foot].reshape(3, 3)[:, 2]) > .005
    env.close()
