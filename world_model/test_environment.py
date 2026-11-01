"""Checks of the physical rig and Gym contract, independent of training rewards."""

import copy

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from .environment import HumanoidRoomEnv
from .humanoid import TORQUE_LIMITS


def test_gym_contract_and_actual_control_timestep():
    env = HumanoidRoomEnv()
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=8)
    assert env.observation_space.contains(observation)
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.nv == env.model.nu + 6
    assert env.model.body_mass.sum() > 50
    assert not any(env.model.actuator_trnid[:, 0] == 0)
    env.step(np.zeros(env.action_space.shape))
    assert env.data.time == pytest.approx(.02, abs=1e-12)
    env.close()


def test_torque_is_bounded_even_for_extreme_action():
    env = HumanoidRoomEnv()
    env.reset(seed=6)
    for sign in [1, -1, 1]:
        _, _, done, _, _ = env.step(np.full(env.action_space.shape, sign * 100.))
        assert np.all(np.abs(env.data.ctrl) <= TORQUE_LIMITS + 1e-12)
        assert np.all(np.abs(env.data.actuator_force) <= TORQUE_LIMITS + 1e-12)
        if done:
            break
    env.close()


def test_randomization_is_seeded_and_does_not_accumulate():
    env = HumanoidRoomEnv(randomize=True)
    first, _ = env.reset(seed=42)
    mass = env.model.body_mass.copy()
    friction = env.model.geom_friction.copy()
    env.reset(seed=7)
    repeated, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(mass, env.model.body_mass)
    np.testing.assert_array_equal(friction, env.model.geom_friction)
    env.close()


def test_external_force_changes_free_root_without_a_placement_override():
    def trajectory(force):
        env = HumanoidRoomEnv()
        env.reset(seed=8)
        for _ in range(10):
            env.apply_push(force)
            env.step(np.zeros(env.action_space.shape))
        position = env.data.qpos[:3].copy()
        env.close()
        return position
    quiet = trajectory([0, 0, 0])
    pushed = trajectory([200, 0, 0])
    assert pushed[0] - quiet[0] > .015


def test_invalid_action_is_rejected_before_dynamics_step():
    env = HumanoidRoomEnv()
    env.reset(seed=8)
    qpos = env.data.qpos.copy()
    time = env.data.time
    with pytest.raises(ValueError):
        env.step(np.full(env.action_space.shape, np.nan))
    np.testing.assert_array_equal(qpos, env.data.qpos)
    assert time == env.data.time
    env.close()


def test_snapshot_is_actual_world_geometry_and_owns_its_arrays():
    env = HumanoidRoomEnv()
    env.reset(seed=8)
    snapshot = env.snapshot()
    frozen = copy.deepcopy(snapshot)
    for i, geom in enumerate(snapshot["geoms"]):
        np.testing.assert_allclose(geom["position"], env.data.geom_xpos[i], atol=1e-12)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(geom["quaternion"]))
        np.testing.assert_allclose(rotation, env.data.geom_xmat[i], atol=1e-12)
    env.apply_push([100, 0, 0])
    env.step(np.zeros(env.action_space.shape))
    assert snapshot == frozen
    env.close()
