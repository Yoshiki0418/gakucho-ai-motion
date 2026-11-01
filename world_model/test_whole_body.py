"""Physical checks for the torque-only WBC baseline, without claiming a gait."""

import json

import mujoco
import numpy as np
import pytest

from .environment import HumanoidRoomEnv
from .gait_audit import audit_gait
from .humanoid import TORQUE_LIMITS
from .whole_body import WholeBodyController


@pytest.mark.parametrize("walk,goal_direction", [(False, False), (True, False), (True, True)])
def test_inverse_dynamics_computation_cannot_move_the_floating_body(walk, goal_direction):
    env = HumanoidRoomEnv(task="walk" if walk else "stand")
    env.reset(seed=0)
    controller = WholeBodyController(env, walk=walk, goal_direction=goal_direction)
    position, velocity, time = env.data.qpos.copy(), env.data.qvel.copy(), env.data.time
    torque = controller.compute_torques()
    np.testing.assert_array_equal(env.data.qpos, position)
    np.testing.assert_array_equal(env.data.qvel, velocity)
    assert env.data.time == time
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert not np.any(env.model.actuator_trnid[:, 0] == 0)
    assert env.model.nmocap == 0
    assert np.all(np.abs(torque) <= TORQUE_LIMITS)
    assert controller.diagnostics["qp_solved"]
    assert controller.diagnostics["dynamics_residual"] < 1e-5
    np.testing.assert_array_equal(env.data.xfrc_applied, 0)
    np.testing.assert_array_equal(env.data.qfrc_applied, 0)
    env.close()


@pytest.mark.parametrize("seed,randomize", [(0, False), (73, True)])
def test_six_seconds_standing_is_supported_by_actual_floor_contacts(seed, randomize):
    env = HumanoidRoomEnv(randomize=randomize, max_episode_seconds=6.02)
    env.reset(seed=seed)
    controller = WholeBodyController(env)
    anchors = [anchor[0].copy() for anchor in controller.anchors]
    minimum_upright = 1.
    for _ in range(300):
        info = controller.step()
        minimum_upright = min(minimum_upright, info["upright"])
        assert not info["fall"]
        assert info["qp_solved"]
        assert np.all(np.abs(env.data.ctrl) <= TORQUE_LIMITS + 1e-10)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
    assert env.data.time == pytest.approx(6., abs=1e-10)
    assert minimum_upright > .98
    assert info["left_foot_contact"] and info["right_foot_contact"]
    assert info["left_foot_force"] + info["right_foot_force"] == pytest.approx(controller.weight, rel=.03)
    assert np.linalg.norm(env.data.qvel[:3]) < .05
    for index in range(2):
        np.testing.assert_array_equal(controller.anchors[index][0], anchors[index])
        assert np.linalg.norm(controller._foot_state(index)[0] - anchors[index]) < .015
    snapshot = controller.snapshot()
    json.dumps(snapshot, allow_nan=False)
    for index, geom in enumerate(snapshot["geoms"]):
        np.testing.assert_array_equal(geom["position"], env.data.geom_xpos[index])
    env.close()


def test_reset_updates_actual_randomized_mass_without_editing_physics():
    env = HumanoidRoomEnv(randomize=True)
    controller = WholeBodyController(env)
    env.reset(seed=997)
    position, velocity = env.data.qpos.copy(), env.data.qvel.copy()
    controller.reset()
    np.testing.assert_array_equal(env.data.qpos, position)
    np.testing.assert_array_equal(env.data.qvel, velocity)
    assert controller.mass == pytest.approx(env.model.body_subtreemass[env.pelvis_body_id])
    assert controller.weight == pytest.approx(controller.mass * 9.81)
    assert controller.completed_steps == 0
    assert controller.phase == "standing"
    env.close()


def test_nominal_flat_floor_walk_uses_real_swings_and_stops_at_the_goal():
    """This gate covers the nominal 1 m baseline, not unseen-room robustness."""
    env = HumanoidRoomEnv(task="walk", max_episode_seconds=25)
    env.reset(seed=0)
    env.goal[:2] = [0., -1.]
    controller = WholeBodyController(env, walk=True)
    trace = {"task": "walk", "controller": "Whole-body inverse dynamics QP (not RL)",
             "sizeConvention": "mujoco", "dt": env.dt,
             "scene": {"goals": [{"kind": "walk", "position": env.goal.tolist()}]},
             "frames": [controller.snapshot()]}
    stopped_at = None
    for _ in range(env.max_steps):
        info = controller.step()
        trace["frames"].append(controller.snapshot())
        assert not info["fall"]
        assert controller.failure is None
        assert info["qp_solved"]
        assert np.all(np.abs(env.data.ctrl) <= TORQUE_LIMITS + 1e-10)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
        if controller.phase == "done":
            stopped_at = env.data.time if stopped_at is None else stopped_at
            if env.data.time - stopped_at >= 1.:
                break
    assert controller.phase == "done"
    assert controller.completed_steps >= 6
    assert info["success"]
    assert info["left_foot_contact"] and info["right_foot_contact"]
    assert np.linalg.norm(env.data.qvel[:3]) < .2
    audit = audit_gait(trace)
    assert audit["walkingVerified"], audit["failedGates"]
    for frame in trace["frames"]:
        assert frame["stats"]["max_external_body_force"] == 0.
        assert frame["stats"]["max_external_generalized_force"] == 0.
    env.close()


def test_goal_direction_diagonal_approach_stops_without_crossing_or_external_forces():
    env = HumanoidRoomEnv(task="walk", max_episode_seconds=40)
    env.reset(seed=0)
    env.goal[:2] = [.28, -.10]
    controller = WholeBodyController(env, walk=True, goal_direction=True)
    stopped_at = None
    for _ in range(env.max_steps):
        info = controller.step()
        assert not info["fall"]
        assert controller.failure is None
        assert np.all(np.abs(env.data.ctrl) <= TORQUE_LIMITS + 1e-10)
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        np.testing.assert_array_equal(env.data.qfrc_applied, 0)
        snapshot = controller.snapshot()
        assert snapshot["stats"]["foot_pair_contact_force"] == 0.
        soles = [controller._foot_state(index)[0] for index in range(2)]
        assert controller._lane_axis @ (soles[0][:2] - soles[1][:2]) > 0
        if controller.phase == "done":
            stopped_at = env.data.time if stopped_at is None else stopped_at
            if (env.data.time - stopped_at >= 1. and info["success"]
                    and info["left_foot_force"] > 50 and info["right_foot_force"] > 50):
                break
    assert controller.phase == "done"
    assert info["success"]
    assert info["goal_distance"] < .12
    assert info["upright"] > .95
    assert info["left_foot_force"] > 50 and info["right_foot_force"] > 50
    env.close()
