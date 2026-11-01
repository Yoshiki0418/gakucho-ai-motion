"""Physics and initialization gates for references, independent of PPO reward."""

import ast
import inspect
import textwrap

import mujoco
import numpy as np
import pytest

from .environment import HumanoidRoomEnv
from .home_scene import create_seated_task_scene
from .imitation import ReferenceHumanoidEnv


def _static_contact_forces(env):
    """Actual solver forces on the dynamic body, expressed in world axes."""
    result = []
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        ids = [int(contact.geom1), int(contact.geom2)]
        bodies = [int(env.model.geom_bodyid[geom]) for geom in ids]
        if (bodies[0] == 0) == (bodies[1] == 0):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, index, local)
        sign = 1 if bodies[0] == 0 else -1
        force = sign * (contact.frame.reshape(3, 3).T @ local[:3])
        normal = sign * contact.frame[:3]
        result.append({"ids": ids, "bodies": bodies, "force": force,
                       "normal": normal, "position": contact.pos.copy(),
                       "normal_force": float(local[0])})
    return result


@pytest.fixture(scope="module")
def seated_baseline():
    scene = create_seated_task_scene()
    env = ReferenceHumanoidEnv(task="sit", scene=scene, reference_start=False,
                               randomize=False, max_episode_seconds=6)
    _, initial = env.reset(seed=71391)
    initial_contacts = []
    for contact in env.data.contact[:env.data.ncon]:
        names = [env.model.geom(int(contact.geom1)).name, env.model.geom(int(contact.geom2)).name]
        initial_contacts.append((names, float(contact.dist)))
    fell = False
    supported_steps = 0
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.zeros(20))
        fell |= terminated
        supported_steps += int(info["pelvis_support_contact"])
        if terminated or truncated:
            break
    yield env, scene, initial, initial_contacts, info, fell, supported_steps
    env.close()


def test_standing_start_has_no_initial_chair_body_overlap(seated_baseline):
    _, scene, initial, contacts, _, _, _ = seated_baseline
    assert initial["root_height"] == pytest.approx(.93)
    assert not initial["pelvis_support_contact"]
    assert scene["goals"][0]["position"][1] == pytest.approx(.30)
    # Internal adjacent anatomical contacts are separate from the furniture.
    assert not any("training-chair/" in name and distance < 0
                   for names, distance in contacts for name in names)


def test_sitting_is_supported_by_actual_seat_top_and_feet(seated_baseline):
    env, scene, _, _, info, fell, supported_steps = seated_baseline
    assert not fell
    assert info["time"] == pytest.approx(6, abs=1e-10)
    assert info["success"] and info["pelvis_support_contact"]
    assert supported_steps > 100
    assert info["left_foot_force"] > 50 and info["right_foot_force"] > 50
    seat_id = env.model.geom("training-chair/seat").id
    seat_top = env.data.geom_xpos[seat_id, 2] + env.model.geom_size[seat_id, 2]
    assert seat_top == pytest.approx(scene["goals"][0]["position"][2], abs=1e-9)
    pelvis_seat = [contact for contact in _static_contact_forces(env)
                   if seat_id in contact["ids"] and env.pelvis_body_id in contact["bodies"]]
    assert pelvis_seat
    assert sum(contact["force"][2] for contact in pelvis_seat) > 100
    for contact in pelvis_seat:
        assert contact["normal"][2] > .98  # Support from above the seat, not its side.
        assert contact["normal_force"] > 1
        assert contact["position"][2] == pytest.approx(seat_top, abs=.005)
    assert seat_top + .06 < env.data.qpos[2] < seat_top + .14
    all_support = sum(contact["force"][2] for contact in _static_contact_forces(env))
    weight = float(env.model.body_mass.sum() * -env.model.opt.gravity[2])
    assert all_support == pytest.approx(weight, rel=.01)


def test_floor_pelvis_contact_is_not_reported_as_a_chair_seat():
    env = ReferenceHumanoidEnv(task="sit", reference_start=False)
    env.reset(seed=9)
    # A deliberately fallen test initialization creates real floor/pelvis
    # contact. It must not pass the chair support flag or the fall gate.
    env.data.qpos[2] = .50
    env.data.qpos[3:7] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
    env.data.ctrl[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(500):
        forces = _static_contact_forces(env)
        if any(env.pelvis_body_id in contact["bodies"] and contact["normal_force"] > 1
               for contact in forces):
            break
        mujoco.mj_step(env.model, env.data)
    else:
        pytest.fail("Unpowered body never established actual pelvis/floor contact")
    info = env._info()
    assert not info["pelvis_support_contact"]
    assert info["fall"] and not info["success"]
    env.close()


def test_reference_start_is_an_initial_condition_distinct_from_standing_start():
    reference = ReferenceHumanoidEnv(task="walk", reference_start=True, randomize=False)
    standing = ReferenceHumanoidEnv(task="walk", reference_start=False, randomize=False)
    reference.reset(seed=91327)
    observation, info = standing.reset(seed=91327)
    assert info["root_height"] == pytest.approx(.93)
    np.testing.assert_array_equal(standing.data.qvel, np.zeros(standing.model.nv))
    np.testing.assert_array_equal(standing.data.qpos[standing._qpos_indices], standing.standing_pose)
    assert np.linalg.norm(reference.data.qvel[:2]) > .25
    assert np.linalg.norm(reference.data.qpos[reference._qpos_indices] - standing.standing_pose) > .2
    assert observation.shape == (59,) and standing.observation_space.contains(observation)
    _, _, _, _, info = standing.step(np.zeros(20))
    assert info["reference_start"] is False
    _, _, _, _, info = reference.step(np.zeros(20))
    assert info["reference_start"] is True
    assert reference.data.time == pytest.approx(.02, abs=1e-12)
    reference.close()
    standing.close()


def test_reference_phase_reset_is_seeded_and_observation_phase_advances():
    env = ReferenceHumanoidEnv(task="walk", reference_start=True, randomize=True)
    first, _ = env.reset(seed=82631)
    qpos = env.data.qpos.copy()
    phase = env.phase_offset
    env.reset(seed=4)
    again, _ = env.reset(seed=82631)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(qpos, env.data.qpos)
    assert env.phase_offset == phase
    after, _, _, _, _ = env.step(np.zeros(20))
    assert after.shape == (59,) and env.observation_space.contains(after)
    assert after[-1] == pytest.approx((float(first[-1]) + .02 / env.reference.duration) % 1, abs=1e-6)
    env.close()


def test_step_controls_torques_without_pose_or_mocap_assignments():
    for cls in (HumanoidRoomEnv, ReferenceHumanoidEnv):
        source = ast.parse(textwrap.dedent(inspect.getsource(cls.step)))
        for node in ast.walk(source):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                assert not any(isinstance(part, ast.Attribute) and part.attr in ("qpos", "mocap_pos", "mocap_quat")
                               for part in ast.walk(target))
    env = ReferenceHumanoidEnv(task="walk", reference_start=False)
    assert env.model.nmocap == 0
    assert env.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert 0 not in env.model.actuator_trnid[:, 0]
    assert all(env.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_PLANE for geom in range(env.model.ngeom))
    env.close()
