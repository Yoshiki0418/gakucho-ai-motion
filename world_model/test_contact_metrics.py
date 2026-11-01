"""Physical contact-point velocity oracles and read-only sensor checks."""

import mujoco
import numpy as np
import pytest

from .contact_metrics import measure_foot_contact_slip
from .environment import HumanoidRoomEnv


_XML = """<mujoco><option timestep=".002"/><worldbody>
<geom name="floor" type="box" pos="0 0 -.1" size="2 2 .1"/>
<body name="left" pos=".2 0 .034"><freejoint/>
  <geom name="humanoid_left_foot" type="box" size=".055 .135 .035" mass="20"/>
</body>
<body name="right" pos="-.2 0 .034"><freejoint/>
  <geom name="humanoid_right_foot" type="box" size=".055 .135 .035" mass="20"/>
</body>
</worldbody></mujoco>"""


def _loaded_feet():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_known_translation_matches_actual_contact_velocity_and_jacobian():
    model, data = _loaded_feet()
    data.qvel[0] = .25
    mujoco.mj_forward(model, data)
    result = measure_foot_contact_slip(model, data, verify_jacobian=True)
    assert result["left"]["loaded"]
    assert result["left"]["contacts"]
    assert result["left"]["weightedSlipSpeedMetersPerSecond"] == pytest.approx(.25, abs=1e-12)
    assert result["right"]["weightedSlipSpeedMetersPerSecond"] == pytest.approx(0., abs=1e-12)
    assert max(contact["jacobianOracleErrorMetersPerSecond"]
               for contact in result["left"]["contacts"]) < 1e-12


def test_fixed_contact_edge_does_not_report_rolling_as_slip():
    model, data = _loaded_feet()
    angle = .30
    data.qpos[2] = np.sin(angle)*.135 + np.cos(angle)*.035 - .001
    data.qpos[3:7] = [np.cos(angle/2), np.sin(angle/2), 0., 0.]
    mujoco.mj_forward(model, data)
    geom_id = model.geom("humanoid_left_foot").id
    contacts = [contact for contact in data.contact
                if geom_id in (contact.geom1, contact.geom2)]
    assert len(contacts) == 2
    point = contacts[0].pos.copy()
    centre = data.geom_xpos[geom_id].copy()
    omega = np.array([1., 0., 0.])
    # This velocity rotates around the actual contact edge. The geom centre
    # moves while both contact points (different X, same Y/Z) remain fixed.
    data.qvel[:3] = -np.cross(omega, point-centre)
    data.qvel[3:6] = omega
    mujoco.mj_forward(model, data)
    result = measure_foot_contact_slip(model, data, verify_jacobian=True)["left"]
    assert np.linalg.norm(result["centreVelocity"]) > .1
    assert result["contacts"]
    assert result["weightedSlipSpeedMetersPerSecond"] < 1e-12
    assert max(contact["jacobianOracleErrorMetersPerSecond"]
               for contact in result["contacts"]) < 1e-12


def test_missing_invalid_geom_ids_and_threshold_are_rejected():
    model, data = _loaded_feet()
    for ids in ((-1, 0), (1, 1), (1,), (1, model.ngeom)):
        with pytest.raises(ValueError):
            measure_foot_contact_slip(model, data, ids)
    with pytest.raises(ValueError):
        measure_foot_contact_slip(model, data, minimum_vertical_force=float("nan"))


def test_unloaded_foot_has_no_fake_contact_or_load():
    model, data = _loaded_feet()
    data.qpos[2] = 1.
    mujoco.mj_forward(model, data)
    result = measure_foot_contact_slip(model, data)["left"]
    assert not result["loaded"]
    assert result["contacts"] == []
    assert result["verticalForceNewtons"] == 0.


def test_actual_humanoid_measurement_preserves_physics_state_and_sensor_force():
    env = HumanoidRoomEnv(task="stand", randomize=False, max_episode_seconds=2)
    try:
        env.reset(seed=17)
        for _ in range(50):
            env.step(np.zeros(20))
        arrays = ("qpos", "qvel", "ctrl", "qacc", "qfrc_applied", "xfrc_applied",
                  "geom_xpos", "geom_xmat", "efc_force")
        before = {key: getattr(env.data, key).copy() for key in arrays}
        contacts = [(contact.dist, contact.geom1, contact.geom2,
                     contact.pos.copy(), contact.frame.copy()) for contact in env.data.contact]
        mass = env.model.body_mass.copy()
        friction = env.model.geom_friction.copy()
        time = env.data.time
        result = measure_foot_contact_slip(env.model, env.data, env._foot_geom_ids,
                                           verify_jacobian=True)
        assert env.data.time == time
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(env.data, key), value)
        np.testing.assert_array_equal(env.model.body_mass, mass)
        np.testing.assert_array_equal(env.model.geom_friction, friction)
        assert env.data.ncon == len(contacts)
        for contact, old in zip(env.data.contact, contacts):
            assert (contact.dist, contact.geom1, contact.geom2) == old[:3]
            np.testing.assert_array_equal(contact.pos, old[3])
            np.testing.assert_array_equal(contact.frame, old[4])
        for side in ("left", "right"):
            assert result[side]["loaded"]
            assert result[side]["verticalForceNewtons"] == pytest.approx(
                env._info()[f"{side}_foot_force"], abs=1e-9)
            assert max(contact["jacobianOracleErrorMetersPerSecond"]
                       for contact in result[side]["contacts"]) < 1e-12
    finally:
        env.close()
