"""Independent dynamics checks, rather than checks of the controller's targets."""

import mujoco
import numpy as np
import pytest


def simulate(xml, seconds):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    initial = data.qpos.copy()
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return model, data, initial


def test_free_fall_matches_gravity_without_position_correction():
    model, data, initial = simulate('''<mujoco><option timestep="0.001" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 3"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
      </worldbody></mujoco>''', .2)
    assert data.qpos[2] == pytest.approx(initial[2] - .5 * 9.81 * .2**2, abs=.0011)
    assert data.qvel[2] == pytest.approx(-9.81 * .2, abs=1e-9)
    assert data.ncon == 0


def test_finite_floor_supports_body_and_does_not_support_outside():
    def xml(x):
        return f'''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
          <worldbody><geom name="floor" type="box" pos="0 0 -.1" size="1 1 .1"/>
          <body pos="{x} 0 1"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
          </worldbody></mujoco>'''
    _, supported, _ = simulate(xml(0), 1)
    _, outside, _ = simulate(xml(2), 1)
    assert supported.ncon > 0
    assert supported.qpos[2] == pytest.approx(.1, abs=.002)
    assert outside.qpos[2] < -3
    assert outside.ncon == 0


def test_contact_friction_changes_sliding_distance():
    def slide(friction):
        xml = f'''<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
          <default><geom friction="{friction} .005 .0001"/></default><worldbody>
          <geom type="box" pos="0 0 -.1" size="10 10 .1"/>
          <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1" mass="2"/>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qvel[0] = 1
        for _ in range(1000):
            mujoco.mj_step(model, data)
        return data.qpos[0]
    slippery, rough = slide(.005), slide(.8)
    assert slippery > .8
    assert 0 < rough < .2
    assert slippery > 5 * rough


def test_zero_gravity_does_not_fall():
    _, data, initial = simulate('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body pos="0 0 3"><freejoint/><geom type="sphere" size=".1" mass="2"/>
      </body></worldbody></mujoco>''', .5)
    np.testing.assert_array_equal(data.qpos, initial)
