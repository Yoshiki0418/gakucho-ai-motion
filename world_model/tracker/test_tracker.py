"""Regression tests for the full-body tracker pipeline (no GPU needed).

Run with the tracker venv (it has JAX on CPU)::

    JAX_PLATFORMS=cpu .runtime/world-model/mjx-venv/bin/python -m pytest -q world_model/tracker
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from . import body as B
from .export_bvh import physics_to_bvh
from .motion_library import DEFAULT_SOURCE
from .path_reference import follow, kinematics, random_bend, walking_cycle
from .retarget import continuous_euler, retarget_bvh

WALK = DEFAULT_SOURCE / "0001/01/001/001/001.bvh"
needs_mocap = pytest.mark.skipif(not WALK.exists(), reason="Custom-T2M-Lab mocap not available")


@pytest.fixture(scope="module")
def model():
    return B.build_model()


@pytest.fixture(scope="module")
def walk(model):
    return retarget_bvh(WALK.read_text(), model, B.load_profile())


def test_body_matches_gakucho_profile(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body_mass.sum() == pytest.approx(B.TOTAL_MASS, abs=1e-3)  # 6-digit densities
    head = model.geom("head").id
    assert data.geom_xpos[head, 2] + model.geom_size[head, 0] == pytest.approx(1.825, abs=2e-3)
    for name in ("left_foot", "right_foot"):
        g = model.geom(name).id
        bottom = data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]).dot(model.geom_size[g])
        assert abs(bottom) < 3e-3
    assert model.nu == len(B.JOINT_NAMES) == 63


def test_continuous_euler_has_no_flips():
    """The middle angle crosses pi/2, where plain decomposition switches branch."""
    from scipy.spatial.transform import Rotation
    k = np.linspace(0., 1., 50)
    true = np.stack([k, 1.25 + .7 * k, -k], axis=1)
    for order in ("XYZ", "ZXY", "ZYX", "XZY"):
        rotation = Rotation.from_euler(order, true)
        plain = rotation.as_euler(order)
        assert np.abs(np.diff(plain, axis=0)).max() > 1.   # the naive branch flips
        angles = continuous_euler(rotation, order)
        assert np.abs(np.diff(angles, axis=0)).max() < .1
        assert np.allclose(Rotation.from_euler(order, angles).as_matrix(), rotation.as_matrix(), atol=1e-6)


@needs_mocap
def test_bvh_export_round_trip(model, walk):
    meta = walk["meta"]
    text = physics_to_bvh(WALK.read_text(), walk["qpos"].astype(np.float64),
                          root_scale=meta["rootScale"], ground_shift=meta["groundShift"])
    again = retarget_bvh(text, model, B.load_profile())
    assert np.abs(again["xpos"] - walk["xpos"]).max() < 1e-3


@needs_mocap
def test_route_following_reference(model, walk):
    qpos = walk["qpos"].astype(np.float64)
    route = np.asarray([[0., 0.], [.5, -1.2], [.5, -2.4], [0., -3.6]])
    reference, scale = follow(qpos, route, start_direction=-np.pi / 2)
    assert np.linalg.norm(reference[-1, :2] - route[-1]) < 1e-6
    assert np.linalg.norm(reference[0, :2] - route[0]) < .05
    assert .7 < scale < 1.3
    # repeating gait cycles must not add joint discontinuities
    assert np.abs(np.diff(reference[:, 7:], axis=0)).max() <= np.abs(np.diff(qpos[:, 7:], axis=0)).max() + .02
    start, period, score = walking_cycle(qpos)
    assert .7 * 50 <= period <= 1.6 * 50 and score > .8
    bent, _ = random_bend(qpos, np.random.default_rng(0), max_curvature=1.)
    assert np.allclose(bent[:, 7:], qpos[:, 7:])
    assert kinematics(model, bent)["xpos"].shape == walk["xpos"].shape


@needs_mocap
def test_jax_forward_kinematics_matches_mujoco(walk):
    pytest.importorskip("jax")
    import jax.numpy as jp
    from .mjx_tracking import TrackingEnv
    env = TrackingEnv.__new__(TrackingEnv)
    model = B.build_model()
    env._nb = len(B.SEGMENTS)
    env._parent = [int(model.body_parentid[1 + b]) - 1 for b in range(env._nb)]
    env._body_pos = jp.asarray(model.body_pos[1:1 + env._nb], dtype=jp.float32)
    env._axes = np.asarray([[model.jnt_axis[j] for j in range(model.body_jntadr[1 + b],
                                                                 model.body_jntadr[1 + b] + 3)]
                            for b in range(1, env._nb)], dtype=np.float32)
    env._depth = 7
    for t in (0, len(walk["qpos"]) // 2, len(walk["qpos"]) - 1):
        pos, quat = env._fk(jp.asarray(walk["qpos"][t]))
        assert np.abs(np.asarray(pos) - walk["xpos"][t]).max() < 1e-5
        same = np.minimum(np.abs(np.asarray(quat) - walk["xquat"][t]).max(-1),
                          np.abs(np.asarray(quat) + walk["xquat"][t]).max(-1))
        assert same.max() < 1e-5


def _pose(pelvis_z, head_z, feet_ahead, frames=100):
    """Reference body positions with the pelvis, head and both feet placed."""
    names = [getattr(s, "name", s) for s in B.SEGMENTS]
    xpos = np.zeros((frames, len(names), 3))
    xpos[:, names.index("pelvis"), 2] = pelvis_z
    xpos[:, names.index("head"), 2] = head_z
    for foot in ("left_foot", "right_foot"):
        xpos[:, names.index(foot), 0] = feet_ahead
    return xpos


def test_needs_seat_uses_posture():
    from .seats import needs_seat
    chair, stand = _pose(.58, 1.25, .4), _pose(.95, 1.7, 0.)
    floor, pushup, squat = _pose(.2, .8, .5), _pose(.55, .45, 1.), _pose(.5, 1.2, .05)
    assert needs_seat("立っている状態から、椅子に座る。", stand)
    # captured on a chair without saying so: fails within a second without a seat
    assert needs_seat("座っている状態から、見回す。", chair)
    assert needs_seat("合掌(いただきますとか)", chair)
    assert not needs_seat("床に座っている状態から、立ち上がる。", floor)
    assert not needs_seat("開脚体前屈(座った状態で足を開いて前に倒れる)", floor)
    assert not needs_seat("座っている状態から、祈る。", stand)   # never sits: no chair
    assert not needs_seat("立っている状態から、腕立て伏せをする。", pushup)
    assert not needs_seat("立っている状態から、しゃがむ。", squat)
    assert not needs_seat("立っている状態から、歩く。", stand)


def test_scene_bodies_do_not_change_the_humanoid():
    # The parked seat box once took 41 kg of the 64.7 kg (the humanoid weighed 23.5 kg).
    from .mjx_tracking import seat_pose, seat_xml
    plain, seated = B.build_model(), B.build_model(extra_world=[seat_xml()])
    pelvis = plain.body("pelvis").id
    assert seated.body_subtreemass[pelvis] == pytest.approx(B.TOTAL_MASS)
    assert np.allclose(plain.body_mass[1:len(B.SEGMENTS) + 1], seated.body_mass[1:len(B.SEGMENTS) + 1])
    runs = []
    for model in (plain, seated):
        data = mujoco.MjData(model)
        data.qpos[2] = 1.
        if model.nmocap:
            data.mocap_pos[0], data.mocap_quat[0] = seat_pose(None)
        data.ctrl[:] = model.qpos0[7:]
        for _ in range(100):
            mujoco.mj_step(model, data)
        runs.append(data.qpos.copy())
    assert np.allclose(runs[0], runs[1], atol=1e-9)
