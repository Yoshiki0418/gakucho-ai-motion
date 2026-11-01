"""Real collision intersections, pinhole geometry and sensor-state purity."""
from types import SimpleNamespace
import base64
import copy
import math

import mujoco
import numpy as np
import pytest

from .depth_perception import DepthCameraConfig, capture_depth, camera_contract


def _world(geoms="", asset=""):
    xml = f'''<mujoco><option gravity="0 0 0"/><asset>{asset}</asset><worldbody>
      <body name="pelvis" pos="0 0 1"><freejoint/>
        <geom name="self" type="sphere" size=".1" mass="1"/>
        <body name="head" pos="0 0 .6"><geom name="head_self" type="sphere" size=".2" mass="1"/></body>
        <body name="hand" pos="0 -.7 .6"><geom name="hand_self" type="sphere" size=".15" mass="1"/></body>
      </body>{geoms}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, data=data, pelvis_body_id=model.body("pelvis").id)


def _valid(metadata):
    bits = np.frombuffer(base64.b64decode(metadata["validPixelMaskBase64"]), dtype=np.uint8)
    return np.unpackbits(bits, bitorder="little")[:metadata["pixelRayCount"]].reshape(metadata["height"], metadata["width"]).astype(bool)


def _meters(image, config):
    return config.near_meters+image*(config.far_meters-config.near_meters)


def test_actual_front_surface_occludes_rear_geometry_and_entire_self_is_filtered():
    env = _world('<geom name="front" type="box" pos="0 -2 1.6" size=".5 .1 .5"/>'
                 '<geom name="behind" type="box" pos="0 -3 1.6" size="1 .1 1"/>')
    config = DepthCameraConfig(width=1, height=1, down_pitch_degrees=0.)
    image, meta = capture_depth(env, config)
    assert image.shape == (1, 1, 1) and image.dtype == np.float32
    assert _meters(image, config)[0, 0, 0] == pytest.approx(1.9, abs=2e-7)
    assert meta["validPixelCount"] == 1 and _valid(meta)[0, 0]
    assert meta["eligibleCollisionGeomCount"] == 2


def test_one_real_ray_per_pixel_has_pinhole_projection_and_radial_range():
    env = _world('<geom name="small" type="box" pos=".4 -2 1.6" size=".05 .01 .05"/>')
    config = DepthCameraConfig(width=9, height=7, down_pitch_degrees=0.)
    image, meta = capture_depth(env, config)
    assert meta["pixelRayCount"] == 63
    assert np.argwhere(_valid(meta)).tolist() == [[3, 5]]
    assert _meters(image, config)[0, 3, 5] == pytest.approx(1.99*math.sqrt(1+(1/4.5)**2), abs=3e-7)
    assert np.count_nonzero(image < 1) == 1
    assert meta["imageResizedOrUpsampled"] is False


def test_mesh_intersection_uses_triangular_surface_not_its_bounding_box():
    asset = '<mesh name="wedge" vertex="-.5 -2 .5 .5 -2 .5 -.5 -2 2.5 -.5 -2.1 .5 .5 -2.1 .5 -.5 -2.1 2.5" face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>'
    env = _world('<geom type="mesh" mesh="wedge"/>', asset)
    config = DepthCameraConfig(width=9, height=1, down_pitch_degrees=0.)
    image, meta = capture_depth(env, config)
    valid = _valid(meta)
    assert valid[0, 3]
    assert not valid[0, 4]  # Center is inside the AABB but outside the wedge.
    assert image[0, 0, 4] == 1.


def test_finite_fov_cannot_observe_side_object_until_actual_body_yaw_changes():
    env = _world('<geom type="box" pos="2 0 1.6" size=".1 .5 .5"/>')
    config = DepthCameraConfig(width=9, height=7, down_pitch_degrees=0.)
    before, meta = capture_depth(env, config)
    assert np.all(before == 1.) and meta["validPixelCount"] == 0
    # Only test setup changes the actual world. The sensor does not move it.
    env.data.qpos[3:7] = [math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)]
    mujoco.mj_forward(env.model, env.data)
    after, meta = capture_depth(env, config)
    assert _meters(after, config)[0, 3, 4] == pytest.approx(1.9, abs=2e-7)
    np.testing.assert_allclose(meta["cameraForwardWorld"], [1, 0, 0], atol=3e-16)


def test_floor_pitch_near_limit_and_nohit_are_distinct_with_validity():
    floor = _world('<geom type="plane" size="5 5 .1"/>')
    config = DepthCameraConfig(width=1, height=1, down_pitch_degrees=30.)
    image, meta = capture_depth(floor, config)
    assert _meters(image, config)[0, 0, 0] == pytest.approx(3.2, abs=3e-7)
    assert _valid(meta)[0, 0]
    empty, meta = capture_depth(_world(), config)
    assert empty.item() == 1. and not _valid(meta)[0, 0]
    near = _world('<geom type="box" pos="0 -.03 1.6" size=".01 .01 .01"/>')
    image, meta = capture_depth(near, DepthCameraConfig(width=1, height=1, down_pitch_degrees=0.))
    assert image.item() == 0 and _valid(meta)[0, 0] and meta["nearClippedPixelCount"] == 1


def test_noncolliding_visual_geom_is_filtered_but_transparent_collider_occludes():
    env = _world('<geom type="box" pos="0 -1 1.6" size=".2 .1 .2" contype="0" conaffinity="0"/>'
                 '<geom name="transparent_collider" type="box" pos="0 -2 1.6" size=".2 .1 .2" rgba="1 1 1 0"/>')
    config = DepthCameraConfig(width=1, height=1, down_pitch_degrees=0.)
    image, meta = capture_depth(env, config)
    assert _meters(image, config).item() == pytest.approx(1.9, abs=2e-7)
    assert meta["eligibleCollisionGeomCount"] == 1
    assert env.model.geom_rgba[env.model.geom("transparent_collider").id, 3] == 0.


def test_batch_and_individual_actual_rays_produce_identical_full_depth_image():
    env = _world('<geom type="plane" size="5 5 .1"/><geom type="box" pos=".2 -2 1" size=".6 .3 1"/>')
    config = DepthCameraConfig()
    batch, batch_meta = capture_depth(env, config)
    individual, individual_meta = capture_depth(env, config, backend="single_ray")
    np.testing.assert_array_equal(batch, individual)
    assert batch_meta["validPixelMaskBase64"] == individual_meta["validPixelMaskBase64"]
    assert batch_meta["pixelRayCount"] == individual_meta["pixelRayCount"] == 3072


def test_actual_head_mount_moves_with_pitch_but_camera_heading_stays_horizontal():
    env = _world()
    env.data.qpos[3:7] = [math.cos(.2), math.sin(.2), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    config = DepthCameraConfig(mount_offset_local=(0., -.03, .02))
    _, meta = capture_depth(env, config)
    head = env.model.body("head").id
    expected = env.data.xpos[head]+env.data.xmat[head].reshape(3,3)@config.mount_offset_local
    np.testing.assert_allclose(meta["cameraOriginWorldMeters"], expected, rtol=0, atol=0)
    assert meta["cameraForwardWorld"][2] == pytest.approx(-math.sin(math.radians(15)))


def test_capture_and_filter_cache_do_not_write_any_physics_or_original_model():
    from .environment import HumanoidRoomEnv
    env = HumanoidRoomEnv()
    try:
        # Null/inactive solver pointers are exposed by MuJoCo as fresh,
        # uninitialized NumPy-owned arrays. They are not simulation storage.
        # Inspect every actual array view, including BVH scratch/constraints.
        data_fields = [name for name in dir(env.data)
                       if isinstance((array := getattr(env.data,name)),np.ndarray) and not array.flags.owndata]
        model_fields = ("geom_group", "geom_rgba", "geom_pos", "geom_quat", "geom_size", "geom_contype",
                        "geom_conaffinity", "body_mass", "body_inertia", "geom_friction", "mesh_vert")
        before_data = {key: getattr(env.data,key).copy() for key in data_fields}
        before_model = {key: getattr(env.model,key).copy() for key in model_fields}
        snapshot = copy.deepcopy(env.snapshot())
        timing = (float(env.data.time), env._step_count, env.data.ncon)
        for _ in range(3): capture_depth(env)
        for key,value in before_data.items(): np.testing.assert_array_equal(getattr(env.data,key),value,err_msg=key)
        for key,value in before_model.items(): np.testing.assert_array_equal(getattr(env.model,key),value,err_msg=key)
        assert timing == (float(env.data.time), env._step_count, env.data.ncon)
        assert env.snapshot() == snapshot
    finally: env.close()


def test_filter_cache_tracks_in_place_geometry_size_changes_without_stale_surfaces():
    env = _world('<geom name="box" type="box" pos="0 -2 1.6" size=".5 .1 .5"/>')
    config = DepthCameraConfig(width=1,height=1,down_pitch_degrees=0.)
    first,_ = capture_depth(env,config)
    env.model.geom_size[env.model.geom("box").id,1] = .3
    mujoco.mj_forward(env.model,env.data)
    second,_ = capture_depth(env,config)
    assert _meters(first,config).item() == pytest.approx(1.9,abs=2e-7)
    assert _meters(second,config).item() == pytest.approx(1.7,abs=2e-7)


@pytest.mark.parametrize("changes", [{"width":True},{"height":0},{"horizontal_fov_degrees":360},
    {"near_meters":0},{"far_meters":.01},{"down_pitch_degrees":90},{"mount_offset_local":(0,float("nan"),0)}])
def test_invalid_camera_configuration_rejects(changes):
    with pytest.raises(ValueError): DepthCameraConfig(**changes)


def test_camera_contract_discloses_range_not_z_and_no_ideal_360_sensor():
    contract = camera_contract(DepthCameraConfig())
    assert contract["shape"] == [1,48,64] and contract["noHitValue"] == 1.
    assert contract["ideal360RaysExposed"] is False and contract["worldStateWrites"] is False
    assert "not optical-axis Z" in contract["depthMetric"]
