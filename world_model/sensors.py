"""Physics-state perception for future furniture/terrain policies.

This uses the actual MuJoCo collision scene, not a camera, a learned world
network, or a replacement floor. It never advances physics or changes poses,
controls, forces, masses, or collision masks. ``geom_group`` is configured only
for ray/render filtering: static scenery is group 0 and the entire own rig is 1.

The existing 59 observations remain the prefix. The standard wrapper appends
51 bounded values: body/knee ranges 32, relative surface heights 9, validity 9,
and head clearance 1. Thus a 59-input policy cannot be loaded as a 110-input
sensor policy without new training or an explicit separate policy adapter.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .humanoid import STANDING_ROOT_HEIGHT

RAY_COUNT = 16
RAY_RANGE_METERS = 5.0
HEIGHT_RANGE_METERS = 5.0
HEADROOM_RANGE_METERS = 3.0
FLOOR_OFFSETS_METERS = (-0.4, 0.0, 0.4)
SENSOR_FEATURE_COUNT = 51
_GROUP_MASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
_EPS = 1e-8


def _base(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _configure_groups(base) -> tuple[int, np.ndarray]:
    model = base.model
    root = int(getattr(base, "pelvis_body_id", getattr(base, "root_body_id", -1)))
    if root < 1:
        try:
            root = model.body("pelvis").id
        except KeyError as error:
            raise ValueError("scene sensors require the own dynamic root body") from error
    own = np.zeros(model.nbody, dtype=bool)
    own[root] = True
    for body in range(1, model.nbody):
        if own[int(model.body_parentid[body])]:
            own[body] = True
    model.geom_group[:] = np.where(own[model.geom_bodyid], 1, 0)
    return root, np.flatnonzero(model.geom_group == 0)


def _ray(base, origin: np.ndarray, direction: np.ndarray, limit: float) -> tuple[float, int | None]:
    geom_id = np.array([-1], dtype=np.int32)
    distance = float(mujoco.mj_ray(base.model, base.data,
                                 np.asarray(origin, dtype=np.float64),
                                 np.asarray(direction, dtype=np.float64),
                                 _GROUP_MASK, True, -1, geom_id))
    if not math.isfinite(distance) or distance < 0 or distance > limit:
        return limit, None
    return max(0.0, distance), int(geom_id[0])


def _mesh_vertices(base, geom_id: int) -> np.ndarray:
    model = base.model
    mesh = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)


def _inside_static_top(base, geom_id: int, point: np.ndarray) -> float | None:
    """Lift only a ray origin out of real overlapping solid geometry, never an actor."""
    model, data = base.model, base.data
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    centre = data.geom_xpos[geom_id]
    local = rotation.T @ (point - centre)
    size, kind = model.geom_size[geom_id], int(model.geom_type[geom_id])
    inside = False
    extents = size
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        inside = bool(np.all(np.abs(local) < size - _EPS))
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        inside = np.linalg.norm(local) < size[0] - _EPS
        extents = np.full(3, size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        inside = np.linalg.norm(local / size) < 1 - _EPS
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        inside = math.hypot(local[0], local[1]) < size[0] - _EPS and abs(local[2]) < size[1] - _EPS
        extents = np.array([size[0], size[0], size[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        distance = np.linalg.norm([local[0], local[1], max(0.0, abs(local[2]) - size[1])])
        inside = distance < size[0] - _EPS
        extents = np.array([size[0], size[0], size[0] + size[1]])
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        vertices = _mesh_vertices(base, geom_id)
        if not np.all(local >= vertices.min(axis=0) - _EPS) or not np.all(local <= vertices.max(axis=0) + _EPS):
            return None
        mesh = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        faces = np.asarray(model.mesh_face[start:start + count], dtype=np.int64)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        signed = np.einsum("ij,ij->i", normals, local - triangles[:, 0])
        inside = bool(np.all(signed <= _EPS) or np.all(signed >= -_EPS))
        if inside:
            return float((vertices @ rotation.T + centre)[:, 2].max())
        return None
    if inside:
        return float(centre[2] + np.abs(rotation[2]) @ extents)
    return None


def _safe_down_origin(base, xy: np.ndarray, height: float, static: np.ndarray) -> tuple[np.ndarray, bool]:
    origin = np.array([xy[0], xy[1], height], dtype=np.float64)
    lifted = False
    # Each lift rises above at least one containing object; bounded by the geom count.
    for _ in range(len(static) + 1):
        tops = [_inside_static_top(base, int(g), origin) for g in static]
        tops = [z for z in tops if z is not None]
        if not tops:
            return origin, lifted
        origin[2] = max(tops) + 1e-4
        lifted = True
    raise ValueError("could not find a ray origin outside the physical scene solids")


def _body_basis(base, root: int) -> tuple[np.ndarray, np.ndarray, float]:
    rotation = base.data.xmat[root].reshape(3, 3)
    projected_forward = -rotation[:2, 1]
    if np.linalg.norm(projected_forward) > 1e-7:
        yaw = math.atan2(projected_forward[0], -projected_forward[1])
    else:
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c, s]), np.array([s, -c]), yaw


def _head_origin(base, root: int, pelvis: np.ndarray) -> np.ndarray:
    try:
        geom = base.model.geom("humanoid_head").id
        position = base.data.geom_xpos[geom].copy()
        position[2] += float(base.model.geom_size[geom, 0]) + 1e-4
        return position
    except KeyError:
        return pelvis + np.array([0.0, 0.0, 0.78])


def measure_scene(env) -> dict[str, Any]:
    """Measure the current already-forwarded physics state without stepping it.

    Floor samples return the nearest actual surface below each root-local sample.
    Values are relative to pelvis height and keep a separate valid bit; a missing
    surface is not replaced by an assumed ground plane. On a missing centre
    surface, horizontal rays use a declared nominal body-height reference only.
    """
    base = _base(env)
    root, static = _configure_groups(base)
    pelvis = np.asarray(base.data.xpos[root], dtype=np.float64).copy()
    if not np.isfinite(pelvis).all():
        raise ValueError("non-finite physical root state")
    lateral, forward, yaw = _body_basis(base, root)
    heights: list[float | None] = []
    valid: list[bool] = []
    floor_ids: list[int | None] = []
    samples, offsets, origins, lifted_flags = [], [], [], []
    for forward_offset in FLOOR_OFFSETS_METERS:
        for lateral_offset in FLOOR_OFFSETS_METERS:
            xy = pelvis[:2] + lateral * lateral_offset + forward * forward_offset
            origin, lifted = _safe_down_origin(base, xy, float(pelvis[2] + 0.2), static)
            distance, geom_id = _ray(base, origin, np.array([0.0, 0.0, -1.0]), HEIGHT_RANGE_METERS)
            height = float(origin[2] - distance) if geom_id is not None else None
            heights.append(height)
            valid.append(geom_id is not None)
            floor_ids.append(geom_id)
            samples.append(xy.tolist())
            offsets.append([lateral_offset, forward_offset])
            origins.append(origin.tolist())
            lifted_flags.append(lifted)
    centre_height = heights[4]
    reference_height = centre_height if centre_height is not None else float(pelvis[2] - STANDING_ROOT_HEIGHT)
    reference_kind = "measured_surface" if centre_height is not None else "nominal_body_height_missing_surface"
    ranges, horizontal_ids, horizontal_origins = [], [], []
    for above_floor in (0.8, 0.4):
        origin = np.array([pelvis[0], pelvis[1], reference_height + above_floor])
        layer, layer_ids = [], []
        horizontal_origins.append(origin.tolist())
        containing = next((int(g) for g in static if _inside_static_top(base, int(g), origin) is not None), None)
        for i in range(RAY_COUNT):
            angle = i * 2 * math.pi / RAY_COUNT
            direction = math.cos(angle) * forward + math.sin(angle) * lateral
            distance, geom_id = _ray(base, origin, np.array([*direction, 0.0]), RAY_RANGE_METERS)
            # An origin in a real obstacle is an immediate obstruction, not its far exit face.
            if containing is not None:
                distance = 0.0
                geom_id = containing
            layer.append(distance)
            layer_ids.append(geom_id)
        ranges.append(layer)
        horizontal_ids.append(layer_ids)
    head_origin = _head_origin(base, root, pelvis)
    headroom, headroom_id = _ray(base, head_origin, np.array([0.0, 0.0, 1.0]), HEADROOM_RANGE_METERS)
    if any(_inside_static_top(base, int(g), head_origin) is not None for g in static):
        headroom = 0.0
    relative_heights = [float(np.clip((height - pelvis[2]) / HEIGHT_RANGE_METERS, -1, 1))
                        if height is not None else -1.0 for height in heights]
    features = ([distance / RAY_RANGE_METERS for layer in ranges for distance in layer]
                + relative_heights + [float(flag) for flag in valid]
                + [headroom / HEADROOM_RANGE_METERS])
    assert len(features) == SENSOR_FEATURE_COUNT
    return {
        "perception": "actual_physics_state_rays", "sim_time": float(base.data.time),
        "feature_count": SENSOR_FEATURE_COUNT, "features": features,
        "feature_order": "body_ranges16,knee_ranges16,surface_height_relative9,surface_valid9,headroom1",
        "root_position_m": pelvis.tolist(),
        "body_basis": {"yaw_radians": yaw, "lateral_xy": lateral.tolist(), "forward_xy": forward.tolist()},
        "horizontal_ranges_m": ranges, "horizontal_hit_geom_ids": horizontal_ids,
        "horizontal_origins_m": horizontal_origins,
        "horizontal_reference_height_m": reference_height, "horizontal_reference": reference_kind,
        "floor_heights_m": heights, "floor_valid": valid, "missing_floor": [not flag for flag in valid],
        "floor_hit_geom_ids": floor_ids, "floor_sample_xy_m": samples,
        "floor_offsets_local_m": offsets, "floor_origins_m": origins, "floor_origin_lifted": lifted_flags,
        "missing_height_feature": -1.0, "height_feature_scale_m": HEIGHT_RANGE_METERS,
        "headroom_m": headroom, "headroom_origin_m": head_origin.tolist(), "headroom_hit_geom_id": headroom_id,
        "excluded_own_geom_count": int(np.count_nonzero(base.model.geom_group == 1)),
        "ray_group": 0, "own_rig_group": 1,
    }


class SceneSensorWrapper(gym.Wrapper):
    """Append bounded scene perception while preserving every base observation."""

    def __init__(self, env):
        super().__init__(env)
        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box) or len(base_space.shape) != 1:
            raise ValueError("SceneSensorWrapper needs a flat Box observation")
        self.observation_prefix_length = base_space.shape[0]
        sensor_low = np.array([0.0] * 32 + [-1.0] * 9 + [0.0] * 10, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate((base_space.low.astype(np.float32), sensor_low)),
            high=np.concatenate((base_space.high.astype(np.float32), np.ones(SENSOR_FEATURE_COUNT, dtype=np.float32))),
            dtype=np.float32,
        )
        _configure_groups(_base(env))

    def _append(self, observation, info):
        measurement = measure_scene(self.env)
        result = np.concatenate((np.asarray(observation, dtype=np.float32),
                                 np.asarray(measurement["features"], dtype=np.float32)))
        info = dict(info)
        info["scene_sensors"] = measurement
        info["scene_sensors"]["observation_prefix_length"] = self.observation_prefix_length
        return result, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._append(observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, info = self._append(observation, info)
        return observation, reward, terminated, truncated, info
