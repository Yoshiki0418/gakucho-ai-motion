"""Read-only first-person depth from the actual MuJoCo collision geometry.

Every pixel has its own pinhole ray. A private model copy holds the sensor's
visibility mask; neither the simulation model nor its MjData is modified.
No OpenGL renderer, image enlargement, scene graph map or 360-degree scan is
used. Distances are ray ranges, rather than optical-axis Z depths.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import math
import time
import weakref

import mujoco
import numpy as np


@dataclass(frozen=True)
class DepthCameraConfig:
    width: int = 64
    height: int = 48
    horizontal_fov_degrees: float = 90.
    near_meters: float = .05
    far_meters: float = 5.
    down_pitch_degrees: float = 15.
    mount_body: str = "head"
    mount_offset_local: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 512
               for value in (self.width, self.height)):
            raise ValueError("depth image width/height must be integers in 1..512")
        if (not math.isfinite(self.horizontal_fov_degrees)
                or not 0 < self.horizontal_fov_degrees < 180):
            raise ValueError("pinhole horizontal FOV must be finite and in (0,180) degrees")
        if not (math.isfinite(self.near_meters) and math.isfinite(self.far_meters)
                and 0 < self.near_meters < self.far_meters):
            raise ValueError("camera range must satisfy 0 < near < far")
        if not math.isfinite(self.down_pitch_degrees) or not -89 < self.down_pitch_degrees < 89:
            raise ValueError("camera down-pitch must be finite and in (-89,89) degrees")
        offset = tuple(float(value) for value in self.mount_offset_local)
        if len(offset) != 3 or not np.isfinite(offset).all():
            raise ValueError("camera mount offset requires three finite local coordinates")
        if not isinstance(self.mount_body, str) or not self.mount_body:
            raise ValueError("camera mount must name an actual MuJoCo body")
        object.__setattr__(self, "mount_offset_local", offset)


def camera_contract(config: DepthCameraConfig) -> dict:
    """The image semantics to pin in a visual policy's provenance."""
    focal = config.width/(2*math.tan(math.radians(config.horizontal_fov_degrees)/2))
    values = asdict(config)
    values["mount_offset_local"] = list(config.mount_offset_local)
    return {"version": 1, "config": values,
        "shape": [1, config.height, config.width], "dtype": "float32",
        "projection": "one normalized pinhole ray per pixel center; square pixels; no resizing",
        "pixelCoordinates": "row zero is image top; column zero is left",
        "verticalFovDegrees": math.degrees(2*math.atan(config.height/(2*focal))),
        "pose": "actual mount-body position plus local offset; horizontal pelvis heading, forward -Y, world +Z up, fixed down pitch",
        "geometry": "nearest actual MuJoCo collision geom surface; static and dynamic world geoms; own humanoid subtree excluded",
        "occlusion": "first visible collision surface only; no objects behind it exposed",
        "depthMetric": "Euclidean range along each unit pixel ray, meters; not optical-axis Z",
        "normalization": "(clip(range,near,far)-near)/(far-near)",
        "noHitValue": 1., "noHitValidity": "metadata validPixelMaskBase64; row-major np.packbits(bitorder=little)",
        "nearClipping": "first hit closer than near remains occluding and is clamped to zero",
        "ideal360RaysExposed": False, "rgbUsed": False,
        "worldStateWrites": False, "openglRequired": False}


@lru_cache(maxsize=32)
def _pixel_rays(config):
    focal = config.width/(2*math.tan(math.radians(config.horizontal_fov_degrees)/2))
    x = (np.arange(config.width, dtype=np.float64)+.5-config.width/2)/focal
    y = (config.height/2-np.arange(config.height, dtype=np.float64)-.5)/focal
    xx, yy = np.meshgrid(x, y)
    # Columns are camera right, forward, up. World conversion happens per
    # capture, using actual current body heading rather than a saved pose.
    rays = np.stack((xx, np.ones_like(xx), yy), axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays.setflags(write=False)
    return rays


_FILTER_MODELS = weakref.WeakKeyDictionary()


def _filtered_model(model, self_root):
    # In-place geometry edits invalidate the filter copy. Rigid-body movement
    # needs no copy/update: queries use the original, current data.geom_xpos
    # and geom_xmat. Mass/friction randomization does not affect optical hits.
    digest = hashlib.sha256()
    for field in ("geom_contype", "geom_conaffinity", "geom_type", "geom_size",
                  "geom_bodyid", "geom_dataid", "body_rootid", "mesh_vert", "mesh_face"):
        digest.update(np.ascontiguousarray(getattr(model, field)).tobytes())
    signature = (self_root, digest.digest())
    cached = _FILTER_MODELS.get(model)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2], cached[3]
    sensor_model = copy.copy(model)
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    own = model.body_rootid[model.geom_bodyid] == self_root if self_root > 0 else np.zeros(model.ngeom, bool)
    eligible = collision & ~own
    # This is a separate MjModel, never the physical model. Group filtering
    # can therefore exclude the entire articulated self, even when the
    # original humanoid and furniture all share the same geom group.
    sensor_model.geom_group[:] = eligible.astype(np.int32)
    sensor_model.geom_rgba[:, 3] = eligible.astype(np.float32)
    sensor_model.mat_rgba[:, 3] = 1.
    query_data = mujoco.MjData(sensor_model)
    _FILTER_MODELS[model] = (signature, sensor_model, query_data, int(eligible.sum()))
    return sensor_model, query_data, int(eligible.sum())


def _camera_pose(env, config):
    body = getattr(env, "unwrapped", env)
    model, data = body.model, body.data
    mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.mount_body)
    if mount < 1:
        raise ValueError(f"camera mount body {config.mount_body!r} is missing")
    pelvis = getattr(body, "pelvis_body_id", None)
    if pelvis is None:
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis < 1:
            pelvis = mount
    rotation = data.xmat[pelvis].reshape(3, 3)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    right = np.asarray([math.cos(heading), math.sin(heading), 0.])
    horizontal_forward = np.asarray([math.sin(heading), -math.cos(heading), 0.])
    pitch = math.radians(config.down_pitch_degrees)
    forward = math.cos(pitch)*horizontal_forward-np.asarray([0., 0., math.sin(pitch)])
    up = np.cross(forward, right)
    origin = data.xpos[mount]+data.xmat[mount].reshape(3, 3)@np.asarray(config.mount_offset_local)
    self_root = int(model.body_rootid[pelvis])
    return body, origin, np.stack((right, forward, up)), self_root


def _cast(model, data, origin, rays, cutoff, backend):
    geom_ids = np.full(len(rays), -1, dtype=np.int32)
    ranges = np.full(len(rays), -1., dtype=np.float64)
    groups = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    if backend == "multi_ray":
        arguments = (model, data, origin, np.ascontiguousarray(rays).reshape(-1), groups, True, -1, geom_ids, ranges)
        # MuJoCo versions before the optional-normal API use one fewer arg.
        if "normal:" in mujoco.mj_multiRay.__doc__:
            mujoco.mj_multiRay(*arguments, None, len(rays), cutoff)
        else:
            mujoco.mj_multiRay(*arguments, len(rays), cutoff)
    elif backend == "single_ray":
        for index, direction in enumerate(rays):
            ranges[index] = mujoco.mj_ray(model, data, origin, direction, groups, True, -1, geom_ids[index:index+1])
    else:
        raise ValueError("depth backend must be multi_ray or single_ray")
    return ranges, geom_ids


def capture_depth(env, config=DepthCameraConfig(), *, backend="multi_ray"):
    """Return normalized float32 [1,H,W] depth and JSON-ready sensor metadata.

    No-hit pixels are far (1), with a separate packed validity bitmap so a
    hit at exactly the range limit is distinguishable from unseen space.
    ``single_ray`` exists for equivalence/performance audits, not resizing.
    """
    started = time.perf_counter()
    body, origin, basis, self_root = _camera_pose(env, config)
    model, query_data, eligible_count = _filtered_model(body.model, self_root)
    # mj_multiRay takes a non-const mjData and uses its scratch/acceleration
    # buffers. Isolate even those changes from the simulation, rather than
    # only checking qpos/qvel after a sensor call.
    mujoco.mj_copyData(query_data, body.model, body.data)
    rays = np.ascontiguousarray(_pixel_rays(config)@basis)
    ray_started = time.perf_counter()
    ranges, geom_ids = _cast(model, query_data, origin, rays, config.far_meters, backend)
    ray_seconds = time.perf_counter()-ray_started
    valid = (geom_ids >= 0) & (ranges >= 0) & (ranges <= config.far_meters)
    normalized = np.ones(len(rays), dtype=np.float32)
    normalized[valid] = ((np.clip(ranges[valid], config.near_meters, config.far_meters)-config.near_meters)
                         /(config.far_meters-config.near_meters)).astype(np.float32)
    return normalized.reshape(1, config.height, config.width), {
        "version": 1, "backend": backend, "pixelRayCount": len(rays),
        "width": config.width, "height": config.height,
        "cameraOriginWorldMeters": origin.tolist(), "cameraForwardWorld": basis[1].tolist(),
        "validPixelCount": int(valid.sum()), "noHitPixelCount": int((~valid).sum()),
        "nearClippedPixelCount": int((valid & (ranges < config.near_meters)).sum()),
        "validPixelMaskBase64": base64.b64encode(np.packbits(valid, bitorder="little").tobytes()).decode(),
        "eligibleCollisionGeomCount": eligible_count,
        "raycastSeconds": ray_seconds,
        "sensorSeconds": time.perf_counter()-started,
        "physicalTimeSeconds": float(body.data.time), "worldStateWritten": False,
        "selfGeometryFiltered": True, "queryModelAndDataIsolated": True,
        "imageResizedOrUpsampled": False}
