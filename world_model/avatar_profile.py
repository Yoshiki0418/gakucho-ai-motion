"""Measured avatar geometry used to build the physical body.

Appearance assets do not provide body mass or joint strength. The physical
model keeps the existing 64.7 kg mass distribution and torque assumptions;
this file validates geometric measurements, never claims to measure dynamics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _finite_vector(value, name, *, positive=False):
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain three finite metre values") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite metre values")
    if positive and np.any(vector <= 0):
        raise ValueError(f"{name} must be positive")
    return vector


def load_avatar_profile(value=None):
    """Return an owned, validated profile; None retains the original body."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if not isinstance(value, Mapping):
        raise ValueError("avatar_profile must be a profile object or JSON path")
    result = copy.deepcopy(dict(value))
    physical = result.get("physical")
    if not isinstance(physical, Mapping):
        raise ValueError("avatar_profile.physical is required")
    coordinates = physical.get("coordinateSystem", "")
    if not ("Z-up" in coordinates and "-Y-forward" in coordinates):
        raise ValueError("avatar physical coordinates must be Z-up, -Y-forward")
    root_height = float(physical.get("rootHeight", float("nan")))
    if not np.isfinite(root_height) or not .3 < root_height < 2.0:
        raise ValueError("avatar physical.rootHeight must be a plausible metre height")
    for section in ("hipOffsets", "legs", "arms", "foot"):
        if not isinstance(physical.get(section), Mapping):
            raise ValueError(f"physical.{section} must be an object")
    for side in ("left", "right"):
        _finite_vector(physical.get("hipOffsets", {}).get(side), f"hipOffsets.{side}")
        for section, names in (("legs", ("thighLength", "shankLength")),
                               ("arms", ("upperArmLength", "forearmLength"))):
            values = physical.get(section, {}).get(side, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"{section}.{side} must be an object")
            for name in names:
                number = float(values.get(name, float("nan")))
                if not np.isfinite(number) or not .08 < number < .9:
                    raise ValueError(f"{section}.{side}.{name} must be a plausible metre length")
        foot = physical.get("foot", {}).get(side, {})
        if not isinstance(foot, Mapping):
            raise ValueError(f"foot.{side} must be an object")
        half = _finite_vector(foot.get("halfSize"), f"foot.{side}.halfSize", positive=True)
        _finite_vector(foot.get("centerOffset"), f"foot.{side}.centerOffset")
        if half.max() > .3 or half.min() < .005:
            raise ValueError(f"foot.{side}.halfSize is outside physical foot bounds")
    joints = physical.get("joints", {})
    if not isinstance(joints, Mapping):
        raise ValueError("physical.joints must be an object")
    for name, joint in joints.items():
        _finite_vector(joint.get("position") if isinstance(joint, Mapping) else joint,
                       f"physical.joints.{name}")
    head = physical.get("head", {})
    if not isinstance(head, Mapping):
        raise ValueError("physical.head must be an object")
    if "radius" in head:
        radius = float(head["radius"])
        if not np.isfinite(radius) or not .02 < radius < .3:
            raise ValueError("physical.head.radius must be a plausible metre radius")
    if "centerOffset" in head:
        _finite_vector(head["centerOffset"], "physical.head.centerOffset")
    pelvis = physical.get("pelvis", {})
    if not isinstance(pelvis, Mapping):
        raise ValueError("physical.pelvis must be an object")
    if pelvis:
        radius = float(pelvis.get("radius", float("nan")))
        axis_half = float(pelvis.get("axisHalfLength", float("nan")))
        if not np.isfinite(radius) or not .03 < radius < .3:
            raise ValueError("physical.pelvis.radius must be a plausible metre radius")
        if not np.isfinite(axis_half) or not 0 <= axis_half < .3:
            raise ValueError("physical.pelvis.axisHalfLength must be a plausible metre extent")
        centre = _finite_vector(pelvis.get("centerOffset"), "physical.pelvis.centerOffset")
        offset = float(pelvis.get("seatOffset", float("nan")))
        if not np.isfinite(offset) or not .03 < offset < .3:
            raise ValueError("physical.pelvis.seatOffset must be a plausible metre offset")
        if abs(offset-(radius-centre[2])) > 1e-5:
            raise ValueError("physical.pelvis.seatOffset must match its actual capsule bottom")
    for field in ("height", "referenceHeight"):
        if field in physical:
            height = float(physical[field])
            if not np.isfinite(height) or not .6 < height < 2.8:
                raise ValueError(f"physical.{field} must be a plausible metre height")
    return result


def avatar_profile_sha256(profile) -> str | None:
    if profile is None:
        return None
    canonical = {key: value for key, value in profile.items() if key != "profileSha256"}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def physical_joint(profile, name, fallback):
    """Look up a canonical measured joint, with explicit simplified fallback."""
    if profile is None:
        return np.asarray(fallback, dtype=np.float64)
    value = profile["physical"].get("joints", {}).get(name)
    if value is None:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value.get("position") if isinstance(value, Mapping) else value,
                      dtype=np.float64)


def leg_dimensions(profile, side):
    if profile is None:
        return .415, .415
    values = profile["physical"]["legs"][side]
    return float(values["thighLength"]), float(values["shankLength"])


def physical_assumptions(profile):
    return {"avatarGeometryMeasured": profile is not None,
            "totalMassKilograms": 64.7, "massWasMeasured": False,
            "massDistribution": "existing physical model segment masses",
            "jointStrengthWasMeasured": False,
            "jointLimits": "existing simplified 20-hinge human limits"}
