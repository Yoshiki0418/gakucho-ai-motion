"""A compact dynamic humanoid for the first physical-control stage.

Coordinates are metres, Z-up, with the body facing -Y.  The floating pelvis
has no actuator: only joint torques can influence its position and balance.
This is a physical control rig, not a conversion of every BVH finger bone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from xml.etree import ElementTree as ET

import numpy as np

from .avatar_profile import load_avatar_profile, physical_joint

STANDING_ROOT_HEIGHT = 0.93
PHYSICS_TIMESTEP = 0.002
CONTROL_SUBSTEPS = 10

# Axis signs use a right-handed Z-up coordinate system.  Positive knee pitch
# bends the shank toward +Y, behind a body whose forward direction is -Y.
JOINT_SPECS = (
    ("torso_pitch", (-0.55, 0.55), 260.0, 26.0, 120.0, 0.20),
    ("torso_roll", (-0.40, 0.40), 260.0, 26.0, 100.0, 0.18),
    ("left_hip_pitch", (-1.65, 0.75), 220.0, 22.0, 150.0, 0.55),
    ("left_hip_roll", (-0.55, 0.55), 200.0, 20.0, 120.0, 0.35),
    ("left_hip_yaw", (-0.60, 0.60), 140.0, 14.0, 80.0, 0.35),
    ("left_knee", (0.0, 2.35), 240.0, 24.0, 150.0, 0.80),
    ("left_ankle_pitch", (-0.85, 0.65), 160.0, 16.0, 90.0, 0.35),
    ("left_ankle_roll", (-0.45, 0.45), 160.0, 16.0, 70.0, 0.30),
    ("right_hip_pitch", (-1.65, 0.75), 220.0, 22.0, 150.0, 0.55),
    ("right_hip_roll", (-0.55, 0.55), 200.0, 20.0, 120.0, 0.35),
    ("right_hip_yaw", (-0.60, 0.60), 140.0, 14.0, 80.0, 0.35),
    ("right_knee", (0.0, 2.35), 240.0, 24.0, 150.0, 0.80),
    ("right_ankle_pitch", (-0.85, 0.65), 160.0, 16.0, 90.0, 0.35),
    ("right_ankle_roll", (-0.45, 0.45), 160.0, 16.0, 70.0, 0.30),
    ("left_shoulder_pitch", (-1.80, 1.80), 65.0, 8.0, 45.0, 0.60),
    ("left_shoulder_roll", (-1.65, 1.65), 65.0, 8.0, 45.0, 0.60),
    ("left_elbow", (0.0, 2.40), 45.0, 6.0, 30.0, 0.65),
    ("right_shoulder_pitch", (-1.80, 1.80), 65.0, 8.0, 45.0, 0.60),
    ("right_shoulder_roll", (-1.65, 1.65), 65.0, 8.0, 45.0, 0.60),
    ("right_elbow", (0.0, 2.40), 45.0, 6.0, 30.0, 0.65),
)
JOINT_NAMES = tuple(spec[0] for spec in JOINT_SPECS)
JOINT_LIMITS = np.asarray([spec[1] for spec in JOINT_SPECS], dtype=np.float64)
PD_KP = np.asarray([spec[2] for spec in JOINT_SPECS], dtype=np.float64)
PD_KD = np.asarray([spec[3] for spec in JOINT_SPECS], dtype=np.float64)
TORQUE_LIMITS = np.asarray([spec[4] for spec in JOINT_SPECS], dtype=np.float64)
RESIDUAL_LIMITS = np.asarray([spec[5] for spec in JOINT_SPECS], dtype=np.float64)


def standing_joint_angles() -> np.ndarray:
    """A slightly unlocked upright stance; no floating-root constraint."""
    pose = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for side in ("left", "right"):
        pose[JOINT_NAMES.index(f"{side}_hip_pitch")] = -0.03
        pose[JOINT_NAMES.index(f"{side}_knee")] = 0.06
        pose[JOINT_NAMES.index(f"{side}_ankle_pitch")] = -0.03
        pose[JOINT_NAMES.index(f"{side}_elbow")] = 0.12
    return pose


def _values(value) -> str:
    return " ".join(f"{float(item):.10g}" for item in value)


def _vector(value, count: int, field: str, *, positive=False):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{field} must contain finite numbers") from error
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain {count} finite numbers")
    if positive and np.any(array <= 0):
        raise ValueError(f"{field} must be positive")
    return array


def build_humanoid_xml(scene: Mapping | None = None, avatar_profile=None) -> str:
    """Build MJCF; scene box sizes are FULL extents, all poses are Z-up.

    A supplied scene must provide its own floor pieces.  In particular, an
    infinite plane is never inserted underneath its explicitly modelled holes.
    Scene mesh assets must already be convex pieces, e.g. triangular prisms.
    """
    avatar_profile = load_avatar_profile(avatar_profile)
    root = ET.Element("mujoco", model="gakucho_physical_humanoid_stage1")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(PHYSICS_TIMESTEP), gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="60", tolerance="1e-9")
    ET.SubElement(root, "size", nconmax="512", njmax="2048")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.5", armature="0.015", limited="true")
    ET.SubElement(default, "geom", friction="0.9 0.005 0.0001", condim="3",
                  solref="0.01 1", solimp="0.9 0.95 0.001", margin="0.001",
                  rgba="0.68 0.73 0.82 1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="2 -3 5", dir="-0.3 0.4 -1", directional="true")
    ET.SubElement(world, "camera", name="overview", pos="3 -4 2.6",
                  xyaxes="0.8 0.6 0 -0.22 0.3 0.93")
    if scene is None:
        ET.SubElement(world, "geom", name="floor", type="box", size="5 5 0.1", pos="0 0 -0.1",
                      rgba="0.30 0.33 0.36 1", friction="0.9 0.005 0.0001")
    else:
        if not isinstance(scene, Mapping) or not isinstance(scene.get("geoms"), list):
            raise ValueError("scene must contain a geoms list")
        seen = set()
        for index, entry in enumerate(scene["geoms"]):
            if not isinstance(entry, Mapping):
                raise ValueError("scene geom must be an object")
            name = str(entry.get("id", f"scene_{index}"))
            if not name or name in seen or name.startswith("humanoid_"):
                raise ValueError("scene geom ids must be unique and non-empty")
            seen.add(name)
            position = _vector(entry.get("position"), 3, f"{name}.position")
            attrs = {"name": name, "pos": _values(position),
                     "rgba": _values(entry.get("rgba", (0.54, 0.47, 0.36, 1)))}
            if "quaternion" in entry:
                quaternion = _vector(entry["quaternion"], 4, f"{name}.quaternion")
                length = np.linalg.norm(quaternion)
                if length < 1e-8:
                    raise ValueError(f"{name}.quaternion is zero")
                attrs["quat"] = _values(quaternion / length)
            else:
                yaw = float(entry.get("yaw", 0))
                if not math.isfinite(yaw):
                    raise ValueError(f"{name}.yaw must be finite")
                attrs["euler"] = f"0 0 {yaw:.10g}"
            if "friction" in entry:
                friction = entry["friction"]
                if isinstance(friction, (float, int)):
                    friction = (friction, 0.005, 0.0001)
                friction = _vector(friction, 3, f"{name}.friction")
                if np.any(friction < 0):
                    raise ValueError(f"{name}.friction must be nonnegative")
                attrs["friction"] = _values(friction)
            geom_type = entry.get("type", "box")
            if geom_type == "box":
                attrs.update(type="box", size=_values(_vector(entry.get("size"), 3,
                                      f"{name}.size", positive=True) / 2))
            elif geom_type == "mesh":
                vertices = np.asarray(entry.get("vertices"), dtype=np.float64)
                faces = np.asarray(entry.get("faces"), dtype=np.int64)
                if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                        or not np.isfinite(vertices).all() or faces.ndim != 2
                        or faces.shape[1] != 3 or len(faces) < 4 or np.any(faces < 0)
                        or np.any(faces >= len(vertices))):
                    raise ValueError(f"{name} must be a finite closed convex mesh")
                mesh_name = f"scene_mesh_{index}"
                ET.SubElement(asset, "mesh", name=mesh_name,
                              vertex=_values(vertices.ravel()),
                              face=" ".join(str(int(value)) for value in faces.ravel()))
                attrs.update(type="mesh", mesh=mesh_name)
            else:
                raise ValueError(f"Unsupported scene geom type {geom_type!r}")
            ET.SubElement(world, "geom", **attrs)

    specs = {spec[0]: spec for spec in JOINT_SPECS}

    def joint(body, name, axis):
        ET.SubElement(body, "joint", name=name, type="hinge", axis=_values(axis),
                      range=_values(specs[name][1]))

    def capsule(body, name, end, radius, mass, color=None):
        attrs = {"name": name, "type": "capsule", "fromto": "0 0 0 " + _values(end),
                 "size": str(radius), "mass": str(mass)}
        if color:
            attrs["rgba"] = _values(color)
        ET.SubElement(body, "geom", **attrs)

    pelvis = ET.SubElement(world, "body", name="pelvis", pos=f"0 0 {STANDING_ROOT_HEIGHT}")
    ET.SubElement(pelvis, "freejoint", name="floating_root")
    ET.SubElement(pelvis, "geom", name="humanoid_pelvis", type="capsule",
                  fromto="-0.085 0 0 0.085 0 0", size="0.09", mass="8")
    torso = ET.SubElement(pelvis, "body", name="torso", pos="0 0 0.10")
    joint(torso, "torso_pitch", (1, 0, 0))
    joint(torso, "torso_roll", (0, 1, 0))
    capsule(torso, "humanoid_torso", (0, 0, 0.37), 0.12, 20)
    head = ET.SubElement(torso, "body", name="head", pos="0 0 0.56")
    ET.SubElement(head, "geom", name="humanoid_head", type="sphere", size="0.115", mass="4.5")
    for side, sign in (("left", 1), ("right", -1)):
        thigh = ET.SubElement(pelvis, "body", name=f"{side}_thigh", pos=f"{sign * .10} 0 -0.02")
        joint(thigh, f"{side}_hip_pitch", (1, 0, 0))
        joint(thigh, f"{side}_hip_roll", (0, 1, 0))
        joint(thigh, f"{side}_hip_yaw", (0, 0, 1))
        capsule(thigh, f"humanoid_{side}_thigh", (0, 0, -0.415), 0.055, 7)
        shank = ET.SubElement(thigh, "body", name=f"{side}_shank", pos="0 0 -0.415")
        joint(shank, f"{side}_knee", (1, 0, 0))
        capsule(shank, f"humanoid_{side}_shank", (0, 0, -0.415), 0.043, 4)
        foot = ET.SubElement(shank, "body", name=f"{side}_foot", pos="0 0 -0.415")
        joint(foot, f"{side}_ankle_pitch", (1, 0, 0))
        joint(foot, f"{side}_ankle_roll", (0, 1, 0))
        ET.SubElement(foot, "geom", name=f"humanoid_{side}_foot", type="box",
                      pos="0 -0.065 -0.045", size="0.055 0.135 0.035", mass="1.4",
                      rgba="0.25 0.30 0.38 1")
        upper = ET.SubElement(torso, "body", name=f"{side}_upper_arm", pos=f"{sign * .20} 0 0.34")
        joint(upper, f"{side}_shoulder_pitch", (1, 0, 0))
        joint(upper, f"{side}_shoulder_roll", (0, 1, 0))
        capsule(upper, f"humanoid_{side}_upper_arm", (0, 0, -0.285), 0.035, 2.0)
        lower = ET.SubElement(upper, "body", name=f"{side}_forearm", pos="0 0 -0.285")
        joint(lower, f"{side}_elbow", (-1, 0, 0))
        capsule(lower, f"humanoid_{side}_forearm", (0, 0, -0.255), 0.029, 1.3)
        ET.SubElement(lower, "geom", name=f"humanoid_{side}_hand", type="sphere",
                      pos="0 0 -0.28", size="0.043", mass="0.4")

    contact = ET.SubElement(root, "contact")
    # Adjacent links are excluded automatically.  These additional pairs are
    # close at rest by anatomy; excluding them avoids artificial internal kicks.
    for side in ("left", "right"):
        ET.SubElement(contact, "exclude", body1="torso", body2=f"{side}_upper_arm")
        ET.SubElement(contact, "exclude", body1="pelvis", body2=f"{side}_forearm")
        ET.SubElement(contact, "exclude", body1="torso", body2=f"{side}_forearm")
    actuator = ET.SubElement(root, "actuator")
    for name, _, _, _, limit, _ in JOINT_SPECS:
        ET.SubElement(actuator, "motor", name=f"motor_{name}", joint=name,
                      gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}",
                      forcelimited="true", forcerange=f"{-limit} {limit}")
    if avatar_profile is not None:
        _apply_avatar_geometry(root, avatar_profile)
    return ET.tostring(root, encoding="unicode")


def _apply_avatar_geometry(root, profile):
    """Replace geometric dimensions only, preserving masses and actuators.

    The control skeleton keeps its compact articulation and downward arm bind.
    Render-bone bind transforms are calibrated separately from these bodies.
    """
    physical = profile["physical"]
    root_height = float(physical["rootHeight"])
    scale = root_height / STANDING_ROOT_HEIGHT
    bodies = {body.get("name"): body for body in root.iter("body")}
    geoms = {geom.get("name"): geom for geom in root.iter("geom")}
    hips = physical_joint(profile, "hips", [0, 0, root_height])
    spine = physical_joint(profile, "spine", hips + [0, 0, .10 * scale])
    chest = physical_joint(profile, "spine4", hips + [0, 0, .47 * scale])
    head = physical_joint(profile, "head", hips + [0, 0, .66 * scale])
    bodies["pelvis"].set("pos", _values([0, 0, root_height]))
    bodies["torso"].set("pos", _values(spine - hips))
    bodies["head"].set("pos", _values(head - spine))
    hip_width = abs(float(physical["hipOffsets"]["left"][0])) + abs(float(physical["hipOffsets"]["right"][0]))
    geoms["humanoid_pelvis"].set("fromto", _values([-hip_width*.425, 0, 0, hip_width*.425, 0, 0]))
    geoms["humanoid_pelvis"].set("size", str(.09 * scale))
    measured_pelvis = physical.get("pelvis")
    if measured_pelvis:
        centre = np.asarray(measured_pelvis["centerOffset"], dtype=np.float64)
        axis = np.asarray([float(measured_pelvis["axisHalfLength"]), 0., 0.])
        geoms["humanoid_pelvis"].set("fromto", _values(np.concatenate((centre-axis, centre+axis))))
        geoms["humanoid_pelvis"].set("size", str(float(measured_pelvis["radius"])))
    geoms["humanoid_torso"].set("fromto", "0 0 0 " + _values(chest - spine))
    geoms["humanoid_torso"].set("size", str(.12 * scale))
    head_geometry = physical.get("head", {})
    geoms["humanoid_head"].set("size", str(float(head_geometry.get("radius", .115 * scale))))
    geoms["humanoid_head"].set("pos", _values(head_geometry.get("centerOffset", [0, 0, 0])))
    for side, sign in (("left", 1), ("right", -1)):
        leg, arm, foot = physical["legs"][side], physical["arms"][side], physical["foot"][side]
        thigh, shank = float(leg["thighLength"]), float(leg["shankLength"])
        upper, lower = float(arm["upperArmLength"]), float(arm["forearmLength"])
        bodies[f"{side}_thigh"].set("pos", _values(physical["hipOffsets"][side]))
        bodies[f"{side}_shank"].set("pos", _values([0, 0, -thigh]))
        bodies[f"{side}_foot"].set("pos", _values([0, 0, -shank]))
        for segment, length, radius in (("thigh", thigh, .055), ("shank", shank, .043),
                                        ("upper_arm", upper, .035), ("forearm", lower, .029)):
            geoms[f"humanoid_{side}_{segment}"].set("fromto", "0 0 0 " + _values([0, 0, -length]))
            geoms[f"humanoid_{side}_{segment}"].set("size", str(radius * scale))
        geoms[f"humanoid_{side}_foot"].set("pos", _values(foot["centerOffset"]))
        geoms[f"humanoid_{side}_foot"].set("size", _values(foot["halfSize"]))
        shoulder = physical_joint(profile, f"{side}UpperArm", hips + [sign*.20*scale, 0, .44*scale])
        bodies[f"{side}_upper_arm"].set("pos", _values(shoulder - spine))
        bodies[f"{side}_forearm"].set("pos", _values([0, 0, -upper]))
        geoms[f"humanoid_{side}_hand"].set("pos", _values([0, 0, -lower-.025*scale]))
        geoms[f"humanoid_{side}_hand"].set("size", str(.043*scale))
