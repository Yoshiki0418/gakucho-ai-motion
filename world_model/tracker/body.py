"""Full-body tracking humanoid in the gakucho proportions and BVH skeleton.

This is a separate body contract from ``world_model.humanoid`` (20 joints).
Joint origins are the gakucho T-pose joints in the avatar profile, and the
segment list follows the T2M/mocap BVH skeleton (fingers omitted, five spine
bones merged into three, two neck bones into one). Every body frame is
world-aligned at rest, so a BVH local rotation becomes a joint rotation by a
fixed basis change only (see ``retarget.py``).

Coordinates: Z-up, forward -Y, metres, like the rest of ``world_model``.
Each segment has three hinge joints whose intrinsic order puts the
smallest-range axis in the middle, away from Euler singularities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

PROFILE_PATH = (Path(__file__).resolve().parents[2]
                / "frontend/public/world-model/gakucho-avatar-profile.json")
TIMESTEP = 0.005
CONTROL_SUBSTEPS = 4          # 50 Hz control, matching the existing 20 ms control
TOTAL_MASS = 64.7             # existing physical assumption, not a measurement
# display (Y-up, +Z forward) -> physics (Z-up, -Y forward)
DISPLAY_TO_PHYSICS = np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


@dataclass(frozen=True)
class Segment:
    name: str
    parent: str | None
    joint: str                # avatar-profile canonical joint at the frame origin
    bvh: tuple[str, ...]      # BVH joints whose local rotations compose this segment
    order: str                # intrinsic hinge order, e.g. "XYZ" -> Rx @ Ry @ Rz
    kp: float
    kd: float
    torque: float


SEGMENTS = (
    Segment("pelvis", None, "hips", ("Hips",), "", 0, 0, 0),
    Segment("spine", "pelvis", "spine", ("Spine", "Spine1"), "XYZ", 600, 60, 300),
    Segment("spine2", "spine", "spine2", ("Spine2", "Spine3"), "XYZ", 600, 60, 300),
    Segment("chest", "spine2", "spine4", ("Spine4",), "XYZ", 500, 50, 300),
    Segment("neck", "chest", "neck", ("Neck", "Neck1"), "XYZ", 150, 15, 60),
    Segment("head", "neck", "head", ("Head",), "XYZ", 100, 10, 40),
    Segment("left_clavicle", "chest", "leftShoulder", ("LeftShoulder",), "ZXY", 300, 30, 120),
    Segment("left_upper_arm", "left_clavicle", "leftUpperArm", ("LeftArm",), "ZXY", 250, 25, 150),
    Segment("left_forearm", "left_upper_arm", "leftLowerArm", ("LeftForeArm",), "ZYX", 150, 15, 100),
    Segment("left_hand", "left_forearm", "leftHand", ("LeftHand",), "XZY", 50, 5, 30),
    Segment("right_clavicle", "chest", "rightShoulder", ("RightShoulder",), "ZXY", 300, 30, 120),
    Segment("right_upper_arm", "right_clavicle", "rightUpperArm", ("RightArm",), "ZXY", 250, 25, 150),
    Segment("right_forearm", "right_upper_arm", "rightLowerArm", ("RightForeArm",), "ZYX", 150, 15, 100),
    Segment("right_hand", "right_forearm", "rightHand", ("RightHand",), "XZY", 50, 5, 30),
    Segment("left_thigh", "pelvis", "leftUpLeg", ("LeftUpLeg",), "XYZ", 600, 60, 300),
    Segment("left_shin", "left_thigh", "leftLowerLeg", ("LeftLeg",), "XYZ", 600, 60, 300),
    Segment("left_foot", "left_shin", "leftFoot", ("LeftFoot",), "XYZ", 400, 40, 200),
    Segment("left_toe", "left_foot", "leftToe", ("LeftToeBase",), "XYZ", 100, 10, 50),
    Segment("right_thigh", "pelvis", "rightUpLeg", ("RightUpLeg",), "XYZ", 600, 60, 300),
    Segment("right_shin", "right_thigh", "rightLowerLeg", ("RightLeg",), "XYZ", 600, 60, 300),
    Segment("right_foot", "right_shin", "rightFoot", ("RightFoot",), "XYZ", 400, 40, 200),
    Segment("right_toe", "right_foot", "rightToe", ("RightToeBase",), "XYZ", 100, 10, 50),
)
SEGMENT_NAMES = tuple(s.name for s in SEGMENTS)
ACTUATED = SEGMENTS[1:]
JOINT_NAMES = tuple(f"{s.name}_{axis.lower()}" for s in ACTUATED for axis in s.order)
KP = np.repeat([s.kp for s in ACTUATED], 3).astype(np.float64)
KD = np.repeat([s.kd for s in ACTUATED], 3).astype(np.float64)
TORQUE = np.repeat([s.torque for s in ACTUATED], 3).astype(np.float64)
# Geoms touching the ground in normal standing/walking; used for contact features.
FOOT_BODIES = ("left_foot", "left_toe", "right_foot", "right_toe")
# Collision bits: the body never collides with itself, only with the world.
BODY_CONTYPE, BODY_CONAFFINITY = 2, 1
WORLD_CONTYPE, WORLD_CONAFFINITY = 1, 2


def load_profile(path=PROFILE_PATH):
    return json.loads(Path(path).read_text())


def profile_sha256(path=PROFILE_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rest_joint_positions(profile):
    """World physics positions of every segment origin at the T-pose."""
    joints = profile["display"]["joints"]
    return {s.name: DISPLAY_TO_PHYSICS @ np.asarray(joints[s.joint]["position"], dtype=np.float64)
            for s in SEGMENTS}


def _v(values):
    return " ".join(f"{float(x):.6g}" for x in values)


def _capsule(body, name, start, end, radius):
    ET.SubElement(body, "geom", name=name, type="capsule", fromto=_v(np.concatenate([start, end])),
                  size=f"{radius:.4g}")


def build_xml(profile=None, *, floor="plane", extra_world=None):
    """Return MJCF text. ``floor="plane"`` adds an infinite floor; ``None`` omits it."""
    profile = load_profile() if profile is None else profile
    rest = rest_joint_positions(profile)
    physical = profile["physical"]
    root = ET.Element("mujoco", model="gakucho_tracking_humanoid")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(TIMESTEP), gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="10", ls_iterations="20")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", type="hinge", damping="0.5", armature="0.02", limited="false")
    ET.SubElement(default, "geom", contype=str(BODY_CONTYPE), conaffinity=str(BODY_CONAFFINITY),
                  condim="3", friction="1.0 0.005 0.0001", density="1000",
                  rgba="0.70 0.72 0.80 1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    if floor == "plane":
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                      contype=str(WORLD_CONTYPE), conaffinity=str(WORLD_CONAFFINITY),
                      rgba="0.35 0.38 0.40 1")
    bodies = {}
    for seg in SEGMENTS:
        parent_xml = world if seg.parent is None else bodies[seg.parent]
        parent_pos = np.zeros(3) if seg.parent is None else rest[seg.parent]
        body = ET.SubElement(parent_xml, "body", name=seg.name, pos=_v(rest[seg.name] - parent_pos))
        bodies[seg.name] = body
        if seg.parent is None:
            ET.SubElement(body, "freejoint", name="root")
        else:
            for axis in seg.order:
                vector = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}[axis]
                ET.SubElement(body, "joint", name=f"{seg.name}_{axis.lower()}", axis=vector)

    def rel(name, target):
        return rest[target] - rest[name]

    pelvis = physical["pelvis"]
    centre = np.asarray(pelvis["centerOffset"], dtype=np.float64)
    half = np.asarray([pelvis["axisHalfLength"], 0., 0.])
    _capsule(bodies["pelvis"], "pelvis", centre - half, centre + half, pelvis["radius"])
    _capsule(bodies["spine"], "spine", np.zeros(3), rel("spine", "spine2"), .105)
    _capsule(bodies["spine2"], "spine2", np.zeros(3), rel("spine2", "chest"), .115)
    _capsule(bodies["chest"], "chest", np.zeros(3), rel("chest", "neck") * .75, .12)
    _capsule(bodies["chest"], "shoulders", np.asarray([-.11, .01, .09]), np.asarray([.11, .01, .09]), .07)
    _capsule(bodies["neck"], "neck", np.zeros(3), rel("neck", "head"), .05)
    head = physical["head"]
    ET.SubElement(bodies["head"], "geom", name="head", type="sphere", size=f"{head['radius']:.4g}",
                  pos=_v(head["centerOffset"]))
    for side in ("left", "right"):
        sign = 1. if side == "left" else -1.
        _capsule(bodies[f"{side}_clavicle"], f"{side}_clavicle", np.zeros(3),
                 rel(f"{side}_clavicle", f"{side}_upper_arm") * .7, .04)
        _capsule(bodies[f"{side}_upper_arm"], f"{side}_upper_arm", np.zeros(3),
                 rel(f"{side}_upper_arm", f"{side}_forearm"), .05)
        _capsule(bodies[f"{side}_forearm"], f"{side}_forearm", np.zeros(3),
                 rel(f"{side}_forearm", f"{side}_hand"), .042)
        _capsule(bodies[f"{side}_hand"], f"{side}_hand", np.asarray([sign * .02, 0., 0.]),
                 np.asarray([sign * .13, 0., 0.]), .035)
        _capsule(bodies[f"{side}_thigh"], f"{side}_thigh", np.zeros(3),
                 rel(f"{side}_thigh", f"{side}_shin"), .08)
        _capsule(bodies[f"{side}_shin"], f"{side}_shin", np.zeros(3),
                 rel(f"{side}_shin", f"{side}_foot"), .05)
        # Shoe box measured from the GLB, split at the ball joint into heel and toe.
        foot = physical["foot"][side]
        size, centre = np.asarray(foot["halfSize"]), np.asarray(foot["centerOffset"])
        toe = rel(f"{side}_foot", f"{side}_toe")
        heel_y, tip_y, bottom = centre[1] + size[1], centre[1] - size[1], centre[2] - size[2]
        # The measured shoe volume is mostly not foot; it gets 40% density.
        ET.SubElement(bodies[f"{side}_foot"], "geom", name=f"{side}_foot", type="box", density="400",
                      pos=_v([centre[0], (heel_y + toe[1]) / 2, centre[2]]),
                      size=_v([size[0], (heel_y - toe[1]) / 2, size[2]]))
        toe_half_height = .025
        ET.SubElement(bodies[f"{side}_toe"], "geom", name=f"{side}_toe", type="box", density="400",
                      pos=_v([centre[0] - toe[0], (tip_y - toe[1]) / 2, bottom - toe[2] + toe_half_height]),
                      size=_v([size[0], (toe[1] - tip_y) / 2, toe_half_height]))

    # Scene bodies come after the humanoid so its bodies keep ids 1..len(SEGMENTS).
    for element in extra_world or ():
        world.append(element)

    actuator = ET.SubElement(root, "actuator")
    for name, kp, kd, limit in zip(JOINT_NAMES, KP, KD, TORQUE):
        # PD on the joint angle: force = kp*ctrl - kp*q - kd*qdot, clipped to +-limit.
        ET.SubElement(actuator, "general", name=name, joint=name, gaintype="fixed",
                      biastype="affine", gainprm=f"{kp:g}", biasprm=f"0 {-kp:g} {-kd:g}",
                      ctrllimited="false", forcelimited="true", forcerange=f"{-limit:g} {limit:g}")
    xml = ET.tostring(root, encoding="unicode")
    return _rescale_mass(xml)


def _rescale_mass(xml):
    """Scale every density so the humanoid weighs ``TOTAL_MASS``.

    Only the humanoid (the pelvis subtree) counts. Until 2026-10-16 the sum ran
    over every body, so the mocap seat box (``extra_world``) took 41 kg and left
    the humanoid at 23.5 kg in all seated runs (v7-seated*, the first v8 start).
    Without scene bodies the result is unchanged.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    factor = TOTAL_MASS / float(model.body_subtreemass[model.body("pelvis").id])
    root = ET.fromstring(xml)
    root.find("default").find("geom").set("density", f"{1000. * factor:.6g}")
    for geom in root.find("worldbody").iter("geom"):
        if "density" in geom.attrib:
            geom.set("density", f"{float(geom.get('density')) * factor:.6g}")
    return ET.tostring(root, encoding="unicode")


def build_model(profile=None, **kwargs):
    return mujoco.MjModel.from_xml_string(build_xml(profile, **kwargs))


def qpos_layout(model):
    """Indices of the hinge qpos/qvel in ``JOINT_NAMES`` order."""
    qpos = np.asarray([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    qvel = np.asarray([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    return qpos, qvel
