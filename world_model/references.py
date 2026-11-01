"""Project cached T2M BVH motions onto the smaller physical control rig.

These are reference targets, not physical rollouts. Sampling them never moves
an environment's floating root. The original trajectory, scale and provenance
are retained; fit/limit diagnostics make the 55-to-20-joint loss explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .avatar_profile import avatar_profile_sha256, load_avatar_profile
from .humanoid import JOINT_LIMITS, JOINT_NAMES, build_humanoid_xml

_ASSETS = Path(__file__).resolve().parents[1] / "frontend/public/motions/room-prototype"
_BASIS = Rotation.from_euler("x", np.pi / 2).as_matrix()
_WALK_LOOP = (260, 372)
_CLEARANCE = .003


@dataclass
class _Bone:
    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int


@dataclass
class _Bvh:
    bones: list[_Bone]
    frames: np.ndarray
    frame_time: float
    local_rotations: np.ndarray
    world_rotations: np.ndarray
    world_positions: np.ndarray
    rest_positions: np.ndarray
    scale: float


def _parse_bvh(text: str, target_height=1.8) -> _Bvh:
    try:
        hierarchy, motion = text.split("MOTION", 1)
    except ValueError as error:
        raise ValueError("BVH has no MOTION section") from error
    tokens = re.findall(r"[{}]|[^\s{}]+", hierarchy)
    bones: list[_Bone] = []
    cursor = 0
    channels = 0

    def expect(value):
        nonlocal cursor
        if cursor >= len(tokens) or tokens[cursor] != value:
            raise ValueError(f"BVH hierarchy expected {value}")
        cursor += 1

    def bone(parent):
        nonlocal cursor, channels
        kind = tokens[cursor]
        cursor += 1
        if kind == "End":
            expect("Site")
            name = f"{bones[parent].name}__end"
        elif kind in ("ROOT", "JOINT"):
            name = tokens[cursor]
            cursor += 1
        else:
            raise ValueError("Unsupported BVH hierarchy token")
        expect("{")
        expect("OFFSET")
        offset = np.asarray([float(value) for value in tokens[cursor:cursor + 3]])
        cursor += 3
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Invalid BVH bone offset")
        names = ()
        start = channels
        if kind != "End":
            expect("CHANNELS")
            count = int(tokens[cursor])
            cursor += 1
            names = tuple(tokens[cursor:cursor + count])
            cursor += count
            allowed = {axis + suffix for axis in "XYZ" for suffix in ("position", "rotation")}
            if any(name not in allowed for name in names):
                raise ValueError("Unsupported BVH channel")
            channels += count
        index = len(bones)
        bones.append(_Bone(name, parent, offset, names, start))
        while tokens[cursor] != "}":
            bone(index)
        expect("}")

    expect("HIERARCHY")
    bone(-1)
    if cursor != len(tokens):
        raise ValueError("Trailing BVH hierarchy data")
    frame_match = re.search(r"Frames:\s*(\d+)", motion)
    time_match = re.search(r"Frame Time:\s*([^\s]+)", motion)
    if not frame_match or not time_match:
        raise ValueError("Missing BVH frame metadata")
    count = int(frame_match[1])
    frame_time = float(time_match[1])
    if count < 2 or not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("Invalid BVH timing")
    numbers = np.fromstring(motion[time_match.end():], sep=" ")
    if numbers.size != count * channels or not np.isfinite(numbers).all():
        raise ValueError("BVH frame channels do not match its declared count")
    frames = numbers.reshape(count, channels)
    positions = np.zeros((count, len(bones), 3))
    world_rotation = np.zeros((count, len(bones), 3, 3))
    local_rotation = np.zeros_like(world_rotation)
    rest = np.zeros((len(bones), 3))
    for index, node in enumerate(bones):
        local_position = np.repeat(node.offset[None], count, axis=0)
        rotation_names = [name for name in node.channels if name.endswith("rotation")]
        if rotation_names:
            rotation_values = np.column_stack([frames[:, node.channel_start + node.channels.index(name)]
                                                for name in rotation_names])
            local_rotation[:, index] = Rotation.from_euler("".join(name[0] for name in rotation_names),
                                                           rotation_values, degrees=True).as_matrix()
        else:
            local_rotation[:, index] = np.eye(3)
        for channel_index, name in enumerate(node.channels):
            if name.endswith("position"):
                local_position[:, "XYZ".index(name[0])] += frames[:, node.channel_start + channel_index]
        if node.parent < 0:
            positions[:, index] = local_position
            world_rotation[:, index] = local_rotation[:, index]
            rest[index] = node.offset
        else:
            positions[:, index] = positions[:, node.parent] + np.einsum("nij,nj->ni", world_rotation[:, node.parent], local_position)
            world_rotation[:, index] = world_rotation[:, node.parent] @ local_rotation[:, index]
            rest[index] = rest[node.parent] + node.offset
    source_height = float(np.ptp(rest[:, 1]))
    if source_height <= 0 or not math.isfinite(source_height):
        raise ValueError("BVH rest skeleton has no Y-up height")
    scale = float(target_height / source_height)
    return _Bvh(bones, frames, frame_time, local_rotation, world_rotation,
                positions * scale, rest * scale, scale)


def _rx_ry(pitch, roll):
    return Rotation.from_euler("XY", [pitch, roll]).as_matrix()


def _two_axis_angles(matrix):
    # Match the local up/down axis with Rx(pitch)*Ry(roll); yaw is deliberately
    # absent from the physical torso/ankle/shoulder articulation.
    return np.asarray([math.atan2(-matrix[1, 2], matrix[2, 2]),
                       math.atan2(matrix[0, 2], math.hypot(matrix[1, 2], matrix[2, 2]))])


def _down_axis_angles(direction):
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return np.asarray([math.atan2(direction[1], -direction[2]),
                       math.asin(float(np.clip(-direction[0], -1, 1)))])


def _box_bottom(model, data, geom_id):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return float(data.geom_xpos[geom_id, 2] - np.abs(rotation[2]).dot(model.geom_size[geom_id]))


@dataclass(frozen=True)
class ReferenceSample:
    joint_targets: np.ndarray
    root_position: np.ndarray
    root_velocity: np.ndarray
    root_quaternion: np.ndarray  # wxyz, Z-up; a reward target, never a command.
    phase: float
    foot_contacts: tuple[bool, bool]
    upright: float

    def to_dict(self):
        return {"joint_targets": self.joint_targets.tolist(), "root_position": self.root_position.tolist(),
                "root_velocity": self.root_velocity.tolist(), "root_quaternion": self.root_quaternion.tolist(),
                "phase": self.phase, "foot_contacts": list(self.foot_contacts), "upright": self.upright}


class MotionReference:
    """A calibrated target trajectory, with the source native path retained."""

    def __init__(self, kind: str, path=None, avatar_profile=None):
        if kind not in ("walk", "sit"):
            raise ValueError("Reference kind must be walk or sit")
        self.kind = kind
        self.avatar_profile = load_avatar_profile(avatar_profile)
        self.avatar_profile_sha256 = avatar_profile_sha256(self.avatar_profile)
        self.path = Path(path) if path is not None else _ASSETS / f"{kind}.bvh"
        source = self.path.read_bytes()
        source_sha256 = hashlib.sha256(source).hexdigest()
        manifest_path = _ASSETS / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        asset_record = next((entry for entry in manifest.get("assets", [])
                             if entry.get("kind") == kind and entry.get("sha256") == source_sha256), None)
        target_height = 1.8
        if self.avatar_profile is not None:
            physical = self.avatar_profile["physical"]
            target_height = float(physical.get("referenceHeight", physical.get(
                "height", 1.8 * float(physical["rootHeight"]) / .93)))
        bvh = _parse_bvh(source.decode("utf-8"), target_height=target_height)
        self.scale = bvh.scale
        self.frame_time = bvh.frame_time
        self.source_frame_count = len(bvh.frames)
        self.source_duration = (self.source_frame_count - 1) * self.frame_time
        self.source_root_positions = bvh.world_positions[:, 0] @ _BASIS.T
        self.source_root_positions.setflags(write=False)
        if kind == "walk":
            start, end = _WALK_LOOP
            if end >= self.source_frame_count:
                raise ValueError("The verified walking loop does not exist in this BVH")
        else:
            start, end = 0, self.source_frame_count - 1
        self.source_frame_range = (start, end)
        self.times = np.arange(end - start + 1) * self.frame_time
        self.duration = float(self.times[-1])
        selected = slice(start, end + 1)
        self.native_root_positions = self.source_root_positions[selected].copy()
        self.stride = self.native_root_positions[-1, :2] - self.native_root_positions[0, :2]
        self.native_heading = math.atan2(float(self.stride[0]), float(-self.stride[1])) if kind == "walk" else 0.
        root_matrices = _BASIS @ bvh.world_rotations[selected, 0] @ _BASIS.T
        root_rotations = Rotation.from_matrix(root_matrices)
        self._root_slerp = Slerp(self.times, root_rotations)
        # Cache the shortest-arc representation once. SciPy remains available as
        # an independent sampling oracle without constructing rotations per step.
        self._root_quaternions = root_rotations.as_quat()
        for index in range(1, len(self._root_quaternions)):
            if self._root_quaternions[index - 1].dot(self._root_quaternions[index]) < 0:
                self._root_quaternions[index] *= -1
        dots = np.einsum("ij,ij->i", self._root_quaternions[:-1], self._root_quaternions[1:])
        self._root_angles = np.arccos(np.clip(dots, 0, 1))
        self._root_inverse_sines = np.zeros_like(self._root_angles)
        np.divide(1., np.sin(self._root_angles), out=self._root_inverse_sines,
                  where=self._root_angles >= 1e-8)
        indices = {bone.name: index for index, bone in enumerate(bvh.bones)}
        required = ["Hips", "Spine4", "Head"] + [side + part for side in ("Left", "Right")
                    for part in ("UpLeg", "Leg", "Foot", "Arm", "ForeArm", "Hand")]
        missing = [name for name in required if name not in indices]
        if missing:
            raise ValueError(f"BVH lacks required major joints: {missing}")
        model = mujoco.MjModel.from_xml_string(build_humanoid_xml(avatar_profile=self.avatar_profile))
        self.body_model_sha256 = hashlib.sha256(
            build_humanoid_xml(avatar_profile=self.avatar_profile).encode()).hexdigest()
        data = mujoco.MjData(model)
        q_indices = np.asarray([model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES])
        feet = [model.geom(f"humanoid_{side}_foot").id for side in ("left", "right")]
        frame_count = len(self.times)
        self.joint_targets = np.zeros((frame_count, len(JOINT_NAMES)))
        unclipped = np.zeros_like(self.joint_targets)
        self.root_positions = self.native_root_positions.copy()
        self.root_positions[:, :2] -= self.root_positions[0, :2]
        self.foot_heights = np.zeros((frame_count, 2))
        rotation_errors = {name: [] for name in ("torso", "left_hip", "right_hip", "left_ankle", "right_ankle")}
        errors = {name: [] for name in required if name != "Hips"}
        # The original generic diagnostic used a simplified chest sample. Keep
        # that oracle unchanged; a measured body uses its actual torso extent.
        chest_sample = np.asarray([0., 0., .27])
        if self.avatar_profile is not None:
            torso_geom = model.geom("humanoid_torso").id
            chest_sample = model.geom_pos[torso_geom] + np.asarray([0., 0., model.geom_size[torso_geom, 1]])
            torso_joint = self.avatar_profile["physical"].get("joints", {}).get("spine4")
            spine_joint = self.avatar_profile["physical"].get("joints", {}).get("spine")
            if torso_joint is not None and spine_joint is not None:
                def measured_position(value):
                    return np.asarray(value["position"] if isinstance(value, dict) else value)
                chest_sample = measured_position(torso_joint) - measured_position(spine_joint)

        def set_angles(target, names, values):
            loc = [JOINT_NAMES.index(name) for name in names]
            unclipped[frame_index, loc] = values
            target[loc] = np.clip(values, JOINT_LIMITS[loc, 0], JOINT_LIMITS[loc, 1])
            return target[loc]

        for frame_index, source_index in enumerate(range(start, end + 1)):
            target = self.joint_targets[frame_index]
            root_source = bvh.world_rotations[source_index, 0]
            # Relative rotations and directions are expressed in pelvis axes.
            def relative_rotation(name):
                return _BASIS @ (root_source.T @ bvh.world_rotations[source_index, indices[name]]) @ _BASIS.T

            def relative_direction(first, second):
                delta = bvh.world_positions[source_index, indices[second]] - bvh.world_positions[source_index, indices[first]]
                return _BASIS @ (root_source.T @ delta)

            chest = relative_rotation("Spine4")
            torso_angles = set_angles(target, ["torso_pitch", "torso_roll"], _two_axis_angles(chest))
            physical_chest = _rx_ry(*torso_angles)
            rotation_errors["torso"].append(float(Rotation.from_matrix(physical_chest.T @ chest).magnitude()))
            for capital, side in (("Left", "left"), ("Right", "right")):
                hip_matrix = relative_rotation(capital + "UpLeg")
                hip_angles = Rotation.from_matrix(hip_matrix).as_euler("XYZ")
                hip_angles = set_angles(target, [side + "_hip_" + axis for axis in ("pitch", "roll", "yaw")], hip_angles)
                physical_hip = Rotation.from_euler("XYZ", hip_angles).as_matrix()
                rotation_errors[side + "_hip"].append(float(Rotation.from_matrix(physical_hip.T @ hip_matrix).magnitude()))
                shank_direction = physical_hip.T @ relative_direction(capital + "Leg", capital + "Foot")
                knee = set_angles(target, [side + "_knee"], [math.atan2(shank_direction[1], -shank_direction[2])])[0]
                physical_shank = physical_hip @ Rotation.from_euler("x", knee).as_matrix()
                foot_matrix = physical_shank.T @ relative_rotation(capital + "Foot")
                ankle_angles = set_angles(target, [side + "_ankle_pitch", side + "_ankle_roll"], _two_axis_angles(foot_matrix))
                rotation_errors[side + "_ankle"].append(float(Rotation.from_matrix(_rx_ry(*ankle_angles).T @ foot_matrix).magnitude()))

                # BVH arms bind along +/-X, physics arms bind down -Z. Fitting
                # segment direction explicitly supplies that rest calibration.
                arm_direction = physical_chest.T @ relative_direction(capital + "Arm", capital + "ForeArm")
                arm_angles = set_angles(target, [side + "_shoulder_pitch", side + "_shoulder_roll"], _down_axis_angles(arm_direction))
                physical_arm = physical_chest @ _rx_ry(*arm_angles)
                forearm_direction = physical_arm.T @ relative_direction(capital + "ForeArm", capital + "Hand")
                set_angles(target, [side + "_elbow"], [math.atan2(-forearm_direction[1], -forearm_direction[2])])

            # This is a separate forward-kinematic proxy used to calibrate the
            # reference. No environment data or rollout root is modified.
            data.qpos[:] = model.qpos0
            data.qpos[:3] = 0
            mujoco.mju_mat2Quat(data.qpos[3:7], root_matrices[frame_index].ravel())
            data.qpos[q_indices] = target
            mujoco.mj_forward(model, data)
            bottom = np.asarray([_box_bottom(model, data, foot) for foot in feet])
            self.root_positions[frame_index, 2] = _CLEARANCE - float(bottom.min())
            self.foot_heights[frame_index] = bottom + self.root_positions[frame_index, 2]
            body_samples = {"Spine4": data.xpos[model.body("torso").id] + data.xmat[model.body("torso").id].reshape(3, 3) @ chest_sample,
                            "Head": data.xpos[model.body("head").id]}
            for capital, side in (("Left", "left"), ("Right", "right")):
                for source_name, body_name in (("UpLeg", "thigh"), ("Leg", "shank"), ("Foot", "foot"),
                                               ("Arm", "upper_arm"), ("ForeArm", "forearm")):
                    body_samples[capital + source_name] = data.xpos[model.body(f"{side}_{body_name}").id]
                forearm_id = model.body(f"{side}_forearm").id
                forearm_length = (.255 if self.avatar_profile is None else
                                  float(self.avatar_profile["physical"]["arms"][side]["forearmLength"]))
                body_samples[capital + "Hand"] = data.xpos[forearm_id] + data.xmat[forearm_id].reshape(3, 3) @ [0, 0, -forearm_length]
            source_relative = (bvh.world_positions[source_index] - bvh.world_positions[source_index, 0]) @ _BASIS.T
            for name, position in body_samples.items():
                errors[name].append(float(np.linalg.norm(position - source_relative[indices[name]])))

        self.root_velocities = np.gradient(self.root_positions, self.frame_time, axis=0)
        self.native_root_velocities = np.gradient(self.native_root_positions, self.frame_time, axis=0)
        self.contacts = self.foot_heights < .025
        self._sample_frames = np.concatenate((self.joint_targets, self.root_positions, self.root_velocities), axis=1)
        self._sample_slopes = np.diff(self._sample_frames, axis=0) / np.diff(self.times)[:, None]
        clipped = np.abs(unclipped - self.joint_targets)
        per_joint = {name: {"sourceTargetMin": float(unclipped[:, i].min()),
                            "sourceTargetMax": float(unclipped[:, i].max()),
                            "clippedFrameCount": int(np.count_nonzero(clipped[:, i] > 1e-6)),
                            "maxLimitCorrectionRadians": float(clipped[:, i].max())}
                     for i, name in enumerate(JOINT_NAMES)}
        self.diagnostics = {
            "version": 1, "kind": kind, "sourcePath": str(self.path),
            "sourceSha256": source_sha256,
            "provenance": "cached_t2m_generated" if asset_record else "provided_bvh_provenance_unverified",
            "originalGenerationSourcePath": asset_record.get("source_path") if asset_record else None,
            "generationPrompt": manifest.get("analysis", {}).get("poses", {}).get(kind, {}).get("generationPrompt") if asset_record else None,
            "sourceManifestHashMatches": asset_record is not None,
            "sourceFrameCount": self.source_frame_count, "sourceFrameRange": [start, end],
            "sourceFrameTimeSeconds": self.frame_time, "durationSeconds": self.duration,
            "normalizationScale": self.scale, "sourceRestHeightMeters": target_height / self.scale,
            "normalizationTargetHeightMeters": target_height,
            "avatarProfileSha256": self.avatar_profile_sha256,
            "bodyModelSha256": self.body_model_sha256,
            "coordinates": "Z-up, forward -Y, quaternion wxyz", "jointNames": list(JOINT_NAMES),
            "nativeHeadingRadians": self.native_heading, "strideMeters": float(np.linalg.norm(self.stride)),
            "speedMetersPerSecond": float(np.linalg.norm(self.stride) / self.duration),
            "sourceNativeRootTrajectoryPreserved": True, "rootPositionUsedAsControl": False,
            "referenceOnly": True, "isPhysicalRollout": False,
            "groundCalibration": {"method": "minimum actual physical foot-box bottom at .003m",
                                  "clearanceMeters": _CLEARANCE,
                                  "rootHeightRange": [float(self.root_positions[:, 2].min()), float(self.root_positions[:, 2].max())],
                                  "maxNativeHeightCorrectionMeters": float(np.max(np.abs(self.root_positions[:, 2] - self.native_root_positions[:, 2])))},
            "positionFitMeters": {name: {"rms": float(np.sqrt(np.mean(np.square(values)))), "max": float(np.max(values))}
                                  for name, values in errors.items()},
            "rotationFitRadians": {name: {"rms": float(np.sqrt(np.mean(np.square(values)))), "max": float(np.max(values))}
                                   for name, values in rotation_errors.items()},
            "jointLimits": per_joint,
            "limitations": ["physical rig omits fingers, shoulder yaw, neck motion and toe joints",
                            "source BVH and simplified physical segment lengths differ; fitted directions do not guarantee identical world joint positions",
                            "contact phases and grounded height are kinematic reference estimates, not simulated contact",
                            "a residual policy must learn to follow these targets under real contact dynamics"],
        }
        for array in (self.times, self.native_root_positions, self.joint_targets,
                      self.root_positions, self.root_velocities, self.native_root_velocities,
                      self.foot_heights, self.contacts, self.stride, self._sample_frames,
                      self._sample_slopes, self._root_quaternions, self._root_angles,
                      self._root_inverse_sines):
            array.setflags(write=False)

    def sample(self, time_seconds: float) -> ReferenceSample:
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("Reference time must be finite and nonnegative")
        if self.kind == "walk":
            cycles = math.floor(time_seconds / self.duration)
            local_time = time_seconds - cycles * self.duration
        else:
            cycles = 0
            local_time = min(time_seconds, self.duration)
        if not 0 <= local_time <= self.duration:
            if -1e-10 < local_time < self.duration + 1e-10:
                # Cycle subtraction may miss an endpoint by a float ulp during
                # a long rollout. Clamp only this residual, retaining period,
                # cycle count, native stride and ordinary interpolation times.
                local_time = min(max(local_time, 0.), self.duration)
            else:
                raise ValueError("Interpolation time is outside the reference interval")
        phase = local_time / self.duration
        # Uniform BVH frame timing supplies the bracket in constant time. The
        # adjacent correction respects the actual float frame values at a knot.
        index = min(int(local_time / self.frame_time), len(self.times) - 2)
        if local_time < self.times[index]:
            index -= 1
        elif local_time > self.times[index + 1]:
            index += 1
        elapsed = local_time - self.times[index]
        fraction = elapsed / (self.times[index + 1] - self.times[index])
        if fraction == 0:
            values = self._sample_frames[index].copy()
        elif fraction == 1:
            values = self._sample_frames[index + 1].copy()
        else:
            values = self._sample_frames[index] + elapsed * self._sample_slopes[index]
        joint_count = len(JOINT_NAMES)
        targets = values[:joint_count]
        position = values[joint_count:joint_count + 3]
        position[:2] += cycles * self.stride
        velocity = values[joint_count + 3:]
        if self.kind == "sit" and time_seconds > self.duration:
            velocity.fill(0)
        first, second = self._root_quaternions[index:index + 2]
        angle = self._root_angles[index]
        if angle < 1e-8:
            quaternion_xyzw = first + fraction * (second - first)
        else:
            quaternion_xyzw = (math.sin((1 - fraction) * angle) * first
                               + math.sin(fraction * angle) * second) * self._root_inverse_sines[index]
        quaternion_xyzw /= np.linalg.norm(quaternion_xyzw)
        nearest = min(int(round(local_time / self.frame_time)), len(self.times) - 1)
        return ReferenceSample(targets, position, velocity, quaternion_xyzw[[3, 0, 1, 2]], phase,
                               tuple(bool(value) for value in self.contacts[nearest]),
                               float(1 - 2 * (quaternion_xyzw[0] ** 2 + quaternion_xyzw[1] ** 2)))


def load_motion_reference(kind="walk", path=None, avatar_profile=None) -> MotionReference:
    return MotionReference(kind, path, avatar_profile=avatar_profile)
