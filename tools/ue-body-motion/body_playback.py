"""Transient Body-only playback in the currently open MetaHuman Level Sequence."""

from dataclasses import dataclass
from math import ceil

import unreal

from retarget import Clip, Retarget, prepare_retarget


@dataclass
class PreparedMotion:
    body: unreal.SkeletalMeshComponent
    seed: unreal.AnimSequence
    clip: Clip
    mapper: Retarget
    reference: dict
    scales: dict
    stride: int
    start: int = -1
    end: int = -1
    key_interval: float = 0.0
    animation: unreal.AnimSequence = None


def _find_body():
    editor = unreal.LevelSequenceEditorBlueprintLibrary
    sequence = editor.get_current_level_sequence()
    if sequence is None:
        raise RuntimeError("Open the MetaHuman Level Sequence before playing motion")
    candidates = []
    for binding in sequence.get_bindings():
        if str(binding.get_name()).lower() != "body":
            continue
        for body in editor.get_bound_objects(sequence.get_binding_id(binding)):
            if not isinstance(body, unreal.SkeletalMeshComponent):
                continue
            for track in binding.find_tracks_by_type(unreal.MovieSceneSkeletalAnimationTrack):
                for section in track.get_sections():
                    seed = section.params.animation
                    if isinstance(seed, unreal.AnimSequence) and seed.get_skeleton() == body.skeletal_mesh_asset.skeleton:
                        candidates.append((body, seed))
                        break
                else:
                    continue
                break
    if len(candidates) != 1:
        raise RuntimeError("Expected a single spawned Body with a compatible Sequencer animation")
    return candidates[0]


def _reference(body):
    reference, scales = {}, {}
    for index in range(body.get_num_bones()):
        name = str(body.get_bone_name(index))
        parent = str(body.get_parent_bone(name)) if index else None
        transform = body.get_ref_pose_transform(index)
        pos, rot = transform.translation, transform.rotation
        reference[name] = (parent, (pos.x, pos.y, pos.z), (rot.x, rot.y, rot.z, rot.w))
        scales[name] = transform.scale3d
    return reference, scales


def _prepare_window(motion, start):
    # A duplicate in /Engine/Transient inherits the skeleton. Never write to
    # the Sequencer animation, its sections, or a saved project package.
    transient_package = unreal.find_object(None, "/Engine/Transient")
    animation = unreal.SystemLibrary.duplicate_object(motion.seed, transient_package)
    controller = animation.controller
    # Bound work per editor tick, including when a late revision seeks into a long clip.
    remaining = motion.clip.frame_count - 1 - start
    key_count = max(2, min(32, ceil(remaining / motion.stride) + 1))
    rotations, pelvis = {}, []
    for index in range(key_count):
        local, translation = motion.mapper.pose(min(start + index * motion.stride, motion.clip.frame_count - 1))
        pelvis.append(unreal.Vector(*translation))
        for name, quaternion in local.items():
            rotations.setdefault(name, []).append(unreal.Quat(*quaternion))

    controller.open_bracket("Prepare conversation Body motion", False)
    try:
        controller.remove_all_bone_tracks(False)
        controller.set_number_of_frames(unreal.FrameNumber(key_count - 1), False)
        for name, keys in rotations.items():
            if not controller.add_bone_curve(name, False):
                raise RuntimeError(f"Cannot create Body bone track {name}")
            positions = pelvis if name == "pelvis" else [unreal.Vector(*motion.reference[name][1])] * key_count
            if not controller.set_bone_track_keys(name, positions, keys, [motion.scales[name]] * key_count, False):
                raise RuntimeError(f"Cannot populate Body bone track {name}")
    finally:
        controller.close_bracket(False)
    motion.start = start
    motion.end = start + (key_count - 1) * motion.stride
    motion.key_interval = animation.get_play_length() / (key_count - 1)
    motion.animation = animation


class BodyPlayback:
    """prepare/show/clear must run on the UE editor thread (Slate callback)."""

    def __init__(self):
        self.body = None
        self.previous_override = None
        self.previously_disabled = False
        self.instance = None
        self.active = None

    def prepare(self, clip):
        body, seed = _find_body()
        reference, scales = _reference(body)
        mapper = prepare_retarget(clip, reference)
        stride = max(1, ceil(1 / (30 * clip.frame_time) - 1e-4))
        return PreparedMotion(body, seed, clip, mapper, reference, scales, stride)

    def show(self, motion, frame_index):
        if self.active is not motion:
            self.clear()
            body = motion.body
            if not unreal.SystemLibrary.is_valid(body):
                raise RuntimeError("The Sequencer Body has despawned")
            previous = body.get_post_process_instance()
            previous_class = previous.get_class() if previous else None
            default_class = body.skeletal_mesh_asset.post_process_anim_blueprint
            self.previous_override = previous_class if previous_class != default_class else None
            self.previously_disabled = body.disable_post_process_blueprint
            self.body = body
            try:
                body.set_override_post_process_anim_bp(unreal.AnimSingleNodeInstance)
                body.disable_post_process_blueprint = False
                self.instance = body.get_post_process_instance()
                if not isinstance(self.instance, unreal.AnimSingleNodeInstance):
                    raise RuntimeError("Could not attach Body playback to the editor mesh")
                self.active = motion
            except Exception:
                self.clear()
                raise
        if not motion.start <= frame_index <= motion.end:
            _prepare_window(motion, frame_index)
            self.instance.set_animation_asset(motion.animation, False, 0)
            self.instance.set_playing(False)
        position = (frame_index - motion.start) / motion.stride * motion.key_interval
        self.instance.set_position(position, False)

    def clear(self):
        body = self.body
        if body is not None and unreal.SystemLibrary.is_valid(body):
            body.set_override_post_process_anim_bp(self.previous_override)
            body.disable_post_process_blueprint = self.previously_disabled
        self.body = self.previous_override = self.instance = self.active = None
