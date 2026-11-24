"""Transient Body-only playback in the currently open MetaHuman Level Sequence."""

from dataclasses import dataclass
from math import ceil
import time

import unreal

from gesture_pose import GesturePose
from retarget import Clip, dot, prepare_retarget, sub


@dataclass
class Window:
    start: int
    end: int
    interval: float
    animation: unreal.AnimSequence


@dataclass
class PreparedMotion:
    body: unreal.SkeletalMeshComponent
    seed: unreal.AnimSequence
    clip: Clip
    gesture: GesturePose
    reference: dict
    scales: dict
    fixed: set
    stride: int
    window: Window = None
    pending: Window = None


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


def _skeleton(body, *, evaluated=False):
    transforms, scales = {}, {}
    for index in range(body.get_num_bones()):
        name = str(body.get_bone_name(index))
        parent = str(body.get_parent_bone(name)) if index else None
        transform = (
            body.get_socket_transform(name, unreal.RelativeTransformSpace.RTS_PARENT_BONE_SPACE)
            if evaluated else body.get_ref_pose_transform(index)
        )
        pos, rot = transform.translation, transform.rotation
        transforms[name] = (parent, (pos.x, pos.y, pos.z), (rot.x, rot.y, rot.z, rot.w))
        scales[name] = transform.scale3d
    return transforms, scales


def _write_animation(motion, poses):
    # The source sequence supplies the compatible skeleton. Only its transient
    # duplicate is edited. Preserve the original idle transforms of undriven bones.
    transient_package = unreal.find_object(None, "/Engine/Transient")
    animation = unreal.SystemLibrary.duplicate_object(motion.seed, transient_package)
    controller = animation.controller
    key_count = len(poses)
    names = motion.fixed.union(poses[0][0])
    controller.open_bracket("Prepare conversation Body motion", False)
    try:
        controller.remove_all_bone_tracks(False)
        controller.set_number_of_frames(unreal.FrameNumber(key_count - 1), False)
        for name in names:
            if not controller.add_bone_curve(name, False):
                raise RuntimeError(f"Cannot create Body bone track {name}")
            _, translation, rotation = motion.reference[name]
            if name in poses[0][0]:
                keys = [unreal.Quat(*rotations[name]) for rotations, _ in poses]
            else:
                keys = [unreal.Quat(*rotation)] * key_count
            positions = ([unreal.Vector(*p) for _, p in poses] if name == "pelvis"
                         else [unreal.Vector(*translation)] * key_count)
            if not controller.set_bone_track_keys(name, positions, keys, [motion.scales[name]] * key_count, False):
                raise RuntimeError(f"Cannot populate Body bone track {name}")
    finally:
        controller.close_bracket(False)
    return animation


def _prepare_window(motion, start):
    remaining = motion.clip.frame_count - 1 - start
    key_count = max(2, min(32, ceil(remaining / motion.stride) + 1))
    poses = []
    previous = motion.window.end if motion.window else -1
    for index in range(key_count):
        frame = min(start + index * motion.stride, motion.clip.frame_count - 1)
        # At the first visible frame (including a late revision), start from
        # the pose on screen. Continue the same filter across subsequent windows.
        dt = 0 if previous < 0 else max(0, frame - previous) * motion.clip.frame_time
        poses.append(motion.gesture.sample(frame, dt))
        previous = frame
    animation = _write_animation(motion, poses)
    return Window(start, start + (key_count - 1) * motion.stride,
                  animation.get_play_length() / (key_count - 1), animation)


class BodyPlayback:
    """prepare/show/clear must run on the UE editor thread (Slate callback)."""

    def __init__(self):
        self.body = None
        self.previous_override = None
        self.previously_disabled = False
        self.instance = None
        self.active = None
        self.idle = None
        self.release_started = None
        self.release_length = 0.0
        self.release_asset = None
        self.release_armed = False

    def prepare(self, clip, whole_body=False):
        body, seed = _find_body()
        reference, _ = _skeleton(body)
        entry, scales = _skeleton(body, evaluated=True)
        idle = self.idle if body == self.body and self.idle is not None else entry
        mapper = prepare_retarget(clip, reference)
        gesture = GesturePose(mapper, idle, entry, whole_body)
        # Bones without tracks default to the mesh's reference pose. Retain
        # tracks wherever the live idle differs, including the component root.
        fixed = set()
        for name, (_, pos, q) in idle.items():
            diff = sub(pos, reference[name][1])
            if dot(diff, diff) > 0.0001 or abs(dot(q, reference[name][2])) < 0.99999:
                fixed.add(name)
        stride = max(1, ceil(1 / (30 * clip.frame_time) - 1e-4))
        return PreparedMotion(body, seed, clip, gesture, idle, scales, fixed, stride)

    def show(self, motion, frame_index):
        body = motion.body
        if not unreal.SystemLibrary.is_valid(body):
            raise RuntimeError("The Sequencer Body has despawned")
        if motion.window is None or frame_index > motion.window.end:
            if motion.pending is None or frame_index > motion.pending.end:
                # Keep the displayed pose this tick. Let UE evaluate the new
                # transient sequence before assigning it to the visible mesh.
                motion.pending = _prepare_window(motion, frame_index)
                return False
            motion.window, motion.pending = motion.pending, None
            attach_window = True
        else:
            attach_window = False
        if self.body != body or self.instance is None:
            self.restore()
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
                self.idle = motion.reference
            except Exception:
                self.restore()
                raise
        self.release_started = self.release_asset = None
        self.active = motion
        window = motion.window
        if attach_window:
            self.instance.set_animation_asset(window.animation, False, 0)
            self.instance.set_playing(False)
        position = (frame_index - window.start) / motion.stride * window.interval
        self.instance.set_position(position, False)
        # Stage the next window on an earlier tick without disturbing this one.
        if (motion.pending is None and window.end < motion.clip.frame_count - 1
                and (window.end - frame_index) * motion.clip.frame_time < 0.4):
            motion.pending = _prepare_window(motion, window.end)
        return True

    def clear(self):
        if self.active is None or self.release_started is not None:
            return
        try:
            motion = self.active
            current, _ = _skeleton(motion.body, evaluated=True)
            driven = motion.gesture.first.keys()
            start = {name: current[name][2] for name in driven}
            end = {name: self.idle[name][2] for name in driven}
            animation = _write_animation(motion, [
                (start, current["pelvis"][1]), (end, self.idle["pelvis"][1])
            ])
            self.release_asset = animation
            self.release_armed = False
            self.release_length = animation.get_play_length()
            self.release_started = time.monotonic()
        except Exception:
            self.restore()
            raise

    def advance_release(self):
        if self.release_started is None:
            return
        if not unreal.SystemLibrary.is_valid(self.body) or not unreal.SystemLibrary.is_valid(self.instance):
            self.restore()
            return
        if self.release_asset is not None:
            if not self.release_armed:
                self.release_armed = True
                return
            self.instance.set_animation_asset(self.release_asset, False, 0)
            self.instance.set_playing(False)
            self.release_asset = None
            self.release_started = time.monotonic()
        t = min((time.monotonic() - self.release_started) / 0.32, 1.0)
        if t >= 1:
            self.restore()
        else:
            self.instance.set_position(self.release_length * t * t * (3 - 2 * t), False)

    def restore(self):
        body = self.body
        if body is not None and unreal.SystemLibrary.is_valid(body):
            body.set_override_post_process_anim_bp(self.previous_override)
            body.disable_post_process_blueprint = self.previously_disabled
        self.body = self.previous_override = self.instance = self.active = None
        self.idle = self.release_started = self.release_asset = None
