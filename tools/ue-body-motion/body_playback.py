"""Transient Body-only playback in the currently open MetaHuman Level Sequence."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import unreal

from gesture_pose import GesturePose, rotate_toward
from pose_stream import PoseStream
from retarget import Clip, add, dot, prepare_retarget, scale, sub


@dataclass
class Window:
    start: int
    end: int
    interval: float
    animation: unreal.AnimSequence


@dataclass
class PreparedMotion:
    body: unreal.SkeletalMeshComponent
    template: unreal.AnimSequence
    clip: Clip
    gesture: GesturePose
    reference: dict
    scales: dict
    fixed: set
    stream: PoseStream
    window: Window = None
    pending: Window = None
    builder: object = None
    last_update: float = 0.0

    def close(self):
        # A suspended writer owns an open UE controller bracket. Always close
        # it on the editor thread when a clip expires or gets superseded.
        self.stream.close()
        if self.builder is not None:
            builder, self.builder = self.builder, None
            builder.close()


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


def _duplicate(animation):
    return unreal.SystemLibrary.duplicate_object(animation, unreal.find_object(None, "/Engine/Transient"))


def _empty_template(seed):
    # Strip the potentially long Sequencer idle once per seed, before making
    # any playback windows. Duplicating and stripping it on every beat stalls Slate.
    asset = _duplicate(seed)
    controller = asset.controller
    controller.open_bracket("Initialize conversation Body template", False)
    try:
        controller.remove_all_bone_tracks(False)
        controller.set_number_of_frames(unreal.FrameNumber(1), False)
    finally:
        controller.close_bracket(False)
    return asset


def _write_animation(motion, poses):
    animation = _duplicate(motion.template)
    controller = animation.controller
    key_count = len(poses)
    driven = poses[0][0]
    controller.open_bracket("Prepare conversation Body motion", False)
    try:
        controller.set_number_of_frames(unreal.FrameNumber(key_count - 1), False)
        for name in motion.fixed.union(driven):
            if not controller.add_bone_curve(name, False):
                raise RuntimeError(f"Cannot create Body bone track {name}")
            _, translation, rotation = motion.reference[name]
            keys = ([unreal.Quat(*rotations[name]) for rotations, _ in poses]
                    if name in driven else [unreal.Quat(*rotation)])
            positions = ([unreal.Vector(*p) for _, p in poses] if name == "pelvis"
                         else [unreal.Vector(*translation)] * len(keys))
            if not controller.set_bone_track_keys(name, positions, keys, [motion.scales[name]] * len(keys), False):
                raise RuntimeError(f"Cannot populate Body bone track {name}")
            yield
    finally:
        controller.close_bracket(False)
    return animation


def _prepare_window(motion, start, tick_interval):
    stream = motion.stream
    end = stream.submit(start, tick_interval, first=motion.window is None)
    while not stream.future.done():
        yield False  # Leave the editor thread immediately while the numeric worker runs.
    poses = stream.future.result()
    animation = yield from _write_animation(motion, poses)
    return Window(start, end, animation.get_play_length() / max(1, end - start), animation)


def _advance_builder(builder, budget):
    deadline = time.perf_counter() + budget
    try:
        while True:
            if next(builder) is False or time.perf_counter() >= deadline:
                return None
    except StopIteration as result:
        return result.value


class BodyPlayback:
    """prepare/show/clear must run on the UE editor thread (Slate callback)."""

    def __init__(self):
        self.workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gakucho-pose")
        self.body = None
        self.seed = self.template = None
        self.next_warmup = 0.0
        self.release_builder = None
        self.previous_override = None
        self.previously_disabled = False
        self.instance = None
        self.init_callback = None
        self.rebound = False
        self.animation = None
        self.play_rate = self.position = 0.0
        self.active = None
        self.idle = None
        self.release_started = None
        self.release_length = 0.0
        self.release_asset = None

    def _set_animation(self, animation, rate):
        self.instance.set_animation_asset(animation, False, rate)
        self.instance.set_playing(rate > 0)
        self.animation, self.play_rate = animation, rate

    def _set_position(self, position):
        self.instance.set_position(position, False)
        self.position = position

    def _on_anim_initialized(self):
        # Sequencer can reinitialize Body when it changes a Face track. Restore
        # the owned animation before the next evaluation of the replacement.
        if self.animation is None:
            return
        current = self.body.get_post_process_instance()
        if current == self.instance or not isinstance(current, unreal.AnimSingleNodeInstance):
            return
        self.instance = current
        self._set_animation(self.animation, self.play_rate)
        self._set_position(self.position)
        self.rebound = True

    def refresh_instance(self):
        if self.body is None:
            return False
        if not unreal.SystemLibrary.is_valid(self.body):
            self.restore()
            return False
        # Some editor reinitialization paths do not broadcast the delegate.
        self._on_anim_initialized()
        defer_capture, self.rebound = self.rebound, False
        return defer_capture

    def warmup(self):
        # Startup scripts can run before Sequencer spawns the Body. Prepare
        # the reusable empty sequence during idle time, ahead of the first clip.
        if self.template is not None or time.monotonic() < self.next_warmup:
            return
        self.next_warmup = time.monotonic() + 2
        try:
            _, seed = _find_body()
        except RuntimeError:
            return
        self.template = _empty_template(seed)
        self.seed = seed

    def prepare(self, clip, whole_body=False):
        body, seed = _find_body()
        if seed != self.seed:
            self.template = _empty_template(seed)
            self.seed = seed
        reference, _ = _skeleton(body)
        entry, scales = _skeleton(body, evaluated=True)
        idle = self.idle if body == self.body and self.idle is not None else entry
        mapper = prepare_retarget(clip, reference)
        gesture = GesturePose(mapper, idle, entry, whole_body, reference)
        # Bones without tracks default to the mesh's reference pose. Retain
        # tracks wherever the live idle differs, including the component root.
        fixed = set()
        for name, (_, pos, q) in idle.items():
            diff = sub(pos, reference[name][1])
            if dot(diff, diff) > 0.0001 or abs(dot(q, reference[name][2])) < 0.99999:
                fixed.add(name)
        stream = PoseStream(self.workers, gesture, clip)
        return PreparedMotion(body, self.template, clip, gesture, idle, scales, fixed, stream)

    def show(self, motion, frame_index):
        body = motion.body
        if not unreal.SystemLibrary.is_valid(body):
            raise RuntimeError("The Sequencer Body has despawned")
        now = time.monotonic()
        tick_interval = now - motion.last_update if motion.last_update else 1 / 30
        motion.last_update = now
        inside = motion.window and frame_index <= motion.window.end
        attach_window = False
        if not inside and motion.pending and frame_index <= motion.pending.end:
            motion.window, motion.pending = motion.pending, None
            attach_window = True
            inside = True

        # Build the next sequence on the editor thread in small slices. Only
        # attach a completed asset on a later tick, after UE evaluates it.
        if motion.pending and frame_index > motion.pending.end:
            motion.pending = None
        more = not inside or motion.window.end < motion.clip.frame_count - 1
        if more and motion.pending is None:
            if motion.builder is None:
                start = motion.window.end if inside else frame_index
                motion.builder = _prepare_window(motion, start, tick_interval)
            budget = max(0.005 if inside else 0.009, min(0.020, tick_interval * 0.16))
            motion.pending = _advance_builder(motion.builder, budget)
            if motion.pending is not None:
                motion.builder = None
        if not inside:
            return False
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
                self.init_callback = self._on_anim_initialized
                body.on_anim_initialized.add_callable(self.init_callback)
                self.idle = motion.reference
                attach_window = True
            except Exception:
                self.restore()
                raise
        self._cancel_release()
        self.active = motion
        window = motion.window
        if attach_window:
            self._set_animation(window.animation, window.interval / motion.clip.frame_time)
        self._set_position((frame_index - window.start) * window.interval)
        return True

    def clear(self):
        if self.active is None or self.release_started is not None:
            return
        try:
            motion = self.active
            self.instance.set_playing(False)
            self.play_rate = 0.0
            current, _ = _skeleton(motion.body, evaluated=True)
            driven = motion.gesture.first.keys()
            start = {name: current[name][2] for name in driven}
            end = {name: self.idle[name][2] for name in driven}
            # The shortest rotation from a crossed arm back to the side can
            # pass through the jacket even when both endpoint poses are safe.
            poses = []
            start_pos = current["pelvis"][1]
            delta = sub(self.idle["pelvis"][1], start_pos)
            for i in range(9):
                t = i / 8
                pose = {n: rotate_toward(start[n], end[n], t) for n in driven}
                if motion.gesture.contact and 0 < i < 8:
                    motion.gesture.contact.correct(pose)
                poses.append((pose, add(start_pos, scale(delta, t))))
            self.release_builder = _write_animation(motion, poses)
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
        if self.release_builder is not None:
            try:
                self.release_asset = _advance_builder(self.release_builder, 0.005)
            except Exception:
                self.restore()
                raise
            if self.release_asset is not None:
                self.release_builder = None
                self.release_length = self.release_asset.get_play_length()
            return
        if self.release_asset is not None:
            self._set_animation(self.release_asset, 0.0)
            self.release_asset = None
            self.release_started = time.monotonic()
        t = min((time.monotonic() - self.release_started) / 0.32, 1.0)
        if t >= 1:
            self.restore()
        else:
            self._set_position(self.release_length * t * t * (3 - 2 * t))

    def _cancel_release(self):
        if self.release_builder is not None:
            self.release_builder.close()
            self.release_builder = None
        self.release_started = self.release_asset = None

    def restore(self):
        self._cancel_release()
        body = self.body
        if body is not None and unreal.SystemLibrary.is_valid(body):
            if self.init_callback is not None:
                body.on_anim_initialized.remove_callable(self.init_callback)
            body.set_override_post_process_anim_bp(self.previous_override)
            body.disable_post_process_blueprint = self.previously_disabled
        self.init_callback = None
        self.rebound = False
        self.body = self.previous_override = self.instance = self.active = None
        self.animation = None
        self.play_rate = self.position = 0.0
        self.idle = self.release_started = self.release_asset = None
