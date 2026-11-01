"""層の合成: 土台(待機またはリズム)の上に、意味の層の動作を重ねて 1 本の BVH にする。

    ③ 待機の層   : 立ち姿 + 呼吸・わずかな揺れ(常に流れる)
    ② リズムの層 : 発話音声から作った身振り(あれば③の代わり)
    ① 意味の層   : 行動タグの動作。動いた部位だけ、ストロークを語に合わせて重ねる

重ね方
  - 腕: 動作の姿勢そのものへ寄せる(手の高さや形が意味を持つため)
  - 首・頭・背骨: 動作の「始めからの変化分」を足す(うなずきの上でリズムの揺れが残る)
  - 腰・脚: 全身動作(お辞儀)のときだけ動かす
  - 入りと抜けは重みを滑らかに上げ下げし、前の姿勢から跳ばずに入る
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bvh import Motion, limit_rotation, qinv, qmul, qnormalize, qpow, qslerp
from .library import Clip, ClipLibrary

FADE_IN = 0.25
FADE_OUT = 0.45
# 収録者が体ごと向きを変えた分(腰のひねり)を、学長では背骨へ分けて足す割合。
# 腰と脚は動かさないので、腕は胴に対する向きのまま、胴ごと指す方向へ向く。
TORSO_TURN = 0.8
SPINE_CHAIN = ("Spine", "Spine1", "Spine2", "Spine3", "Spine4")
# 首・頭の動作(うなずき等)で、顔を上げる向きはこの角度まで(収録のうなずきは戻りで 17 度上を向く)
HEAD_MAX_UP = 5.0
START_BLEND = 0.35  # 前の発話の終わりの姿勢からつなぐ時間


@dataclass
class Overlay:
    clip: Clip
    start: float
    """発話の先頭から見たクリップ先頭の時刻(秒)。負なら前の発話から続いている"""
    length: float
    """使うクリップの長さ(秒)"""
    tag: str = ""
    amplitude: float = 1.0
    explicit: bool = False

    @property
    def end(self) -> float:
        return self.start + self.length

    @property
    def stroke_time(self) -> float:
        return self.start + self.clip.stroke


@dataclass
class Pose:
    root: np.ndarray
    quats: np.ndarray


@dataclass
class Composition:
    motion: Motion
    overlays: list[Overlay] = field(default_factory=list)
    base: str = "idle"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def idle_base(library: ClipLibrary, frames: int, clock: float) -> Motion:
    """待機の層: 立ち姿に、呼吸(胸の上下)と体重移動程度の揺れを足す。

    clock は会話全体の通し時刻。発話をまたいでも位相が連続するようにする。
    """
    motion = library.neutral_motion(frames)
    t = clock + np.arange(frames) / library.fps
    skeleton = library.skeleton

    def rotate(name: str, axis: int, degrees: np.ndarray) -> None:
        j = skeleton.index(name)
        vector = np.zeros((frames, 3))
        vector[:, axis] = np.radians(degrees)
        half = vector / 2
        angle = np.linalg.norm(half, axis=1, keepdims=True)
        q = np.concatenate([np.where(angle > 1e-9, np.sin(angle) / np.maximum(angle, 1e-9), 1.0) * half,
                            np.cos(angle)], axis=1)
        motion.quats[:, j] = qmul(motion.quats[:, j], q)

    breath = np.sin(2 * np.pi * 0.23 * t)
    sway = np.sin(2 * np.pi * 0.07 * t + 0.6)
    rotate("Spine2", 0, -0.8 * breath)
    rotate("Spine4", 0, 0.5 * breath)
    rotate("Hips", 2, 0.6 * sway)
    rotate("Spine", 2, -0.5 * sway)
    rotate("Neck", 1, 1.2 * np.sin(2 * np.pi * 0.05 * t + 1.3))
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        rotate(f"{side}Arm", 2, sign * 0.8 * breath)
    motion.root[:, 1] += 0.002 * breath
    return motion


def _sample(clip: Clip, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """クリップを任意の時刻で取り出す(範囲外は端を保持)。"""
    motion = clip.motion
    position = np.clip(times * motion.fps, 0, motion.frames - 1)
    i0 = np.floor(position).astype(int)
    i1 = np.minimum(i0 + 1, motion.frames - 1)
    alpha = position - i0
    quats = qslerp(motion.quats[i0], motion.quats[i1], alpha[:, None])
    root = motion.root[i0] * (1 - alpha[:, None]) + motion.root[i1] * alpha[:, None]
    return quats, root


def _sample_yaw(clip: Clip, times: np.ndarray) -> np.ndarray:
    position = np.clip(times * clip.motion.fps, 0, clip.motion.frames - 1)
    i0 = np.floor(position).astype(int)
    i1 = np.minimum(i0 + 1, clip.motion.frames - 1)
    return qslerp(clip.torso_yaw[i0], clip.torso_yaw[i1], position - i0)


def apply_overlay(motion: Motion, overlay: Overlay, library: ClipLibrary) -> None:
    fps = motion.fps
    first = max(int(np.floor(overlay.start * fps)), 0)
    last = min(int(np.ceil(overlay.end * fps)), motion.frames - 1)
    if last < first:
        return
    frames = np.arange(first, last + 1)
    local_time = frames / fps - overlay.start
    envelope = (_smoothstep(local_time / FADE_IN) * _smoothstep((overlay.length - local_time) / FADE_OUT)
                * overlay.amplitude * overlay.clip.amplitude)
    clip_quats, clip_root = _sample(overlay.clip, local_time)
    reference = overlay.clip.motion.quats[0]
    for group, (_main, members) in library.groups.items():
        weight = overlay.clip.weights.get(group, 0.0)
        if weight <= 0.0:
            continue
        w = envelope * weight
        for j in members:
            current = motion.quats[frames, j]
            if group in ("right_arm", "left_arm") or (group == "hips" and j != 0):
                target = clip_quats[:, j]  # 姿勢そのもの
            else:
                delta = qmul(clip_quats[:, j], qinv(reference[j])[None])  # 始めからの変化分
                if group == "head":
                    delta = limit_rotation(delta, 60.0, HEAD_MAX_UP / len(members))
                target = qmul(delta, current)
            motion.quats[frames, j] = qslerp(current, target, w)
        if group == "hips":
            displacement = clip_root - overlay.clip.motion.root[0]
            motion.root[frames] += displacement * w[:, None]
    # 腕の動作で収録者が胴ごと向きを変えていれば、その分を背骨へ分けて足す
    arm_weight = max(overlay.clip.weights.get("right_arm", 0.0), overlay.clip.weights.get("left_arm", 0.0))
    if not overlay.clip.full_body and arm_weight > 0.0:
        share = (envelope * arm_weight * TORSO_TURN / len(SPINE_CHAIN))
        step = qpow(_sample_yaw(overlay.clip, local_time), share)
        for name in SPINE_CHAIN:
            j = library.skeleton.index(name)
            motion.quats[frames, j] = qmul(step, motion.quats[frames, j])


def compose(library: ClipLibrary, duration: float, overlays: list[Overlay], *, base: Motion | None = None,
            start_pose: Pose | None = None, clock: float = 0.0) -> Composition:
    """1 発話ぶんの動作を作る。長さは発話と、はみ出す動作の終わりの長い方。"""
    fps = library.fps
    end = max([duration] + [o.end for o in overlays])
    frames = int(np.ceil(end * fps)) + 1
    if base is not None:
        motion = base.copy()
        if motion.frames < frames:
            # リズムの層が発話より短い分は、待機の層へ滑らかに戻す
            idle = idle_base(library, frames, clock)
            pad = frames - motion.frames
            last_q = np.repeat(motion.quats[-1:], pad, axis=0)
            last_r = np.repeat(motion.root[-1:], pad, axis=0)
            fade = _smoothstep(np.arange(1, pad + 1) / (0.5 * fps))[:, None]
            quats = qslerp(last_q, idle.quats[motion.frames:], fade)
            root = last_r * (1 - fade) + idle.root[motion.frames:] * fade
            motion = Motion(motion.skeleton, np.concatenate([motion.root, root]),
                            np.concatenate([motion.quats, quats]), fps)
        else:
            motion = motion.slice(0, frames)
        base_name = "rhythm"
    else:
        motion = idle_base(library, frames, clock)
        base_name = "idle"
    for overlay in sorted(overlays, key=lambda o: o.start):
        apply_overlay(motion, overlay, library)
    if start_pose is not None:
        count = min(int(START_BLEND * fps), frames)
        blend = _smoothstep(np.arange(count) / max(count - 1, 1))
        motion.quats[:count] = qslerp(np.repeat(start_pose.quats[None], count, axis=0), motion.quats[:count],
                                      blend[:, None])
        motion.root[:count] = start_pose.root * (1 - blend[:, None]) + motion.root[:count] * blend[:, None]
    motion.quats = qnormalize(motion.quats)
    return Composition(motion, list(overlays), base_name)


def pose_at(motion: Motion, seconds: float) -> Pose:
    index = int(np.clip(round(seconds * motion.fps), 0, motion.frames - 1))
    return Pose(motion.root[index].copy(), motion.quats[index].copy())

