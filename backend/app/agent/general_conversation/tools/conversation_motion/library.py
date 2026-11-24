"""意味の層のクリップライブラリ(build_library.py で作った assets を読む)。"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bvh import Motion, Skeleton, forward_kinematics, parse_bvh, qinv, qmul, qnormalize
from .groups import group_joint_indices

ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "conversation_motion"

# 部位ごとの重み: 動いた角度(度)がこの範囲で 0 → 1 に上がる。
# 右手の動作で左腕が少し揺れる程度なら、リズムの層の左腕を残すため。
GROUP_WEIGHT_RANGE = {
    "right_arm": (12.0, 40.0),
    "left_arm": (12.0, 40.0),
    "head": (5.0, 12.0),
    "spine": (5.0, 12.0),
}

# 全身で行う動作(腰から曲げる)。それ以外は上半身だけ重ねる。
FULL_BODY = {"MOT_711"}
# 既定の振幅。収録のお辞儀は 80 度近い最敬礼なので、会話では 4 割程度にする。うなずき(約 36 度)は 7 割。
DEFAULT_AMPLITUDE = {"MOT_711": 0.4, "MOT_702": 0.7}


@dataclass
class Clip:
    id: str
    label: str
    source: str
    motion: Motion
    stroke: float
    """動作の山場(クリップ先頭からの秒)"""
    weights: dict[str, float]
    """部位 → 重み(0..1)"""
    full_body: bool
    entry: dict
    torso_yaw: np.ndarray
    """(T, 4) 胸の向きの、クリップ先頭からの鉛直軸まわりの回転。体ごと向けて指す動作のため"""
    amplitude: float = 1.0

    @property
    def duration(self) -> float:
        return self.motion.duration


def _smooth01(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def group_weights(groups: dict, full_body: bool) -> dict[str, float]:
    weights = {}
    for group, (low, high) in GROUP_WEIGHT_RANGE.items():
        value = groups.get(group, {}).get("rangeDeg", 0.0)
        weights[group] = _smooth01((value - low) / (high - low))
    if full_body:
        weights["spine"] = max(weights["spine"], 1.0)
        weights["hips"] = 1.0
    return weights


def yaw_only(q: np.ndarray) -> np.ndarray:
    """鉛直(y)軸まわりの回転成分だけを取り出す。"""
    out = np.zeros_like(q)
    out[..., 1] = q[..., 1]
    out[..., 3] = q[..., 3]
    return qnormalize(out)


def torso_yaw(motion: Motion) -> np.ndarray:
    """胸(Spine4)の向きの、先頭フレームからの鉛直軸まわりの変化 (T, 4)。"""
    _, rotations = forward_kinematics(motion)
    chest = rotations[:, motion.skeleton.index("Spine4")]
    return yaw_only(qmul(chest, qinv(chest[0])[None]))


def mirror_motion(motion: Motion) -> Motion:
    """左右反転(正中面 x=0 に対する鏡映)。右手の収録から左手の動作を作る。"""
    names = motion.skeleton.names
    swap = [
        names.index(
            n.replace("Left", "@").replace("Right", "Left").replace("@", "Right")
        )
        for n in names
    ]
    quats = motion.quats[:, swap] * np.array([1.0, -1.0, -1.0, 1.0])
    root = motion.root.copy()
    root[:, 0] *= -1.0
    return Motion(motion.skeleton, root, quats, motion.fps)


class ClipLibrary:
    def __init__(self, directory: Path = ASSET_DIR) -> None:
        catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
        data = np.load(directory / "library.npz")
        skeleton_motion = parse_bvh((directory / "skeleton.bvh").read_text())
        self.skeleton: Skeleton = skeleton_motion.skeleton
        self.fps = float(catalog["fps"])
        self.neutral = data["neutral"].astype(np.float64)
        self.root_height = float(catalog["rootHeight"])
        self.groups = group_joint_indices(self.skeleton)
        root, quats, offsets = data["root"], data["quats"], data["offsets"]
        self.clips: dict[str, Clip] = {}
        for entry, start, stop in zip(catalog["clips"], offsets[:-1], offsets[1:]):
            motion = Motion(
                self.skeleton,
                root[start:stop].astype(np.float64),
                quats[start:stop].astype(np.float64),
                self.fps,
            )
            full_body = entry["id"] in FULL_BODY
            self.clips[entry["id"]] = Clip(
                entry["id"],
                entry["label"],
                entry["source"],
                motion,
                float(entry["stroke"]),
                group_weights(entry["groups"], full_body),
                full_body,
                entry,
                torso_yaw(motion),
                DEFAULT_AMPLITUDE.get(entry["id"], 1.0),
            )
        self._mirrored: dict[str, Clip] = {}

    def get(self, catalog_id: str | None, *, mirrored: bool = False) -> Clip | None:
        if not catalog_id:
            return None
        clip = self.clips.get(catalog_id)
        if clip is None or not mirrored:
            return clip
        if catalog_id not in self._mirrored:
            weights = dict(clip.weights)
            weights["right_arm"], weights["left_arm"] = (
                clip.weights["left_arm"],
                clip.weights["right_arm"],
            )
            mirrored_motion = mirror_motion(clip.motion)
            self._mirrored[catalog_id] = Clip(
                clip.id,
                clip.label,
                clip.source,
                mirrored_motion,
                clip.stroke,
                weights,
                clip.full_body,
                clip.entry,
                torso_yaw(mirrored_motion),
                clip.amplitude,
            )
        return self._mirrored[catalog_id]

    def neutral_motion(self, frames: int) -> Motion:
        root = np.tile([[0.0, self.root_height, 0.0]], (frames, 1))
        return Motion(
            self.skeleton, root, np.tile(self.neutral, (frames, 1, 1)), self.fps
        )


_library: ClipLibrary | None = None
_lock = threading.Lock()


def get_library() -> ClipLibrary:
    global _library
    if _library is None:
        with _lock:
            if _library is None:
                _library = ClipLibrary()
    return _library


def library_available() -> bool:
    return (ASSET_DIR / "catalog.json").exists() and (
        ASSET_DIR / "library.npz"
    ).exists()
