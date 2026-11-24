"""会話画面(BvhViewer)の待機姿勢と同じ立ち姿と、収録データをその立ち姿へ載せ替える処理。

BvhViewer の待機姿勢は「T ポーズ + 部位ごとの固定角」(index.tsx の createAvatarAnimationState)。
現行方式の T2M 出力も学長の GLB 上で自然に見えるのはこの約束事に近いからで、
モーキャプ(被験者 005)をそのまま使うと、上腕と前腕が逆向きに 40〜60 度ねじれ、手首が外へ折れて見える。

そこで収録の各関節を「被験者自身の立ち姿からのワールド回転の差分」として取り出し、
この立ち姿の上に載せ直す。動作の形(骨の向きの変化)は保ち、ねじれと手首の癖だけを入れ替える。
指は手に対する形そのものに意味がある(OK・ピースなど)ので、収録のローカル回転をそのまま使う。
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import (
    Motion,
    Skeleton,
    forward_kinematics,
    local_from_global,
    qinv,
    qmul,
    qnormalize,
    qpow,
)

# BvhViewer の neutralOffset(度、three.js の Euler 'XYZ')。side は左 -1・右 +1。
VIEWER_OFFSETS = {
    "root": lambda s: (-1.5, 0.0, 0.0),
    "hips": lambda s: (1.5, 0.0, 0.0),
    "chest": lambda s: (-1.2, 0.0, 0.0),
    "shoulder": lambda s: (0.0, s * 2.0, s * 3.0),
    "upperArm": lambda s: (2.0, s * 5.0, s * 77.0),
    "lowerArm": lambda s: (0.0, s * 11.0, s * 3.0),
    "hand": lambda s: (-2.0, s * 3.0, 0.0),
    "thigh": lambda s: (-3.0, 0.0, 0.0),
    "shin": lambda s: (5.0, 0.0, 0.0),
    "foot": lambda s: (-2.0, 0.0, 0.0),
}
FINGERS = re.compile(r"(Thumb|Index|Middle|Ring|Pinky)")
# 手首のねじれ(手のひらを上・下に返す動き)のうち前腕へ移す割合。
# 収録では前腕の回内・回外が手首の関節に乗っており、そのままだと GLB の手首だけが 90 度ねじれて見える。
FOREARM_TWIST_SHARE = 0.6


def viewer_role(name: str, is_root: bool) -> str:
    """BvhViewer と同じ規則で部位を決める(正規表現の順番も合わせている)。"""
    n = name.lower()
    if is_root:
        return "root"
    for role, pattern in (
        ("hips", r"hips?|pelvis"),
        ("head", r"head"),
        ("neck", r"neck"),
        ("lowerArm", r"forearm|lowerarm"),
        ("upperArm", r"(left|right)(upper)?arm$"),
        ("hand", r"(left|right)(hand|wrist)$"),
        ("thigh", r"upleg|thigh"),
        ("shin", r"(left|right)(lowerleg|leg|shin)$"),
        ("foot", r"(left|right)(foot|ankle)$"),
        ("chest", r"chest|upperback|spine[ _-]?[34]"),
        ("spine", r"spine"),
        ("shoulder", r"shoulder|clavicle"),
    ):
        if re.search(pattern, n):
            return role
    return "other"


def viewer_neutral(skeleton: Skeleton, fingers: np.ndarray | None = None) -> np.ndarray:
    """BvhViewer の待機姿勢のローカル回転 (J, 4)。指だけは与えられた形(自然に曲げた指)を使う。"""
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(skeleton.joints), 1))
    for j, joint in enumerate(skeleton.joints):
        if fingers is not None and FINGERS.search(joint.name):
            quats[j] = fingers[j]
            continue
        role = viewer_role(joint.name, joint.parent < 0)
        if role not in VIEWER_OFFSETS:
            continue
        lower = joint.name.lower()
        side = (
            -1.0
            if re.search(r"left|^l[_-]", lower)
            else 1.0 if re.search(r"right|^r[_-]", lower) else 0.0
        )
        quats[j] = Rotation.from_euler(
            "XYZ", VIEWER_OFFSETS[role](side), degrees=True
        ).as_quat()
    return quats


def onto_neutral(motion: Motion, reference: np.ndarray, neutral: np.ndarray) -> Motion:
    """各関節のワールド回転の差分(reference からの変化)を neutral の上に載せ直す。"""
    skeleton = motion.skeleton
    root = motion.root[:1]
    _, rotations = forward_kinematics(motion)
    _, reference_global = forward_kinematics(
        Motion(skeleton, root, reference[None], motion.fps)
    )
    _, neutral_global = forward_kinematics(
        Motion(skeleton, root, neutral[None], motion.fps)
    )
    result = qmul(qmul(rotations, qinv(reference_global)), neutral_global)
    for j, joint in enumerate(skeleton.joints):
        if FINGERS.search(joint.name):
            result[:, j] = qmul(result[:, joint.parent], motion.quats[:, j])
    converted = Motion(
        skeleton, motion.root.copy(), local_from_global(skeleton, result), motion.fps
    )
    return share_wrist_twist(converted)


def share_wrist_twist(motion: Motion, share: float = FOREARM_TWIST_SHARE) -> Motion:
    """手首(手の関節)にある前腕軸まわりのねじれの一部を前腕へ移す。手の向き(ワールド)は変わらない。"""
    motion = motion.copy()
    for side in ("Left", "Right"):
        hand = motion.skeleton.index(f"{side}Hand")
        forearm = motion.skeleton.index(f"{side}ForeArm")
        axis = motion.skeleton.joints[hand].offset
        axis = axis / np.linalg.norm(axis)
        q = motion.quats[:, hand]
        twist = qnormalize(
            np.concatenate([(q[:, :3] @ axis)[:, None] * axis, q[:, 3:]], axis=1)
        )
        moved = qpow(twist, share)
        motion.quats[:, forearm] = qnormalize(qmul(motion.quats[:, forearm], moved))
        motion.quats[:, hand] = qnormalize(qmul(qinv(moved), q))
    return motion
