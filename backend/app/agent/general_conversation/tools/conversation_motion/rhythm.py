"""リズムの層: SentiAvatar の発話同期モーションを学長の標準骨格へ移す。

SentiAvatar(研究評価専用・非商用ライセンス)は音声から全身動作を作る。移し方は部位で変える。

- 腕(上腕・前腕・手): **骨の向きを合わせる**。SuSu の上腕が前を向けば学長の上腕も前を向く。
  向きだけを最小の回転で合わせ、骨軸まわりのねじれは学長の立ち姿のまま保つ。
  以前は「SuSu の平均姿勢からの回転の差分」を足していたが、SuSu の平均姿勢は手を体の前に
  置いているため、手を下ろす動きがそのまま学長の(既に下ろした)腕を後ろへ回し、
  手が胸より 20cm 後ろへ行く・腕が背中側へ回る、という破綻が出た。
- 背骨・首・頭: 平均的な話し姿勢からの差分を弱めて足し、角度に上限を設ける(のけぞり防止)。
- 鎖骨: 差分を半分だけ足す(肩をすくめる程度)。
- 骨盤と脚: 動かさない(足が滑らないように)。
"""

from __future__ import annotations

import numpy as np

from .bvh import (Motion, Skeleton, forward_kinematics, limit_rotation, parse_bvh, qangle, qinv, qmean, qmul,
                  qnormalize, qpow, qrotate, resample)

PELVIS = "pelvis"
# 胴と頭は「合計の傾き」で抑える。関節ごとに上限を付けても、5 つの背骨と 3 つの首・頭で足し合わさると
# 大きくのけぞる(短い発話で 16 度あごが上がったままになった)。
CHEST_SCALE, CHEST_MAX = 0.5, 8.0          # 胸(背骨 5 関節に分配)
HEAD_SCALE, HEAD_MAX, HEAD_MAX_UP = 0.5, 8.0, 4.0   # 頭(胸に対して。首 2 + 頭 1 に分配。上向きは 4 度まで)
SPINE_CHAIN = ("Spine", "Spine1", "Spine2", "Spine3", "Spine4")
NECK_CHAIN = ("Neck", "Neck1", "Head")
# 鎖骨は差分を半分だけ(肩をすくめる程度)
DELTA_JOINTS = {"clavicle_l": ("LeftShoulder", 0.5, 8.0), "clavicle_r": ("RightShoulder", 0.5, 8.0)}
REFERENCE_JOINTS = ("spine_05", "head", "clavicle_l", "clavicle_r")
# 向きを合わせる骨: (標準骨格の関節, その骨の先の関節, SentiAvatar の関節, その骨の先の関節, 強さ)
DIRECTION_BONES = [
    (f"{side}{bone}", f"{side}{child}", f"{ue}_{s}", f"{ue_child}_{s}", weight)
    for side, s in (("Left", "l"), ("Right", "r"))
    for bone, child, ue, ue_child, weight in (
        ("Arm", "ForeArm", "upperarm", "lowerarm", 1.0),
        ("ForeArm", "Hand", "lowerarm", "hand", 1.0),
        ("Hand", "HandMiddle1", "hand", "middle_01", 0.6),
    )
]
JOINT_MAP = {"spine_05": "Spine4", "head": "Head",
             **{source: target for source, (target, _, _) in DELTA_JOINTS.items()},
             **{source: target for target, _, source, _, _ in DIRECTION_BONES}}


def _arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """単位ベクトル a を b へ向ける最小の回転 (..., 4)。"""
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    q = np.concatenate([cross, 1.0 + dot], axis=-1)
    # ほぼ逆向きのときは、a に直交する軸で 180 度回す
    opposite = (1.0 + dot[..., 0]) < 1e-6
    if np.any(opposite):
        axis = np.cross(a[opposite], np.array([0.0, 1.0, 0.0]))
        axis = np.where(np.linalg.norm(axis, axis=-1, keepdims=True) < 1e-6, np.array([1.0, 0.0, 0.0]), axis)
        q[opposite] = np.concatenate([axis, np.zeros((len(axis), 1))], axis=-1)
    return qnormalize(q)


def _limit(q: np.ndarray, scale: float, max_degrees: float) -> np.ndarray:
    q = qpow(q, scale) if abs(scale - 1.0) > 1e-6 else q
    angle = np.degrees(qangle(q))
    over = angle > max_degrees
    if np.any(over):
        q[over] = qpow(q[over], max_degrees / angle[over])
    return q


class RhythmRetargeter:
    def __init__(self, skeleton: Skeleton, neutral: np.ndarray, root_height: float, fps: float,
                 reference: dict | None = None) -> None:
        self.skeleton = skeleton
        self.neutral = neutral
        self.root_height = root_height
        self.fps = fps
        self.reference = reference  # SentiAvatar 側の平均話し姿勢(差分で移す関節の基準)

    @staticmethod
    def pelvis_frozen(motion: Motion) -> tuple[list[str], np.ndarray, np.ndarray]:
        """骨盤の向きを平均に固定したときの各関節のワールド回転と位置(骨盤原点)。"""
        positions, rotations = forward_kinematics(motion)
        names = motion.skeleton.names
        p = names.index(PELVIS)
        pelvis = rotations[:, p]
        mean_pelvis = qmean(pelvis, axis=0)
        fix = qmul(np.broadcast_to(mean_pelvis, pelvis.shape), qinv(pelvis))  # (T, 4)
        frozen_rot = qnormalize(qmul(fix[:, None], rotations))
        frozen_pos = qrotate(fix[:, None], positions - positions[:, p:p + 1])
        return names, frozen_rot, frozen_pos

    @classmethod
    def build_reference(cls, bvh_texts: list[str]) -> dict[str, list[float]]:
        """複数の生成結果から、SentiAvatar の平均的な話し姿勢を求める(差分で移す関節の基準)。"""
        frames = []
        names = None
        for text in bvh_texts:
            names, frozen, _ = cls.pelvis_frozen(parse_bvh(text))
            frames.append(frozen)
        stacked = np.concatenate(frames)
        return {source: qmean(stacked[:, names.index(source)], axis=0).tolist() for source in REFERENCE_JOINTS}

    def retarget(self, bvh_text: str, *, amplitude: float = 1.0) -> Motion:
        source = resample(parse_bvh(bvh_text), self.fps)
        names, frozen_rot, frozen_pos = self.pelvis_frozen(source)
        T = source.frames
        skeleton = self.skeleton
        local = np.repeat(self.neutral[None], T, axis=0)

        def reference_of(name: str, column: np.ndarray) -> np.ndarray:
            if self.reference and name in self.reference:
                return np.asarray(self.reference[name])
            return qmean(column, axis=0)

        def add_world_delta(joint_names, delta) -> None:
            """ワールドの回転 delta を、関節の並びに等分して足す(親から順に)。"""
            step = qpow(delta, 1.0 / len(joint_names))
            for name in joint_names:
                j = skeleton.index(name)
                _, globals_ = forward_kinematics(Motion(skeleton, np.zeros((T, 3)), local, self.fps))
                parent = globals_[:, skeleton.joints[j].parent]
                local[:, j] = qmul(qinv(parent), qmul(step, globals_[:, j]))

        # 胸: SuSu の胸の向きの、平均的な話し姿勢からのずれ(弱めて合計に上限)
        chest = frozen_rot[:, names.index("spine_05")]
        chest_delta = qmul(chest, qinv(reference_of("spine_05", chest))[None])
        add_world_delta(SPINE_CHAIN, limit_rotation(qpow(chest_delta, CHEST_SCALE * amplitude), CHEST_MAX))
        # 頭: 胸に対する頭のずれ(上向きは特に抑える)
        head = frozen_rot[:, names.index("head")]
        head_delta = qmul(head, qinv(reference_of("head", head))[None])
        head_relative = qmul(qinv(chest_delta), head_delta)
        add_world_delta(NECK_CHAIN, limit_rotation(qpow(head_relative, HEAD_SCALE * amplitude), HEAD_MAX, HEAD_MAX_UP))
        # 鎖骨
        for source_name, (target_name, scale, max_degrees) in DELTA_JOINTS.items():
            if source_name not in names:
                continue
            column = frozen_rot[:, names.index(source_name)]
            delta = _limit(qmul(column, qinv(reference_of(source_name, column))[None]), scale * amplitude, max_degrees)
            add_world_delta((target_name,), delta)

        # 腕: 骨の向きを SuSu に合わせる(親から順に。ねじれは立ち姿のまま)
        for joint, child, source_joint, source_child, weight in DIRECTION_BONES:
            if source_joint not in names or source_child not in names:
                continue
            target = frozen_pos[:, names.index(source_child)] - frozen_pos[:, names.index(source_joint)]
            target /= np.linalg.norm(target, axis=-1, keepdims=True).clip(1e-9)
            j = skeleton.index(joint)
            _, globals_ = forward_kinematics(Motion(skeleton, np.zeros((T, 3)), local, self.fps))
            bone = skeleton.joints[skeleton.index(child)].offset
            current = qrotate(globals_[:, j], bone / np.linalg.norm(bone))
            swing = _arc(current, target)
            swing = qpow(swing, weight * amplitude) if abs(weight * amplitude - 1.0) > 1e-6 else swing
            new_global = qmul(swing, globals_[:, j])
            parent = globals_[:, skeleton.joints[j].parent]
            local[:, j] = qmul(qinv(parent), new_global)

        root = np.tile([[0.0, self.root_height, 0.0]], (T, 1))
        return Motion(skeleton, root, qnormalize(local), self.fps)

