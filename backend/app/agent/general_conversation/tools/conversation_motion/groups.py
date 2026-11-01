"""標準骨格(T2M / モーキャプ共通の 55 関節)の部位分け。"""

from __future__ import annotations

from .bvh import Skeleton

# 部位 → 主要関節(動いているかの判定に使う)。指は腕の子孫として一緒に扱う。
GROUPS: dict[str, tuple[str, ...]] = {
    "right_arm": ("RightShoulder", "RightArm", "RightForeArm", "RightHand"),
    "left_arm": ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"),
    "head": ("Neck", "Neck1", "Head"),
    "spine": ("Spine", "Spine1", "Spine2", "Spine3", "Spine4"),
    "hips": ("Hips",),
}

LEGS = ("LeftUpLeg", "RightUpLeg")


def group_joint_indices(skeleton: Skeleton) -> dict[str, tuple[list[int], list[int]]]:
    """部位ごとに (主要関節, 部位に属する全関節) の番号を返す。"""
    result: dict[str, tuple[list[int], list[int]]] = {}
    for group, names in GROUPS.items():
        main = [skeleton.index(name) for name in names]
        if group in ("right_arm", "left_arm"):
            members = skeleton.descendants(names[0])
        elif group == "hips":
            members = [skeleton.index("Hips")]
            for leg in LEGS:
                members += skeleton.descendants(leg)
        else:
            members = main
        result[group] = (main, members)
    return result
