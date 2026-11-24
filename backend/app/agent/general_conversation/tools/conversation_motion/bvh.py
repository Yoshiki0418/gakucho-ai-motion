"""BVH の読み書きと四元数ユーティリティ(会話モーションの層合成用)。

四元数はすべて scipy と同じ (x, y, z, w) 順で持つ。
BVH の回転チャンネルは「列挙順の内因性回転」なので、
`Zrotation Xrotation Yrotation` は scipy の `from_euler("ZXY", ...)` に一致する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------- 骨格と動作


@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str]
    end_site: np.ndarray | None = None


@dataclass
class Skeleton:
    joints: list[Joint]
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {joint.name: i for i, joint in enumerate(self.joints)}

    @property
    def names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def parents(self) -> np.ndarray:
        return np.asarray([joint.parent for joint in self.joints])

    @property
    def offsets(self) -> np.ndarray:
        return np.stack([joint.offset for joint in self.joints])

    def index(self, name: str) -> int:
        return self._index[name]

    def get(self, name: str) -> int | None:
        return self._index.get(name)

    def descendants(self, name: str) -> list[int]:
        """name 自身とその子孫の関節番号。"""
        root = self.index(name)
        found = [root]
        for i, joint in enumerate(self.joints):
            if joint.parent in found:
                found.append(i)
        return found


@dataclass
class Motion:
    skeleton: Skeleton
    root: np.ndarray
    """(T, 3) ルート位置"""
    quats: np.ndarray
    """(T, J, 4) 各関節のローカル回転 (x, y, z, w)"""
    fps: float

    @property
    def frames(self) -> int:
        return len(self.root)

    @property
    def duration(self) -> float:
        return max(self.frames - 1, 0) / self.fps

    def copy(self) -> "Motion":
        return Motion(self.skeleton, self.root.copy(), self.quats.copy(), self.fps)

    def slice(self, start: int, stop: int) -> "Motion":
        return Motion(
            self.skeleton,
            self.root[start:stop].copy(),
            self.quats[start:stop].copy(),
            self.fps,
        )


# ---------------------------------------------------------------- 四元数


def qnormalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-12)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """四元数 q でベクトル v を回す(ブロードキャスト可)。"""
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qangle(q: np.ndarray) -> np.ndarray:
    """回転角(ラジアン)。"""
    return 2.0 * np.arccos(np.clip(np.abs(q[..., 3]), 0.0, 1.0))


def qslerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """球面線形補間。t は a, b の先頭次元にブロードキャストできる形。"""
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    small = sin < 1e-6
    safe = np.where(small, 1.0, sin)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    return qnormalize(wa * a + wb * b)


def qpow(q: np.ndarray, s) -> np.ndarray:
    """回転角を s 倍する(軸は保つ)。"""
    shape = q.shape
    rotvec = Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(shape[:-1] + (3,))
    rotvec = rotvec * np.asarray(s, dtype=np.float64)[..., None]
    return Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_quat().reshape(shape)


def limit_rotation(
    q: np.ndarray, max_degrees: float, max_up_degrees: float | None = None
) -> np.ndarray:
    """回転の大きさを max_degrees までに抑える。max_up_degrees を与えると、
    左右軸(x)まわりの「顔を上げる向き」(負の角度)をその角度までに抑える。"""
    q = qnormalize(np.array(q, dtype=np.float64))
    angle = np.degrees(qangle(q))
    scale = np.minimum(1.0, max_degrees / np.maximum(angle, 1e-9))
    if max_up_degrees is not None:
        pitch = np.degrees(
            2.0
            * np.arctan2(q[..., 0], np.abs(q[..., 3]) + 1e-12)
            * np.sign(q[..., 3] + 1e-12)
        )
        up = np.maximum(-pitch, 0.0)
        scale = np.minimum(
            scale,
            np.where(up > max_up_degrees, max_up_degrees / np.maximum(up, 1e-9), 1.0),
        )
    return qpow(q, scale) if np.any(scale < 1.0) else q


def qcontinuous(q: np.ndarray) -> np.ndarray:
    """時系列(先頭次元)で符号を揃え、補間で遠回りしないようにする。"""
    q = q.copy()
    for t in range(1, len(q)):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0.0
        q[t][flip] *= -1.0
    return q


def qmean(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """符号を揃えた単純平均(近い回転同士の平均に十分)。"""
    q = np.moveaxis(q, axis, 0)
    ref = q[0]
    aligned = np.where((np.sum(q * ref, axis=-1, keepdims=True) < 0.0), -q, q)
    return qnormalize(aligned.mean(axis=0))


# ---------------------------------------------------------------- BVH 読み込み


_TOKEN = re.compile(r"\S+")


def _rotation_sequence(channels: list[str]) -> str:
    return "".join(c[0].upper() for c in channels if c.lower().endswith("rotation"))


def parse_bvh(text: str) -> Motion:
    header, _, motion_text = text.replace("\r\n", "\n").partition("MOTION")
    if not motion_text:
        raise ValueError("BVH に MOTION セクションがありません")
    tokens = _TOKEN.findall(header)
    joints: list[Joint] = []
    stack: list[int] = []
    i = 0
    pending: tuple[str, int] | None = None
    in_end_site = False
    while i < len(tokens):
        token = tokens[i]
        if token in ("ROOT", "JOINT"):
            pending = (tokens[i + 1], stack[-1] if stack else -1)
            i += 2
        elif token == "End":
            in_end_site = True
            i += 2  # "End Site"
        elif token == "{":
            if in_end_site:
                stack.append(-2)
            else:
                assert pending is not None
                joints.append(Joint(pending[0], pending[1], np.zeros(3), []))
                stack.append(len(joints) - 1)
                pending = None
            i += 1
        elif token == "}":
            top = stack.pop()
            if top == -2:
                in_end_site = False
            i += 1
        elif token == "OFFSET":
            values = np.asarray([float(v) for v in tokens[i + 1 : i + 4]])
            if in_end_site:
                owner = next(s for s in reversed(stack) if s >= 0)
                joints[owner].end_site = values
            else:
                joints[stack[-1]].offset = values
            i += 4
        elif token == "CHANNELS":
            count = int(tokens[i + 1])
            joints[stack[-1]].channels = tokens[i + 2 : i + 2 + count]
            i += 2 + count
        else:
            i += 1
    skeleton = Skeleton(joints)

    lines = [line.strip() for line in motion_text.strip().split("\n") if line.strip()]
    frames = int(lines[0].split(":")[1])
    frame_time = float(lines[1].split(":")[1])
    data = np.asarray(
        [[float(v) for v in line.split()] for line in lines[2 : 2 + frames]],
        dtype=np.float64,
    )
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("BVH のフレームが空です")

    root = np.zeros((len(data), 3))
    quats = np.zeros((len(data), len(joints), 4))
    column = 0
    for j, joint in enumerate(joints):
        rotation_columns = []
        for channel in joint.channels:
            lower = channel.lower()
            if lower.endswith("position") and j == 0:
                root[:, "xyz".index(lower[0])] = data[:, column]
            elif lower.endswith("rotation"):
                rotation_columns.append(column)
            column += 1
        if rotation_columns:
            sequence = _rotation_sequence(joint.channels)
            quats[:, j] = Rotation.from_euler(
                sequence, data[:, rotation_columns], degrees=True
            ).as_quat()
        else:
            quats[:, j] = [0.0, 0.0, 0.0, 1.0]
    if not any(c.lower().endswith("position") for c in joints[0].channels):
        root[:] = joints[0].offset
    return Motion(skeleton, root, qcontinuous(quats), 1.0 / frame_time)


# ---------------------------------------------------------------- BVH 書き出し


def write_bvh(
    motion: Motion, *, rotation_digits: int = 3, position_digits: int = 5
) -> str:
    """ルート 6ch・他 3ch (ZXY) の素直な BVH を書く。"""
    skeleton = motion.skeleton
    children: dict[int, list[int]] = {i: [] for i in range(len(skeleton.joints))}
    for i, joint in enumerate(skeleton.joints):
        if joint.parent >= 0:
            children[joint.parent].append(i)

    lines = ["HIERARCHY"]

    def emit(j: int, depth: int) -> None:
        joint = skeleton.joints[j]
        pad = "  " * depth
        lines.append(f"{pad}{'ROOT' if joint.parent < 0 else 'JOINT'} {joint.name}")
        lines.append(f"{pad}{{")
        lines.append(
            f"{pad}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}"
        )
        if joint.parent < 0:
            lines.append(
                f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
            )
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for child in children[j]:
            emit(child, depth + 1)
        if not children[j]:
            end = joint.end_site if joint.end_site is not None else np.zeros(3)
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET {end[0]:.6f} {end[1]:.6f} {end[2]:.6f}")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")

    emit(0, 0)
    T, J = motion.quats.shape[:2]
    euler = (
        Rotation.from_quat(motion.quats.reshape(-1, 4))
        .as_euler("ZXY", degrees=True)
        .reshape(T, J * 3)
    )
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / motion.fps:.6f}"]
    rotation_format = f"{{:.{rotation_digits}f}}"
    position_format = f"{{:.{position_digits}f}}"
    for t in range(T):
        values = [position_format.format(v) for v in motion.root[t]]
        values += [rotation_format.format(v) for v in euler[t]]
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 時間方向


def resample(motion: Motion, fps: float) -> Motion:
    """フレームレートを変える(回転は slerp、位置は線形)。"""
    if abs(motion.fps - fps) < 1e-9:
        return motion.copy()
    duration = motion.duration
    count = max(int(round(duration * fps)) + 1, 1)
    times = np.arange(count) / fps * motion.fps
    i0 = np.clip(np.floor(times).astype(int), 0, motion.frames - 1)
    i1 = np.clip(i0 + 1, 0, motion.frames - 1)
    alpha = (times - i0).clip(0.0, 1.0)
    root = motion.root[i0] * (1 - alpha[:, None]) + motion.root[i1] * alpha[:, None]
    quats = qslerp(motion.quats[i0], motion.quats[i1], alpha[:, None])
    return Motion(motion.skeleton, root, qcontinuous(quats), fps)


# ---------------------------------------------------------------- 順運動学


def forward_kinematics(motion: Motion) -> tuple[np.ndarray, np.ndarray]:
    """(T, J, 3) の関節位置と (T, J, 4) のワールド回転。"""
    skeleton = motion.skeleton
    T, J = motion.quats.shape[:2]
    rotations = np.zeros((T, J, 4))
    positions = np.zeros((T, J, 3))
    for j, joint in enumerate(skeleton.joints):
        if joint.parent < 0:
            rotations[:, j] = motion.quats[:, j]
            positions[:, j] = motion.root
        else:
            parent_rotation = rotations[:, joint.parent]
            rotations[:, j] = qmul(parent_rotation, motion.quats[:, j])
            positions[:, j] = positions[:, joint.parent] + qrotate(
                parent_rotation, joint.offset
            )
    return positions, rotations


def local_from_global(skeleton: Skeleton, rotations: np.ndarray) -> np.ndarray:
    """ワールド回転からローカル回転へ戻す。"""
    local = np.zeros_like(rotations)
    for j, joint in enumerate(skeleton.joints):
        if joint.parent < 0:
            local[:, j] = rotations[:, j]
        else:
            local[:, j] = qmul(qinv(rotations[:, joint.parent]), rotations[:, j])
    return qnormalize(local)
