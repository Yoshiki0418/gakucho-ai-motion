"""Retarget generated BVH to the local reference transforms of a MetaHuman Body.

Pure Python for the editor's embedded interpreter. Quaternions use UE's (x,y,z,w)
order. Only the hips translate. Other bones keep their original length and scale.
"""

import io
import math
import re
from array import array
from typing import NamedTuple

IDENTITY = (0.0, 0.0, 0.0, 1.0)
ZERO = (0.0, 0.0, 0.0)
AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s):
    return tuple(x * s for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(a):
    norm = math.sqrt(dot(a, a))
    if norm < 1e-8:
        raise ValueError("Degenerate reference bone or facing direction")
    return scale(a, 1 / norm)


def cross(a, b):
    x, y, z = a
    i, j, k = b
    return (y * k - z * j, z * i - x * k, x * j - y * i)


def mul(a, b):
    x, y, z, w = a
    i, j, k, r = b
    return (
        w * i + x * r + y * k - z * j,
        w * j - x * k + y * r + z * i,
        w * k + x * j - y * i + z * r,
        w * r - x * i - y * j - z * k,
    )


def inverse(q):
    return (-q[0], -q[1], -q[2], q[3])


def rotate(q, v):
    return mul(mul(q, (*v, 0.0)), inverse(q))[:3]


def swing(a, b):
    a, b = unit(a), unit(b)
    w = 1 + dot(a, b)
    if w < 1e-8:
        perpendicular = min(AXES, key=lambda axis: abs(dot(a, axis)))
        return (*unit(cross(a, perpendicular)), 0.0)
    return unit((*cross(a, b), w))


class Joint(NamedTuple):
    name: str
    parent: int
    offset: tuple
    rotations: tuple  # (column, axis index) in intrinsic BVH channel order


class Clip(NamedTuple):
    joints: list
    rows: list
    frame_time: float
    root_columns: tuple

    @property
    def frame_count(self):
        return len(self.rows)

    def world_rotations(self, frame):
        row = self.rows[frame]
        world = []
        for joint in self.joints:
            q = world[joint.parent] if joint.parent >= 0 else IDENTITY
            for column, axis in joint.rotations:
                angle = math.radians(row[column]) * 0.5
                v = [0.0, 0.0, 0.0]
                v[axis] = math.sin(angle)
                q = mul(q, (*v, math.cos(angle)))
            world.append(q)
        return world

    def root_position(self, frame):
        return tuple(
            self.rows[frame][i] if i is not None else 0.0 for i in self.root_columns
        )


def parse_bvh(text):
    """Read one-root BVH with root translation and intrinsic rotation channels."""
    if not isinstance(text, str) or len(text) > 16 * 1024 * 1024:
        raise ValueError("Invalid BVH size")
    sections = re.split(r"(?m)^[ \t]*MOTION[ \t\r]*$", text, maxsplit=1)
    if len(sections) != 2 or len(sections[0]) > 256 * 1024:
        raise ValueError("Invalid BVH hierarchy")
    tokens = iter(re.findall(r"[^\s{}]+|[{}]", sections[0]))
    joints, root_columns = [], [None] * 3
    width = 0

    def take(expected):
        if next(tokens) != expected:
            raise ValueError(f"Expected BVH {expected}")

    def read_joint(parent):
        nonlocal width
        if len(joints) >= 256:
            raise ValueError("Too many BVH joints")
        name = next(tokens)
        take("{")
        take("OFFSET")
        offset = tuple(float(next(tokens)) for _ in range(3))
        if not all(math.isfinite(v) for v in offset):
            raise ValueError("Non-finite BVH offset")
        take("CHANNELS")
        count = int(next(tokens))
        if not 0 <= count <= 6:
            raise ValueError("Invalid channel count")
        rotation, seen = [], set()
        for i in range(width, width + count):
            channel = next(tokens).lower()
            if not re.fullmatch("[xyz](position|rotation)", channel) or channel in seen:
                raise ValueError("Invalid BVH channel")
            seen.add(channel)
            axis = "xyz".index(channel[0])
            if channel.endswith("rotation"):
                rotation.append((i, axis))
            elif parent == -1:
                root_columns[axis] = i
            else:
                raise ValueError("Only the BVH root can translate")
        width += count
        index = len(joints)
        joints.append(Joint(name, parent, offset, rotation))
        token = next(tokens)
        while token != "}":
            if token == "JOINT":
                read_joint(index)
            elif token == "End":
                take("Site")
                take("{")
                take("OFFSET")
                for _ in range(3):
                    float(next(tokens))
                take("}")
            else:
                raise ValueError("Invalid BVH joint")
            token = next(tokens)

    try:
        take("HIERARCHY")
        take("ROOT")
        read_joint(-1)
        if next(tokens, None) is not None:
            raise ValueError("Unexpected BVH header tokens")
    except (StopIteration, RuntimeError) as error:
        raise ValueError("Incomplete BVH hierarchy") from error

    lines = (line.strip() for line in io.StringIO(sections[1]) if line.strip())
    frames = re.fullmatch(r"Frames:\s*(\d+)", next(lines, ""))
    period = re.fullmatch(r"Frame\s+Time:\s*(\S+)", next(lines, ""))
    if not frames or not period:
        raise ValueError("Missing BVH timing")
    count, dt = int(frames[1]), float(period[1])
    if (
        not 0 < count <= 36000
        or not 0 < dt <= 1
        or width == 0
        or count * width > 4_000_000
    ):
        raise ValueError("Invalid BVH frame count, rate or size")
    rows = []
    for line in lines:
        values = line.split()
        if len(values) != width or len(rows) >= count:
            raise ValueError("BVH frame dimensions do not match hierarchy")
        row = array("f", map(float, values))
        if not all(math.isfinite(x) for x in row):
            raise ValueError("Non-finite BVH pose")
        rows.append(row)
    if len(rows) != count:
        raise ValueError("Incomplete BVH frames")
    return Clip(joints, rows, dt, tuple(root_columns))


# Canonical names are the UE body skeleton's names. Unmapped correction, face and
# twist joints keep the transforms supplied by UE's existing animation setup.
NAMES = {
    "hips": "pelvis",
    "head": "head",
    "neck": "neck_01",
    "neck1": "neck_02",
    "spine": "spine_01",
    **{f"spine{i}": f"spine_0{i+1}" for i in range(1, 5)},
}
for side, suffix in (("left", "l"), ("right", "r")):
    for src, dst in (
        ("shoulder", "clavicle"),
        ("arm", "upperarm"),
        ("forearm", "lowerarm"),
        ("hand", "hand"),
        ("upleg", "thigh"),
        ("leg", "calf"),
        ("foot", "foot"),
        ("toebase", "ball"),
    ):
        NAMES[side + src] = f"{dst}_{suffix}"
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        for n in (1, 2, 3):
            NAMES[f"{side}hand{finger}{n}"] = f"{finger}_0{n}_{suffix}"
NAMES.update({name: name for name in list(NAMES.values())})


def body_axes(positions, is_ue):
    # Pelvis/head identify the vertical capture axis. Derive lateral direction
    # from the hips to support UE meshes facing either +X or +Y.
    height = sub(positions["head"], positions["pelvis"])
    vertical = max(range(3), key=lambda i: abs(height[i]))
    up = scale(AXES[vertical], math.copysign(1, height[vertical]))
    lateral = sub(positions["thigh_l"], positions["thigh_r"])
    left = unit(sub(lateral, scale(up, dot(lateral, up))))
    front = cross(up, left) if is_ue else cross(left, up)
    return left, up, front


class Retarget:
    def __init__(self, clip, reference):
        self.clip = clip
        source = {}
        for i, joint in enumerate(clip.joints):
            name = re.split(r"[:|]", joint.name)[-1].lower().removeprefix("mixamorig")
            canonical = NAMES.get(name)
            if canonical in reference:
                if canonical in source:
                    raise ValueError(f"Duplicate BVH joint for {canonical}")
                source[canonical] = i
        required = {"pelvis", "head", "thigh_l", "thigh_r", "calf_l", "foot_l"}
        if not required.issubset(source):
            raise ValueError(
                "BVH and target must share a humanoid pelvis, head and legs"
            )

        # SuSu exports UE-named bones with joint pre-rotations baked into every
        # frame. The first articulated frame supplies its calibration axes.
        embedded_bind = clip.joints[source["pelvis"]].name.lower() == "pelvis"
        src_bind = (
            clip.world_rotations(0) if embedded_bind else [IDENTITY] * len(clip.joints)
        )
        src_xyz = []
        for joint in clip.joints:
            p = joint.parent
            src_xyz.append(
                add(src_xyz[p], rotate(src_bind[p], joint.offset)) if p >= 0 else ZERO
            )
        source_positions = {key: src_xyz[i] for key, i in source.items()}
        target_xyz, bind_world = {}, {}
        for name, (parent, t, q) in reference.items():
            parent_q = bind_world[parent] if parent else IDENTITY
            bind_world[name] = mul(parent_q, q)
            target_xyz[name] = (
                add(target_xyz[parent], rotate(parent_q, t)) if parent else t
            )
        self.src_axes = body_axes(source_positions, False)
        self.dst_axes = body_axes(target_xyz, True)

        def leg_length(positions):
            return math.dist(positions["thigh_l"], positions["calf_l"]) + math.dist(
                positions["calf_l"], positions["foot_l"]
            )

        self.scale = leg_length(target_xyz) / max(leg_length(source_positions), 1e-8)
        self.root_origin = clip.root_position(0)
        parent, self.pelvis_rest, _ = reference["pelvis"]
        self.pelvis_parent_inv = inverse(bind_world[parent]) if parent else IDENTITY

        # Nearest matched children decide the swing needed to align an A-pose
        # to the BVH rest directions. Branch joints keep their reference twist.
        children = {key: [] for key in source}
        for child in source:
            parent = reference[child][0]
            while parent and parent not in source:
                parent = reference[parent][0]
            if parent:
                children[parent].append(child)
        self.links = []
        aligned = {}
        for name, (parent, _, local) in reference.items():
            parent_q = aligned[parent] if parent else IDENTITY
            current = mul(parent_q, local)
            if name in source:
                targets = children[name]
                if len(targets) == 1:
                    child = targets[0]
                    offset = sub(target_xyz[child], target_xyz[name])
                    actual = rotate(mul(current, inverse(bind_world[name])), offset)
                    desired = self.vector(
                        sub(source_positions[child], source_positions[name])
                    )
                    if dot(actual, actual) > 1e-8 and dot(desired, desired) > 1e-8:
                        current = mul(swing(actual, desired), current)
                ancestor = parent
                while ancestor and ancestor not in source:
                    ancestor = reference[ancestor][0]
                i = source[name]
                self.links.append(
                    (name, i, source.get(ancestor), inverse(parent_q), current)
                )
            aligned[name] = current
        self.source_bind_inv = [inverse(q) for q in src_bind]

    def vector(self, v):
        components = tuple(dot(axis, v) for axis in self.src_axes)
        return tuple(
            sum(components[i] * self.dst_axes[i][j] for i in range(3)) for j in range(3)
        )

    def pose(self, frame_index):
        world = self.clip.world_rotations(frame_index)
        deltas, rotations = {}, {}
        for name, i, ancestor, parent_inv, rest in self.links:
            q = mul(world[i], self.source_bind_inv[i])
            # An axial vector changes sign under handedness reversal.
            delta = (*scale(self.vector(q[:3]), -1), q[3])
            parent_delta = deltas[ancestor] if ancestor is not None else IDENTITY
            rotations[name] = mul(
                mul(parent_inv, inverse(parent_delta)), mul(delta, rest)
            )
            deltas[i] = delta
        movement = self.vector(
            sub(self.clip.root_position(frame_index), self.root_origin)
        )
        pelvis = add(
            self.pelvis_rest,
            rotate(self.pelvis_parent_inv, scale(movement, self.scale)),
        )
        return rotations, pelvis


def prepare_retarget(clip, reference):
    """reference maps bone -> (parent or None, local xyz, local xyzw), parents first."""
    return Retarget(clip, reference)
