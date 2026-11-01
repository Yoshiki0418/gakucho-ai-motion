"""Mirror generated single-hand motions while retaining their timing and poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata


RIGHT_WAVE_CAPTION = (
    '右手を肩の高さに上げ、手のひらを前方に向けて軽く左右に振って挨拶する'
)
RIGHT_RAISE_CAPTION = '右手を上げる'


def simple_left_generation_caption(prompt: str) -> str | None:
    """Recognize only an uncomplicated left-hand wave or hand/arm raise."""
    text = unicodedata.normalize('NFKC', prompt)
    text = re.sub(r'[\s、。,.!?！？]', '', text)
    prefix = r'(?:立っている状態(?:から|で))?'
    if re.fullmatch(prefix + r'左手を(?:振る|振って(?:ください|下さい)?)', text):
        return RIGHT_WAVE_CAPTION
    if re.fullmatch(
        prefix + r'左(?:手|腕)を(?:上げる|あげる|挙げる|上げて(?:ください|下さい)?)',
        text,
    ):
        return RIGHT_RAISE_CAPTION
    return None


@dataclass(frozen=True)
class HandRaiseConstraint:
    requested_side: str | None
    overhead: bool = False


def simple_hand_raise_constraint(prompt: str) -> HandRaiseConstraint | None:
    """Keep side/height constraints for a single, uncomplicated hand raise."""
    text = unicodedata.normalize('NFKC', prompt)
    text = re.sub(r'[\s、。,.!?！？]', '', text)
    match = re.fullmatch(
        r'(?:立っている状態(?:から|で))?'
        r'(?P<side>左|右)?(?:手|腕)を'
        r'(?P<height>(?:頭上に|頭の上に)(?:高く)?|高く)?'
        r'(?:上げる|あげる|挙げる|持ち上げる|(?:上げて|持ち上げて)(?:ください|下さい)?)',
        text,
    )
    if match is None:
        return None
    side = {'左': 'left', '右': 'right'}.get(match.group('side'))
    return HandRaiseConstraint(side, overhead=match.group('height') is not None)


@dataclass
class _Joint:
    path: tuple[str, ...]
    offset: tuple[float, float, float] | None = None
    channels: tuple[str, ...] = ()
    channel_start: int = 0


def _opposite_name(name: str) -> str:
    if name.startswith('Left'):
        return 'Right' + name[4:]
    if name.startswith('Right'):
        return 'Left' + name[5:]
    return name


def _parse_hierarchy(hierarchy: str) -> tuple[list[_Joint], int]:
    tokens = re.findall(r'[{}]|[^\s{}]+', hierarchy)
    cursor = 0
    channel_count = 0
    joints: list[_Joint] = []

    def take(expected: str | None = None) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError('Unsupported BVH rig: incomplete hierarchy')
        token = tokens[cursor]
        cursor += 1
        if expected is not None and token != expected:
            raise ValueError(f'Unsupported BVH rig: expected {expected!r}')
        return token

    def parse_joint(parent: tuple[str, ...], kind: str) -> None:
        nonlocal channel_count
        if kind == 'End':
            take('Site')
            name = 'EndSite'
        else:
            name = take()
        joint = _Joint(parent + (name,), channel_start=channel_count)
        joints.append(joint)
        take('{')
        while True:
            item = take()
            if item == '}':
                break
            if item == 'OFFSET':
                if joint.offset is not None:
                    raise ValueError('Unsupported BVH rig: duplicate offset')
                try:
                    offset = tuple(float(take()) for _ in range(3))
                except ValueError as error:
                    raise ValueError('Unsupported BVH rig: invalid offset') from error
                if not all(math.isfinite(value) for value in offset):
                    raise ValueError('Unsupported BVH rig: non-finite offset')
                joint.offset = offset
            elif item == 'CHANNELS':
                if joint.channels or kind == 'End':
                    raise ValueError('Unsupported BVH rig: invalid channels')
                try:
                    count = int(take())
                except ValueError as error:
                    raise ValueError('Unsupported BVH rig: invalid channel count') from error
                if not 0 <= count <= 6:
                    raise ValueError('Unsupported BVH rig: invalid channel count')
                joint.channels = tuple(take() for _ in range(count))
                supported = {
                    'Xposition', 'Yposition', 'Zposition',
                    'Xrotation', 'Yrotation', 'Zrotation',
                }
                if len(set(joint.channels)) != count or set(joint.channels) - supported:
                    raise ValueError('Unsupported BVH rig: unsupported channels')
                joint.channel_start = channel_count
                channel_count += count
            elif item in ('JOINT', 'End'):
                parse_joint(joint.path, item)
            else:
                raise ValueError(f'Unsupported BVH rig: unexpected {item!r}')
        if joint.offset is None:
            raise ValueError('Unsupported BVH rig: missing offset')

    take('HIERARCHY')
    take('ROOT')
    parse_joint((), 'ROOT')
    if cursor != len(tokens) or channel_count == 0:
        raise ValueError('Unsupported BVH rig: invalid hierarchy')
    return joints, channel_count


def _parse_bvh(bvh: str):
    """Reflect X and swap Left/Right joint motion on a symmetric BVH rig.

    For each axis-rotation channel, M R M keeps X rotations and negates Y/Z
    rotations, where M = diag(-1, 1, 1). Root X translation also changes sign.
    The symmetric skeleton, frame count and frame time remain unchanged.
    """
    motion_marker = re.search(r'(?m)^[ \t]*MOTION[ \t]*\r?$', bvh)
    if motion_marker is None:
        raise ValueError('Unsupported BVH rig: missing MOTION section')
    joints, channel_count = _parse_hierarchy(bvh[:motion_marker.start()])
    lookup = {joint.path: joint for joint in joints}
    if len(lookup) != len(joints):
        raise ValueError('Unsupported BVH rig: duplicate joint paths')
    counterparts: list[_Joint] = []
    for joint in joints:
        opposite_path = tuple(_opposite_name(name) for name in joint.path)
        opposite = lookup.get(opposite_path)
        if opposite is None:
            raise ValueError('Unsupported BVH rig: missing Left/Right joint pair')
        if opposite.channels != joint.channels:
            raise ValueError('Unsupported BVH rig: Left/Right channels differ')
        reflected_offset = (-opposite.offset[0], opposite.offset[1], opposite.offset[2])
        if any(abs(a - b) > 1e-6 for a, b in zip(joint.offset, reflected_offset)):
            raise ValueError('Unsupported BVH rig: offsets are not mirror-symmetric')
        counterparts.append(opposite)
    if not any(joint.path[-1].startswith('Left') for joint in joints):
        raise ValueError('Unsupported BVH rig: no Left/Right joints')

    tail = bvh[motion_marker.end():]
    metadata = re.match(
        r'\s*Frames:[ \t]*(\d+)[ \t]*\r?\n[ \t]*Frame Time:[ \t]*'
        r'([^\s]+)[ \t]*\r?\n',
        tail,
    )
    if metadata is None:
        raise ValueError('Unsupported BVH motion: missing frame metadata')
    frame_count = int(metadata.group(1))
    try:
        frame_time = float(metadata.group(2))
    except ValueError as error:
        raise ValueError('Unsupported BVH motion: invalid frame time') from error
    if frame_count < 1 or not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError('Unsupported BVH motion: invalid frame metadata')
    rows = [row.split() for row in tail[metadata.end():].splitlines() if row.strip()]
    if len(rows) != frame_count:
        raise ValueError('Unsupported BVH motion: frame count differs from data')
    for row in rows:
        if len(row) != channel_count:
            raise ValueError('Unsupported BVH motion: channel count differs from rig')
        try:
            finite = all(math.isfinite(float(value)) for value in row)
        except ValueError as error:
            raise ValueError('Unsupported BVH motion: invalid channel value') from error
        if not finite:
            raise ValueError('Unsupported BVH motion: non-finite channel value')
    prefix = bvh[:motion_marker.end() + metadata.end()]
    newline = '\r\n' if '\r\n' in prefix else '\n'
    return joints, counterparts, rows, prefix, newline, channel_count


def _mirror_parsed(parsed) -> str:
    joints, counterparts, rows, prefix, newline, channel_count = parsed
    mirrored_rows = []
    for row in rows:
        mirrored = [''] * channel_count
        for joint, opposite in zip(joints, counterparts):
            for offset, channel in enumerate(joint.channels):
                value = row[opposite.channel_start + offset]
                if channel in ('Xposition', 'Yrotation', 'Zrotation'):
                    # Change only the sign so all original numeric precision survives.
                    value = value[1:] if value.startswith('-') else '-' + value.lstrip('+')
                mirrored[joint.channel_start + offset] = value
        mirrored_rows.append(' '.join(mirrored))
    return prefix + newline.join(mirrored_rows) + newline


def mirror_bvh_sagittal(bvh: str) -> str:
    """Reflect generated motion across X=0 and swap paired Left/Right joints."""
    return _mirror_parsed(_parse_bvh(bvh))


def _multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _rotate(matrix, point):
    return tuple(sum(matrix[i][j] * point[j] for j in range(3)) for i in range(3))


def _axis_rotation(axis, degrees):
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    if axis == 'X':
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis == 'Y':
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _raised_side(parsed, overhead: bool) -> str:
    joints, _, rows, _, _, _ = parsed
    targets = {}
    for name in ('Hips', 'Head', 'LeftHand', 'RightHand'):
        matches = [joint.path for joint in joints if joint.path[-1] == name]
        if len(matches) != 1:
            raise ValueError(f'Unsupported BVH hand raise: expected one {name}')
        targets[name] = matches[0]
    # Only the hand/head ancestors need forward kinematics; fingers and legs
    # do not affect their heights.
    needed = {path[:length] for path in targets.values()
              for length in range(1, len(path) + 1)}
    selected = [joint for joint in joints if joint.path in needed]
    head_path = targets['Head']
    torso_height = sum(joint.offset[1] for joint in selected
                       if joint.path != targets['Hips'] and head_path[:len(joint.path)] == joint.path)
    if torso_height <= 0:
        raise ValueError('Unsupported BVH hand raise: invalid torso scale')
    # The dedicated T2M canonical torso is ~0.64 m. Scaling keeps the same
    # qualification for other symmetric rigs expressed in centimetres.
    scale = torso_height / 0.64
    active_min = (0.10 if overhead else -0.08) * scale
    inactive_max = -0.20 * scale
    separation_min = 0.20 * scale
    qualified = set()
    hand_peaks = {'left': -math.inf, 'right': -math.inf}
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for row in rows:
        positions = {}
        rotations = {}
        for joint in selected:
            local_rotation = identity
            translation = list(joint.offset)
            for channel_index, channel in enumerate(joint.channels):
                value = float(row[joint.channel_start + channel_index])
                if channel.endswith('rotation'):
                    local_rotation = _multiply(local_rotation, _axis_rotation(channel[0], value))
                else:
                    translation['XYZ'.index(channel[0])] += value
            parent = joint.path[:-1]
            if parent:
                shifted = _rotate(rotations[parent], translation)
                positions[joint.path] = tuple(positions[parent][axis] + shifted[axis] for axis in range(3))
                rotations[joint.path] = _multiply(rotations[parent], local_rotation)
            else:
                positions[joint.path] = tuple(translation)
                rotations[joint.path] = local_rotation
        head_y = positions[targets['Head']][1]
        hand_y = {side: positions[targets[name]][1] - head_y
                  for side, name in (('left', 'LeftHand'), ('right', 'RightHand'))}
        for side in hand_peaks:
            hand_peaks[side] = max(hand_peaks[side], hand_y[side])
        for side, other in (('left', 'right'), ('right', 'left')):
            if (hand_y[side] >= active_min and hand_y[other] <= inactive_max
                    and hand_y[side] - hand_y[other] > separation_min):
                qualified.add(side)
    # A single-arm request also excludes a later bilateral raise. The other
    # arm must stay lowered throughout the complete generated clip.
    qualified = {side for side in qualified
                 if hand_peaks['right' if side == 'left' else 'left'] <= inactive_max}
    if len(qualified) != 1:
        raise ValueError(
            'T2M generated an ambiguous hand raise: expected one raised hand '
            'near the head with the other hand clearly lowered'
        )
    return qualified.pop()


def correct_generated_hand_raise(
    bvh: str, constraint: HandRaiseConstraint,
) -> tuple[str, str, bool]:
    """Correct only a proven opposite-side raise; retain its generated poses."""
    parsed = _parse_bvh(bvh)
    generated_side = _raised_side(parsed, constraint.overhead)
    mirrored = constraint.requested_side is not None and generated_side != constraint.requested_side
    corrected = _mirror_parsed(parsed) if mirrored else bvh
    return corrected, generated_side, mirrored
