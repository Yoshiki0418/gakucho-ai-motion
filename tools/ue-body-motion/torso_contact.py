"""Keep the forearm, palm and fingertips outside the president's jacket."""
import math

from retarget import IDENTITY, ZERO, add, cross, dot, inverse, mul, rotate, scale, sub, swing, unit


def _chain(rig, leaves, root=None):
    needed = set()
    for name in leaves:
        branch = []
        while name is not None and name != root and name not in needed:
            branch.append(name)
            name = rig[name][0]
        if name == root or name in needed:
            needed.update(branch)
    return [(n, *rig[n]) for n in rig if n in needed]


def _positions(chain, rotations, root=None):
    result = {root: (ZERO, IDENTITY)}
    for name, parent, offset, rest in chain:
        pos, orient = result[parent]
        result[name] = add(pos, rotate(orient, offset)), mul(orient, rotations.get(name, rest))
    return result


class Jacket:
    def __init__(self, pelvis, left, right):
        self.origin = pelvis
        axis = sub(scale(add(left, right), 0.5), pelvis)
        self.height = math.sqrt(dot(axis, axis))
        self.up = unit(axis)
        shoulders = sub(left, right)
        self.half_width = math.sqrt(dot(shoulders, shoulders)) * 0.5
        self.front = unit(cross(self.up, shoulders))
        self.side = cross(self.front, self.up)

    def push(self, point, pad):
        local = sub(point, self.origin)
        height = dot(local, self.up) / self.height
        # Rounded hem and collar, with widths scaled to the current skeleton.
        # At the middle of the jacket the front is about half a shoulder span
        # from the spine. Probes include the thickness of the hand or sleeve.
        cap = max(0.0, (-0.18 - height) / 0.28, (height - 0.94) / 0.27)
        if cap >= 1:
            return ZERO
        section = math.sqrt(1 - cap * cap)
        chest = max(0, min(1, (height - 0.52) / 0.42))
        width = section * self.half_width * (1.11 + 0.08 * chest) + pad
        depth = section * self.half_width * (1.00 - 0.12 * chest) + pad
        x, y = dot(local, self.side), dot(local, self.front)
        nx, ny = x / width, y / depth
        inside = nx * nx + ny * ny
        if inside >= 1:
            return ZERO
        # A hand resting against the chest slides along the front, including
        # across the midline. At the flanks it clears the near side.
        dx, dy = nx ** 3, ny
        if abs(nx) < 0.75 and dy > -0.25:
            dy = max(0.5, dy)
        if abs(dx) + abs(dy) < 1e-6:
            dy = 1.0
        a = (dx / width) ** 2 + (dy / depth) ** 2
        b = nx * dx / width + ny * dy / depth
        travel = (math.sqrt(b * b + a * (1 - inside)) - b) / a
        return add(scale(self.side, dx * travel), scale(self.front, dy * travel))


class TorsoContact:
    def __init__(self, rig, arms):
        self.chain = _chain(rig, ("pelvis", "upperarm_l", "upperarm_r"))
        self.hands = []
        for arm in arms:
            side = arm.hand[-1]
            # Use animated finger joints and extrapolate the last phalanx to
            # the fingertip. Sparse rigs still get forearm and wrist checks.
            probes = []
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                distal = f"{finger}_03_{side}"
                if distal in rig:
                    probes.append((distal, scale(rig[distal][1], 0.65), 1.45))
            knuckle = f"middle_01_{side}"
            if knuckle in rig:
                probes.append((knuckle, ZERO, 3.0))
            fingers = _chain(rig, (n for n, _, _ in probes), arm.hand)
            connected = {n for n, _, _, _ in fingers}
            probes = [p for p in probes if p[0] in connected]
            self.hands.append((arm, rig[arm.upper][0], fingers, probes))
        self.idle = rig

    def correct(self, rotations):
        torso = _positions(self.chain, rotations)
        try:
            jacket = Jacket(torso["pelvis"][0], torso["upperarm_l"][0], torso["upperarm_r"][0])
        except ValueError:
            # A collapsed shoulder span cannot define a torso-facing direction.
            return
        for arm, parent, chain, probes in self.hands:
            fingers = _positions(chain, rotations, arm.hand)
            pads = [(add(fingers[n][0], rotate(fingers[n][1], tip)), radius) for n, tip, radius in probes]
            self._clear_arm(arm, rotations, torso[parent][1], torso[arm.upper][0], jacket, pads)

    def _clear_arm(self, arm, pose, parent_q, shoulder, jacket, pads):
        length_a, length_b = math.sqrt(dot(arm.arm, arm.arm)), math.sqrt(dot(arm.forearm, arm.forearm))
        for _ in range(5):
            upper, lower, hand = (pose[n] for n in (arm.upper, arm.elbow, arm.hand))
            upper_world = mul(parent_q, upper)
            lower_world = mul(upper_world, lower)
            hand_world = mul(lower_world, hand)
            elbow = add(shoulder, rotate(upper_world, arm.arm))
            wrist = add(elbow, rotate(lower_world, arm.forearm))
            mid = scale(add(elbow, wrist), 0.5)
            points = [(mid, 3.4), (wrist, 3.2)]
            points.extend((add(wrist, rotate(hand_world, p)), radius) for p, radius in pads)
            pushes = [jacket.push(p, radius) for p, radius in points]
            index = max(range(len(pushes)), key=lambda i: dot(pushes[i], pushes[i]))
            push = pushes[index]
            if dot(push, push) < 0.0025:
                return

            if index == 0:
                # Clear a crossing forearm around the shoulder while retaining
                # flexion, even if the wrist itself is already outside.
                ray = sub(mid, shoulder)
                target = add(ray, push)
                if dot(ray, ray) < 1e-8 or dot(target, target) < 1e-8:
                    return
                upper_world = mul(swing(ray, target), upper_world)
            else:
                reach = add(sub(wrist, shoulder), push)
                if dot(reach, reach) < 1e-8:
                    return
                cosine = (dot(reach, reach) - length_a**2 - length_b**2) / (2 * length_a * length_b)
                lower = arm.flex_at(lower, math.acos(max(-1, min(1, cosine))))
                chain = add(arm.arm, rotate(lower, arm.forearm))
                upper_world = mul(swing(rotate(upper_world, chain), reach), upper_world)
            pose[arm.upper] = mul(inverse(parent_q), upper_world)
            pose[arm.elbow] = lower
            wrist_q = mul(inverse(mul(upper_world, lower)), hand_world)
            pose[arm.hand] = arm.limit_wrist(self.idle[arm.hand][2], wrist_q)
