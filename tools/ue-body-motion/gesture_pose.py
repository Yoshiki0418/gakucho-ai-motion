"""Shape a conversational gesture around the actual on-screen idle pose."""
import math

from torso_contact import TorsoContact
from retarget import IDENTITY, add, cross, dot, inverse, mul, rotate, scale, sub, swing, unit


def rotate_toward(start, target, amount, limit=math.pi):
    """Shortest-arc spherical interpolation with an angular step limit."""
    cosine = dot(start, target)
    if cosine < 0:
        target, cosine = scale(target, -1), -cosine
    half_angle = math.acos(min(1.0, cosine))
    if half_angle < 1e-6:
        return start
    t = min(amount, limit / (2 * half_angle))
    if t >= 1:
        return target
    a = math.sin((1 - t) * half_angle)
    b = math.sin(t * half_angle)
    return unit(add(scale(start, a), scale(target, b)))


def _axis_angle(axis, radians):
    return (*scale(axis, math.sin(radians / 2)), math.cos(radians / 2))


def _twist_angle(q, axis):
    # q and -q describe the same rotation. Choose the signed shortest twist.
    return math.remainder(2 * math.atan2(dot(q[:3], axis), q[3]), 2 * math.pi)


class Arm:
    """Match the captured wrist direction with the target rig's elbow hinge."""

    def __init__(self, side, reference):
        self.upper, self.elbow, self.hand = (f"{joint}_{side}" for joint in ("upperarm", "lowerarm", "hand"))
        parent, self.arm, rest = reference[self.elbow]
        if parent != self.upper or reference[self.hand][0] != self.elbow:
            raise ValueError("Expected a connected upper arm, forearm and hand")
        self.forearm = reference[self.hand][1]
        self.axis = unit(self.arm)
        self.forearm_axis = unit(self.forearm)
        bent = rotate(rest, self.forearm_axis)
        self.hinge = unit(cross(self.axis, bent))
        self.straight = mul(swing(bent, self.axis), rest)
        self.parents = []
        parent = reference[self.upper][0]
        while parent:
            self.parents.append(parent)
            parent = reference[parent][0]
        self.parents.reverse()

    def solve(self, mapped, visible, idle):
        # The conversational torso is quieter than the capture. Convert the
        # source shoulder to the parent frame that will actually be displayed.
        source_parent = visible_parent = IDENTITY
        for n in self.parents:
            source_parent = mul(source_parent, mapped.get(n, idle[n][2]))
            visible_parent = mul(visible_parent, visible.get(n, idle[n][2]))
        shoulder = mul(mul(inverse(visible_parent), source_parent), mapped[self.upper])
        lower = mapped[self.elbow]
        wanted = rotate(lower, self.forearm)
        elbow = self.flex(lower)

        # The elbow can now flex only forwards. Aim the shoulder so that
        # straightening a sideways/backwards bend preserves the wrist target.
        reach = add(self.arm, wanted)
        aim = swing(add(self.arm, rotate(elbow, self.forearm)), reach) if dot(reach, reach) > 1e-8 else IDENTITY
        upper = mul(shoulder, aim)
        hand = mul(mul(inverse(mul(upper, elbow)), mul(shoulder, lower)), mapped[self.hand])
        return upper, elbow, self.limit_wrist(idle[self.hand][2], hand)

    @staticmethod
    def limit_wrist(neutral, target):
        return rotate_toward(neutral, target, 1.0, math.radians(65))

    def flex(self, lower):
        direction = rotate(lower, self.forearm_axis)
        bend = math.acos(max(-1.0, min(1.0, dot(self.axis, direction))))
        return self.flex_at(lower, bend)

    def flex_at(self, lower, bend):
        flex = mul(_axis_angle(self.hinge, min(math.radians(145), bend)), self.straight)
        twist = _twist_angle(mul(inverse(flex), lower), self.forearm_axis)
        twist = max(-math.radians(75), min(math.radians(75), twist))
        return mul(flex, _axis_angle(self.forearm_axis, twist))


def _joint_response(name):
    # Several local spine and neck rotations accumulate at the camera-facing head.
    if name.startswith("spine"):
        return 0.22, 1.4
    if name.startswith("neck") or name == "head":
        return (0.48, 5.0) if name == "head" else (0.32, 2.5)
    if name.startswith("clavicle"):
        return 0.55, 9
    if name.startswith(("thigh", "calf", "foot", "ball")) or name == "pelvis":
        return 0, 0
    return 0.72, 65


class GesturePose:
    def __init__(self, mapper, idle, entry, whole_body, reference=None):
        self.mapper = mapper
        self.idle = idle
        self.whole_body = whole_body
        self.first = mapper.pose(0)[0]
        self.arms = []
        for side in ("l", "r"):
            if reference and all(f"{n}_{side}" in self.first for n in ("upperarm", "lowerarm", "hand")):
                try:
                    self.arms.append(Arm(side, reference))
                except ValueError:
                    # A straight bind pose or missing chain has no calibrated hinge.
                    pass
        self.contact = (TorsoContact(idle, self.arms) if self.arms and
                        all(n in idle for n in ("pelvis", "upperarm_l", "upperarm_r")) else None)
        self.arm_bones = {name for arm in self.arms for name in (arm.upper, arm.elbow, arm.hand)}
        self.rotations = {name: entry[name][2] for name in self.first}
        self.pelvis = entry["pelvis"][1]

    def sample(self, frame, dt):
        if dt <= 0:
            return self.rotations, self.pelvis
        mapped, pelvis = self.mapper.pose(frame)
        # Use source time, so results are stable across different editor tick rates.
        alpha = -math.expm1(-dt / (0.12 if self.whole_body else 0.18))
        max_speed = math.radians(190 if self.whole_body else 120)
        next_rotations = {}
        for name, q in mapped.items():
            if name in self.arm_bones:
                continue
            base = self.idle[name][2]
            if self.whole_body:
                target = q
            else:
                gain, degrees = _joint_response(name)
                delta = mul(q, inverse(self.first[name]))
                target = rotate_toward(base, mul(delta, base), gain, math.radians(degrees))
            next_rotations[name] = rotate_toward(
                self.rotations[name], target, alpha, max_speed * dt
            )
        for arm in self.arms:
            names = (arm.upper, arm.elbow, arm.hand)
            # Idle recordings and the resting arm of a one-handed action
            # should keep the president's own pose. Use the recorded reach
            # when that arm becomes active, including mirrored gestures.
            change = min(abs(dot(mapped[n], self.first[n])) for n in names)
            angle = 2 * math.acos(min(1.0, change))
            weight = 1.0 if self.whole_body else min(1.0, angle / math.radians(30))
            for n, target in zip(names, arm.solve(mapped, next_rotations, self.idle)):
                target = rotate_toward(self.idle[n][2], target, weight)
                next_rotations[n] = rotate_toward(self.rotations[n], target, alpha, math.radians(190) * dt)
            next_rotations[arm.elbow] = arm.flex(next_rotations[arm.elbow])
        if self.contact:
            self.contact.correct(next_rotations)
        self.rotations = next_rotations
        destination = self.idle["pelvis"][1]
        if self.whole_body:
            destination = add(destination, sub(pelvis, self.mapper.pelvis_rest))
        self.pelvis = add(self.pelvis, scale(sub(destination, self.pelvis), alpha))
        return self.rotations, self.pelvis
