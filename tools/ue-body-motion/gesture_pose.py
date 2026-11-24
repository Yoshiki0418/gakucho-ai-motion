"""Shape a conversational gesture around the actual on-screen idle pose."""
import math

from retarget import add, dot, inverse, mul, scale, sub, unit


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
    def __init__(self, mapper, idle, entry, whole_body):
        self.mapper = mapper
        self.idle = idle
        self.whole_body = whole_body
        self.first = mapper.pose(0)[0]
        self.rotations = {name: entry[name][2] for name in self.first}
        self.pelvis = entry["pelvis"][1]

    def sample(self, frame, dt):
        mapped, pelvis = self.mapper.pose(frame)
        # Use source time, so results are stable across different editor tick rates.
        alpha = -math.expm1(-dt / (0.12 if self.whole_body else 0.18))
        max_speed = math.radians(190 if self.whole_body else 120)
        next_rotations = {}
        for name, q in mapped.items():
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
        self.rotations = next_rotations
        destination = self.idle["pelvis"][1]
        if self.whole_body:
            destination = add(destination, sub(pelvis, self.mapper.pelvis_rest))
        self.pelvis = add(self.pelvis, scale(sub(destination, self.pelvis), alpha))
        return self.rotations, self.pelvis
