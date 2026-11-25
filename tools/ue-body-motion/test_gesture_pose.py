import math
import unittest
from unittest.mock import Mock

from gesture_pose import Arm, GesturePose, rotate_toward
from retarget import IDENTITY, add, cross, dot, mul, rotate, unit


def around_z(degrees):
    angle = math.radians(degrees) / 2
    return 0, 0, math.sin(angle), math.cos(angle)


def distance(q, other=IDENTITY):
    return math.degrees(2 * math.acos(min(1, abs(dot(q, other)))))


class GestureTest(unittest.TestCase):
    def setUp(self):
        self.names = ("pelvis", "thigh_l", "spine_01", "head", "upperarm_r")
        self.idle = {name: (None, (0, 0, 95), around_z(20)) for name in self.names}
        self.first = {name: around_z(-85) for name in self.names}
        self.moved = {name: around_z(15) for name in self.names}
        self.mapper = Mock(pelvis_rest=(0, 0, 90))
        self.mapper.pose.side_effect = lambda frame: (
            self.moved if frame else self.first, (35, -20, 110)
        )

    def test_gesture_starts_in_visible_pose_and_keeps_a_stationary_base(self):
        gesture = GesturePose(self.mapper, self.idle, self.idle, False)
        pose, pelvis = gesture.sample(5, 0)
        self.assertLess(distance(pose["upperarm_r"], self.idle["upperarm_r"][2]), 1e-5)
        for frame in range(1, 61):
            pose, pelvis = gesture.sample(frame, 1 / 30)
        base = self.idle["pelvis"][2]
        self.assertEqual(pelvis, self.idle["pelvis"][1])
        self.assertLess(distance(pose["thigh_l"], base), 1e-5)
        self.assertLess(distance(pose["spine_01"], base), 1.41)
        self.assertLess(distance(pose["head"], base), 5.01)
        self.assertGreater(distance(pose["upperarm_r"], base), 30)

    def test_revision_seeks_at_current_arm_pose_and_limits_per_frame_rotation(self):
        entry = dict(self.idle, upperarm_r=(None, (0, 0, 95), around_z(-25)))
        gesture = GesturePose(self.mapper, self.idle, entry, False)
        previous = gesture.sample(17, 0)[0]["upperarm_r"]
        self.assertLess(distance(previous, entry["upperarm_r"][2]), 1e-5)
        for frame in (18, 19, 60, 61):
            current = gesture.sample(frame, 1 / 30)[0]["upperarm_r"]
            self.assertLessEqual(distance(previous, current), 120 / 30 + 1e-5)
            previous = current

    def test_whole_body_action_keeps_pelvis_travel_and_leg_rotation(self):
        gesture = GesturePose(self.mapper, self.idle, self.idle, True)
        for frame in range(50):
            pose, pelvis = gesture.sample(frame, 1 / 30)
        self.assertGreater(pelvis[0], 30)
        self.assertLess(distance(pose["thigh_l"], self.moved["thigh_l"]), 0.1)

    def test_backward_and_sideways_elbows_reach_the_target_on_either_side(self):
        for side, sign in (("l", 1), ("r", -1)):
            u, e, h = (f"{n}_{side}" for n in ("upperarm", "lowerarm", "hand"))
            rig = {
                "chest": (None, (0, 0, 0), IDENTITY),
                u: ("chest", (0, 0, 0), IDENTITY),
                e: (u, (25 * sign, 0, 0), around_z(36)),
                h: (e, (23 * sign, 0, 0), IDENTITY),
            }
            arm = Arm(side, rig)
            visible = {"chest": around_z(-12)}
            for lower in (around_z(-44), (0, 0.5, 0, math.sqrt(0.75)), around_z(105)):
                with self.subTest(side=side, lower=lower):
                    raw = {"chest": around_z(20), u: around_z(32), e: lower, h: IDENTITY}
                    upper, elbow, wrist = arm.solve(raw, visible, rig)
                    direction = rotate(elbow, arm.forearm_axis)
                    self.assertGreaterEqual(dot(cross(arm.axis, direction), arm.hinge), -1e-7)
                    self.assertAlmostEqual(dot(arm.hinge, direction), 0)
                    wanted = rotate(mul(raw["chest"], raw[u]), add(arm.arm, rotate(lower, arm.forearm)))
                    actual = rotate(mul(visible["chest"], upper), add(arm.arm, rotate(elbow, arm.forearm)))
                    self.assertLess(math.dist(actual, wanted), 1e-6)
                    self.assertLessEqual(distance(wrist), 65.0001)

    def test_fully_folded_equal_length_arm_keeps_a_finite_limited_pose(self):
        rig = {
            "upperarm_l": (None, (0, 0, 0), IDENTITY),
            "lowerarm_l": ("upperarm_l", (24, 0, 0), around_z(30)),
            "hand_l": ("lowerarm_l", (24, 0, 0), IDENTITY),
        }
        arm = Arm("l", rig)
        raw = {n: item[2] for n, item in rig.items()}
        raw[arm.elbow] = around_z(180)
        upper, elbow, wrist = arm.solve(raw, {}, rig)
        self.assertTrue(all(math.isfinite(x) for q in (upper, elbow, wrist) for x in q))
        self.assertAlmostEqual(math.degrees(math.acos(dot(unit(arm.arm), rotate(elbow, arm.forearm_axis)))), 145)

    def test_antipodal_representation_has_no_flip(self):
        q = around_z(172)
        self.assertLess(distance(rotate_toward(q, tuple(-x for x in q), 0.3), q), 1e-5)
