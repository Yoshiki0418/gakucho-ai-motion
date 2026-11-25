import math
import unittest
from unittest.mock import Mock

from gesture_pose import Arm, GesturePose, rotate_toward
from retarget import IDENTITY, add, cross, dot, mul, rotate, unit


def around_z(degrees):
    angle = math.radians(degrees) / 2
    return 0, 0, math.sin(angle), math.cos(angle)


def wrist_rig(side, sign):
    upper, elbow, hand = (f"{name}_{side}" for name in ("upperarm", "lowerarm", "hand"))
    return {
        upper: (None, (0, 0, 0), IDENTITY),
        elbow: (upper, (25 * sign, 0, 0), around_z(30)),
        hand: (elbow, (23 * sign, 0, 0), IDENTITY),
        f"index_metacarpal_{side}": (hand, (3 * sign, 1, 0), IDENTITY),
        f"pinky_metacarpal_{side}": (hand, (3 * sign, -1, 0), IDENTITY),
    }


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

    def test_wrist_keeps_palm_flexion_and_limits_sideways_bending_on_both_hands(self):
        for side, sign in (("l", 1), ("r", -1)):
            with self.subTest(side=side):
                arm = Arm(side, wrist_rig(side, sign))
                sideways = arm.limit_wrist(IDENTITY, around_z(sign * 60))
                self.assertLess(distance(sideways, around_z(sign * 22)), 1e-5)
                palm_flex = (0, math.sin(math.radians(20)), 0, math.cos(math.radians(20)))
                self.assertLess(distance(arm.limit_wrist(IDENTITY, palm_flex), palm_flex), 1e-5)
                self.assertLess(distance(arm.limit_wrist(IDENTITY, sideways), sideways), 1e-5)

    def test_turning_the_palm_rolls_the_forearm_and_keeps_the_hand_position_and_orientation(self):
        for side, sign in (("l", 1), ("r", -1)):
            with self.subTest(side=side):
                rig = wrist_rig(side, sign)
                arm = Arm(side, rig)
                raw = {n: item[2] for n, item in rig.items()}
                raw[arm.hand] = (sign * 0.5, 0, 0, math.sqrt(0.75))
                upper, elbow, wrist = arm.solve(raw, {}, rig)
                self.assertAlmostEqual(distance(elbow, raw[arm.elbow]), 48)
                self.assertAlmostEqual(distance(wrist), 12)
                self.assertLess(math.dist(rotate(elbow, arm.forearm), rotate(raw[arm.elbow], arm.forearm)), 1e-6)
                original_palm = mul(raw[arm.elbow], raw[arm.hand])
                self.assertLess(distance(mul(mul(upper, elbow), wrist), original_palm), 1e-5)

    def test_conversational_lift_compacts_a_folded_arm_and_preserves_an_extended_reach(self):
        for side, sign in (("l", 1), ("r", -1)):
            rig = wrist_rig(side, sign)
            arm = Arm(side, rig)
            raw = {n: item[2] for n, item in rig.items()}
            raw[arm.upper] = around_z(90)
            for bend in (10, 85):
                with self.subTest(side=side, bend=bend):
                    raw[arm.elbow] = around_z(bend)
                    full = arm.solve(raw, {}, rig)[0]
                    conversational = arm.solve(raw, {}, rig, conversational=True)[0]
                    self.assertAlmostEqual(distance(full), 90)
                    if bend == 10:
                        self.assertLess(distance(full, conversational), 1e-5)
                    else:
                        self.assertLess(distance(conversational), 75)

    def test_conversational_stroke_reaches_the_wrist_and_elbow_faster_than_the_upper_arm(self):
        rig = wrist_rig("l", 1)
        rig["pelvis"] = (None, (0, 0, 95), IDENTITY)
        rest = {n: rig[n][2] for n in ("upperarm_l", "lowerarm_l", "hand_l")}
        palm_flex = (0, math.sin(math.radians(10)), 0, math.cos(math.radians(10)))
        stroke = dict(upperarm_l=around_z(20), lowerarm_l=around_z(65), hand_l=palm_flex)
        mapper = Mock()
        mapper.pose.side_effect = lambda frame: (stroke if frame else rest, rig["pelvis"][1])
        pose, _ = GesturePose(mapper, rig, rig, False, rig).sample(1, 1 / 30)
        fraction = {n: distance(pose[n], rest[n]) / distance(stroke[n], rest[n]) for n in rest}
        self.assertLess(fraction["upperarm_l"], fraction["lowerarm_l"])
        self.assertLess(fraction["lowerarm_l"], fraction["hand_l"])

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
