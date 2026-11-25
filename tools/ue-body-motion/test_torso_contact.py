import math
import unittest

from gesture_pose import Arm
from retarget import IDENTITY, ZERO, add, cross, dot, inverse, mul, rotate, scale, sub, swing
from torso_contact import Jacket, TorsoContact


def make_rig():
    rig = {"pelvis": (None, ZERO, IDENTITY)}
    for side, sign in (("l", 1), ("r", -1)):
        upper, lower, hand = (f"{n}_{side}" for n in ("upperarm", "lowerarm", "hand"))
        rig[upper] = ("pelvis", (sign * 15, 0, 40), IDENTITY)
        rig[lower] = (upper, (sign * 24, 0, 0), (0, 0, math.sin(0.3), math.cos(0.3)))
        rig[hand] = (lower, (sign * 23, 0, 0), IDENTITY)
        rig[f"middle_01_{side}"] = (hand, (sign * 7, 0, 0), IDENTITY)
        rig[f"middle_03_{side}"] = (f"middle_01_{side}", (sign * 5, 0, 0), IDENTITY)
    return rig


class ContactTest(unittest.TestCase):
    def setUp(self):
        self.rig = make_rig()
        self.jacket = Jacket(ZERO, (15, 0, 40), (-15, 0, 40))

    def pose_at(self, arm, wrist):
        pose = {n: data[2] for n, data in self.rig.items()}
        reach = sub(wrist, self.rig[arm.upper][1])
        bend = math.acos((dot(reach, reach) - 24**2 - 23**2) / (2 * 24 * 23))
        pose[arm.elbow] = arm.flex_at(pose[arm.elbow], bend)
        pose[arm.upper] = swing(add(arm.arm, rotate(pose[arm.elbow], arm.forearm)), reach)
        return pose

    def pads(self, arm, pose):
        upper = pose[arm.upper]
        lower = mul(upper, pose[arm.elbow])
        hand = mul(lower, pose[arm.hand])
        elbow = add(self.rig[arm.upper][1], rotate(upper, arm.arm))
        wrist = add(elbow, rotate(lower, arm.forearm))
        sign = 1 if arm.hand.endswith("_l") else -1
        tip = add(wrist, rotate(hand, (sign * (7 + 5 + 5 * 0.65), 0, 0)))
        palm = add(wrist, rotate(hand, (sign * 7, 0, 0)))
        return [(scale(add(elbow, wrist), 0.5), 3.4), (wrist, 3.2), (palm, 3.0), (tip, 1.45)]

    def test_crossing_chest_corrects_either_arm_without_reversing_the_elbow(self):
        for side in ("l", "r"):
            with self.subTest(side=side):
                arm = Arm(side, self.rig)
                pose = self.pose_at(arm, (0, 9, 28))
                point, pad = self.pads(arm, pose)[1]
                self.assertGreater(math.dist(self.jacket.push(point, pad), ZERO), 5)
                TorsoContact(self.rig, [arm]).correct(pose)
                for point, pad in self.pads(arm, pose):
                    self.assertLess(math.dist(self.jacket.push(point, pad), ZERO), 0.05)
                direction = rotate(pose[arm.elbow], arm.forearm_axis)
                self.assertAlmostEqual(dot(direction, arm.hinge), 0)
                self.assertGreater(dot(cross(arm.axis, direction), arm.hinge), 0)
                self.assertGreaterEqual(abs(dot(pose[arm.hand], IDENTITY)), math.cos(math.radians(65) / 2) - 1e-9)

    def test_fingers_enter_the_jacket_while_wrist_is_still_clear(self):
        arm = Arm("r", self.rig)
        pose = self.pose_at(arm, (0, 22, 27))
        forearm = mul(pose[arm.upper], pose[arm.elbow])
        pose[arm.hand] = mul(inverse(forearm), swing((-1, 0, 0), (0, -1, 0)))
        wrist, tip = self.pads(arm, pose)[1::2]
        self.assertEqual(self.jacket.push(*wrist), ZERO)
        self.assertGreater(math.dist(self.jacket.push(*tip), ZERO), 3)
        TorsoContact(self.rig, [arm]).correct(pose)
        for point, pad in self.pads(arm, pose):
            self.assertLess(math.dist(self.jacket.push(point, pad), ZERO), 0.05)

    def test_open_arm_and_disconnected_finger_leave_safe_pose_untouched(self):
        self.rig["index_03_r"] = ("pelvis", ZERO, IDENTITY)
        arm = Arm("r", self.rig)
        pose = {n: v[2] for n, v in self.rig.items()}
        original = pose.copy()
        TorsoContact(self.rig, [arm]).correct(pose)
        self.assertEqual(pose, original)
