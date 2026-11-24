import math
import unittest
from unittest.mock import Mock

from gesture_pose import GesturePose, rotate_toward
from retarget import IDENTITY, dot


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

    def test_antipodal_representation_has_no_flip(self):
        q = around_z(172)
        self.assertLess(distance(rotate_toward(q, tuple(-x for x in q), 0.3), q), 1e-5)
