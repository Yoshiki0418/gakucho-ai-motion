"""Handedness corrections preserve generated BVH kinematics and timing."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from test_motion_backend_contract import BACKENDS

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app/agent/general_conversation/tools/motion_handedness.py"
)
SPEC = importlib.util.spec_from_file_location(
    "motion_handedness_test_module", MODULE_PATH
)
HANDEDNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HANDEDNESS
SPEC.loader.exec_module(HANDEDNESS)

HEADER = """HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Spine
 {
  OFFSET 0 0.4 0.1
  CHANNELS 3 Zrotation Xrotation Yrotation
  JOINT LeftArm
  {
   OFFSET 0.2 0.15 0
   CHANNELS 3 Zrotation Xrotation Yrotation
   JOINT LeftHand
   {
    OFFSET 0.4 0 0.02
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
     OFFSET 0.1 0 0
    }
   }
  }
  JOINT RightArm
  {
   OFFSET -0.2 0.15 0
   CHANNELS 3 Zrotation Xrotation Yrotation
   JOINT RightHand
   {
    OFFSET -0.4 0 0.02
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
     OFFSET -0.1 0 0
    }
   }
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.01
"""
VALUES = np.array(
    [
        [
            0.12345678901234567,
            0.9,
            -0.2,
            5,
            7,
            -11,
            -2,
            4,
            3,
            -40,
            10,
            8,
            12,
            4,
            -18,
            25,
            -12,
            -3,
            -5,
            20,
            2,
        ],
        [
            0.16,
            0.93,
            -0.15,
            -13,
            9,
            27,
            15,
            -3,
            6,
            30,
            -8,
            40,
            -25,
            9,
            13,
            -55,
            6,
            -10,
            21,
            -7,
            4,
        ],
    ]
)
BVH = (
    HEADER
    + "\n".join(" ".join(format(v, ".17g") for v in row) for row in VALUES)
    + "\n"
)


def motion_rows(bvh):
    return np.array(
        [
            [float(value) for value in row.split()]
            for row in bvh.split("Frame Time: 0.01", 1)[1].splitlines()
            if row.strip()
        ]
    )


def rotation(zxy):
    """Independent elemental Z-X-Y rotation oracle."""
    z, x, y = np.deg2rad(zxy)
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry


def world_kinematics(row):
    """FK from the physical fixture, independent of the helper's parser."""
    offsets = [
        np.zeros(3),
        np.array([0, 0.4, 0.1]),
        np.array([0.2, 0.15, 0]),
        np.array([0.4, 0, 0.02]),
        np.array([-0.2, 0.15, 0]),
        np.array([-0.4, 0, 0.02]),
    ]
    parents = [-1, 0, 1, 2, 1, 4]
    starts = [3, 6, 9, 12, 15, 18]
    positions, rotations = [], []
    for index, (offset, parent, start) in enumerate(zip(offsets, parents, starts)):
        local = rotation(row[start : start + 3])
        if parent == -1:
            positions.append(row[:3])
            rotations.append(local)
        else:
            positions.append(positions[parent] + rotations[parent] @ offset)
            rotations.append(rotations[parent] @ local)
    return np.array(positions), np.array(rotations)


class HandednessTransformTest(unittest.TestCase):
    def test_world_motion_is_the_exact_sagittal_reflection_of_opposite_joint(self):
        mirrored = motion_rows(HANDEDNESS.mirror_bvh_sagittal(BVH))
        matrix = np.diag([-1, 1, 1])
        opposite = [0, 1, 4, 5, 2, 3]
        for before, after in zip(VALUES, mirrored):
            old_pos, old_rot = world_kinematics(before)
            new_pos, new_rot = world_kinematics(after)
            np.testing.assert_allclose(new_pos, old_pos[opposite] @ matrix, atol=1e-14)
            np.testing.assert_allclose(
                new_rot, matrix @ old_rot[opposite] @ matrix, atol=1e-14
            )

    def test_double_mirror_restores_all_values_without_precision_loss(self):
        mirrored = HANDEDNESS.mirror_bvh_sagittal(BVH)
        twice = HANDEDNESS.mirror_bvh_sagittal(mirrored)
        np.testing.assert_array_equal(motion_rows(twice), VALUES)
        self.assertTrue(mirrored.startswith(HEADER))
        self.assertTrue(twice.startswith(HEADER))

    def test_windows_header_and_frame_timing_survive(self):
        mirrored = HANDEDNESS.mirror_bvh_sagittal(BVH.replace("\n", "\r\n"))
        self.assertTrue(mirrored.startswith(HEADER.replace("\n", "\r\n")))
        self.assertEqual(len(motion_rows(mirrored)), 2)

    def test_asymmetric_rig_is_rejected_instead_of_producing_wrong_motion(self):
        with self.assertRaisesRegex(ValueError, "not mirror-symmetric"):
            HANDEDNESS.mirror_bvh_sagittal(
                BVH.replace("OFFSET -0.4 0 0.02", "OFFSET -0.41 0 0.02")
            )

    def test_mismatched_paired_channels_are_rejected(self):
        bad = BVH.replace(
            "OFFSET -0.2 0.15 0\n   CHANNELS 3 Zrotation Xrotation Yrotation",
            "OFFSET -0.2 0.15 0\n   CHANNELS 3 Xrotation Yrotation Zrotation",
        )
        with self.assertRaisesRegex(ValueError, "channels differ"):
            HANDEDNESS.mirror_bvh_sagittal(bad)

    def test_nonfinite_or_malformed_motion_is_rejected(self):
        for bvh in (
            BVH.replace("Frames: 2", "Frames: 3"),
            BVH.replace("0.90000000000000002", "nan"),
            BVH.replace("Frame Time: 0.01", "Frame Time: 0"),
        ):
            with self.subTest(bvh=bvh[-100:]), self.assertRaises(ValueError):
                HANDEDNESS.mirror_bvh_sagittal(bvh)


class NarrowRequestTest(unittest.TestCase):
    def test_simple_left_requests_use_the_verified_right_generation_caption(self):
        for prompt in (
            "左手を振る",
            "立っている状態から、左手を振る。",
            "左手を振ってください",
        ):
            self.assertEqual(
                HANDEDNESS.simple_left_generation_caption(prompt),
                HANDEDNESS.RIGHT_WAVE_CAPTION,
            )
        for prompt in (
            "左手を上げる",
            "左腕を上げる",
            "立っている状態で、左手を上げる。",
        ):
            self.assertEqual(
                HANDEDNESS.simple_left_generation_caption(prompt),
                HANDEDNESS.RIGHT_RAISE_CAPTION,
            )

    def test_detailed_bilateral_or_other_requests_remain_unmodified(self):
        for prompt in (
            "左手を頭上に高く上げる",
            "左手を胸の前で小さく振る",
            "左手だけを振る",
            "両手を振る",
            "右手を振る",
            "左を指差す",
            "座ったまま左手を振る",
            "左手を上げてから右手を振る",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(HANDEDNESS.simple_left_generation_caption(prompt))


class BackendHandednessIntegrationTest(unittest.TestCase):
    def test_left_motion_is_generated_and_mirrored_with_original_prompt(self):
        for prompt, caption in [("左手を振る", HANDEDNESS.RIGHT_WAVE_CAPTION)]:
            with self.subTest(prompt=prompt), patch.dict("os.environ", {}, clear=True):
                backend = BACKENDS.CustomT2MBackend()
                with patch.object(
                    backend,
                    "_request_generation",
                    return_value={
                        "bvh_path": "generated.bvh",
                        "latency_seconds": 0.3,
                    },
                ) as generate, patch.object(backend, "_download_bvh", return_value=BVH):
                    result = backend.generate(BACKENDS.MotionRequest(prompt=prompt))
                generate.assert_called_once_with(caption)
                self.assertEqual(result.prompt, prompt)
                self.assertEqual(result.backend, "custom-t2m")
                self.assertEqual(result.latency_seconds, 0.3)
                self.assertEqual(result.bvh, HANDEDNESS.mirror_bvh_sagittal(BVH))

    def test_detailed_left_request_is_forwarded_without_mirroring(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = BACKENDS.CustomT2MBackend()
        prompt = "左手を胸の前で小さく振る"
        with patch.object(
            backend, "_request_generation", return_value={"bvh_path": "generated.bvh"}
        ) as generate, patch.object(
            backend, "_download_bvh", return_value=BVH
        ), patch.object(
            BACKENDS, "mirror_bvh_sagittal"
        ) as mirror:
            result = backend.generate(BACKENDS.MotionRequest(prompt=prompt))
        generate.assert_called_once_with(prompt)
        mirror.assert_not_called()
        self.assertEqual(result.bvh, BVH)


RAISE_HEADER = (
    HEADER.replace("JOINT Spine", "JOINT Head")
    .replace("OFFSET 0 0.4 0.1", "OFFSET 0 0.64 0")
    .replace("OFFSET 0.2 0.15 0", "OFFSET 0.2 -0.19 0")
    .replace("OFFSET -0.2 0.15 0", "OFFSET -0.2 -0.19 0")
    .replace("OFFSET 0.4 0 0.02", "OFFSET 0.5 0 0")
    .replace("OFFSET -0.4 0 0.02", "OFFSET -0.5 0 0")
)
# The hands descend from Head in this small symmetric physical fixture.
# Their arm origins are 0.45 m above the hips; an inactive arm points down.
LEFT_RAISE_VALUES = np.array(
    [
        [0.123, 0.9, -0.2, 0, 0, 0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0],
        [0.16, 0.92, -0.15, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0],
    ]
)


def raise_bvh(values):
    return (
        RAISE_HEADER
        + "\n".join(" ".join(format(v, ".17g") for v in row) for row in values)
        + "\n"
    )


def raised_fixture_positions(row):
    # Independent, direct FK for each root -> head -> arm -> hand chain.
    root_position, root_rotation = row[:3], rotation(row[3:6])
    head_rotation = root_rotation @ rotation(row[6:9])
    head_position = root_position + root_rotation @ np.array([0, 0.64, 0])
    left_arm = head_position + head_rotation @ np.array([0.2, -0.19, 0])
    right_arm = head_position + head_rotation @ np.array([-0.2, -0.19, 0])
    left_hand = left_arm + head_rotation @ rotation(row[9:12]) @ np.array([0.5, 0, 0])
    right_hand = right_arm + head_rotation @ rotation(row[15:18]) @ np.array(
        [-0.5, 0, 0]
    )
    return left_hand, right_hand, head_position


class GeneratedRaiseQualificationTest(unittest.TestCase):
    def test_already_requested_left_is_kept_without_a_second_side_reversal(self):
        bvh = raise_bvh(LEFT_RAISE_VALUES)
        result, side, mirrored = HANDEDNESS.correct_generated_hand_raise(
            bvh, HANDEDNESS.HandRaiseConstraint("left")
        )
        self.assertEqual(result, bvh)
        self.assertEqual(side, "left")
        self.assertFalse(mirrored)

    def test_generated_left_is_reflected_only_when_requested_side_is_right(self):
        result, side, mirrored = HANDEDNESS.correct_generated_hand_raise(
            raise_bvh(LEFT_RAISE_VALUES), HANDEDNESS.HandRaiseConstraint("right")
        )
        self.assertEqual(side, "left")
        self.assertTrue(mirrored)
        left, right, head = raised_fixture_positions(motion_rows(result)[1])
        self.assertGreater(right[1] - head[1], 0.1)
        self.assertLess(left[1] - head[1], -0.2)

    def test_generated_right_can_be_corrected_to_left(self):
        right = HANDEDNESS.mirror_bvh_sagittal(raise_bvh(LEFT_RAISE_VALUES))
        result, side, mirrored = HANDEDNESS.correct_generated_hand_raise(
            right, HANDEDNESS.HandRaiseConstraint("left")
        )
        self.assertEqual(side, "right")
        self.assertTrue(mirrored)
        np.testing.assert_array_equal(motion_rows(result), LEFT_RAISE_VALUES)

    def test_unspecified_hand_keeps_the_qualified_generated_side(self):
        bvh = raise_bvh(LEFT_RAISE_VALUES)
        result, side, mirrored = HANDEDNESS.correct_generated_hand_raise(
            bvh, HANDEDNESS.HandRaiseConstraint(None)
        )
        self.assertEqual(result, bvh)
        self.assertEqual(side, "left")
        self.assertFalse(mirrored)

    def test_both_or_neither_raised_hands_are_rejected(self):
        both = LEFT_RAISE_VALUES.copy()
        both[1, 15] = -60
        neither = LEFT_RAISE_VALUES.copy()
        neither[1, 9] = -90
        successive = LEFT_RAISE_VALUES.copy()
        successive[0, 9] = 60
        successive[1, 9] = -90
        successive[1, 15] = -60
        for values in (both, neither, successive):
            with self.subTest(values=values.tolist()), self.assertRaisesRegex(
                ValueError, "ambiguous hand raise"
            ):
                HANDEDNESS.correct_generated_hand_raise(
                    raise_bvh(values), HANDEDNESS.HandRaiseConstraint("left")
                )

    def test_left_raise_followed_by_both_hands_is_rejected(self):
        values = LEFT_RAISE_VALUES.copy()
        values[0, 9] = 60
        values[1, 15] = -60
        # The first frame qualifies as left-only, but the complete motion
        # violates a single-arm request because both arms rise afterward.
        with self.assertRaisesRegex(ValueError, "ambiguous hand raise"):
            HANDEDNESS.correct_generated_hand_raise(
                raise_bvh(values), HANDEDNESS.HandRaiseConstraint("left")
            )

    def test_overhead_constraint_requires_height_above_head(self):
        high, side, mirrored = HANDEDNESS.correct_generated_hand_raise(
            raise_bvh(LEFT_RAISE_VALUES),
            HANDEDNESS.HandRaiseConstraint("right", overhead=True),
        )
        self.assertTrue(mirrored)
        self.assertEqual(side, "left")
        low = LEFT_RAISE_VALUES.copy()
        low[1, 9] = 15
        HANDEDNESS.correct_generated_hand_raise(
            raise_bvh(low), HANDEDNESS.HandRaiseConstraint("left")
        )
        with self.assertRaisesRegex(ValueError, "ambiguous hand raise"):
            HANDEDNESS.correct_generated_hand_raise(
                raise_bvh(low), HANDEDNESS.HandRaiseConstraint("left", overhead=True)
            )

    def test_simple_raise_grammar_preserves_side_and_supported_height(self):
        for prompt, side, high in [
            ("右手を上げる", "right", False),
            ("左腕を上げる", "left", False),
            ("手を上げる", None, False),
            ("立っている状態から、左手を頭上に上げる。", "left", True),
            ("右手を頭の上に高く持ち上げる", "right", True),
            ("左手を高く上げる", "left", True),
        ]:
            with self.subTest(prompt=prompt):
                constraint = HANDEDNESS.simple_hand_raise_constraint(prompt)
                self.assertEqual(constraint.requested_side, side)
                self.assertEqual(constraint.overhead, high)
        for prompt in (
            "左手だけを上げる",
            "両手を上げる",
            "左手をゆっくり上げる",
            "座ったまま右手を上げる",
            "左手を上げてから右手を振る",
            "左手を胸の前に上げる",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(HANDEDNESS.simple_hand_raise_constraint(prompt))


class BackendRaiseIntegrationTest(unittest.TestCase):
    def test_requested_left_generated_left_is_unchanged_and_requested_right_is_reflected(
        self,
    ):
        bvh = raise_bvh(LEFT_RAISE_VALUES)
        for prompt, requested in [
            ("左手を上げる", "left"),
            ("右手を上げる", "right"),
            ("立っている状態から、左手を頭上に上げる。", "left"),
        ]:
            with self.subTest(prompt=prompt), patch.dict("os.environ", {}, clear=True):
                backend = BACKENDS.CustomT2MBackend()
                with patch.object(
                    backend,
                    "_request_generation",
                    return_value={
                        "bvh_path": "generated.bvh",
                        "latency_seconds": 0.3,
                    },
                ) as generate, patch.object(backend, "_download_bvh", return_value=bvh):
                    result = backend.generate(BACKENDS.MotionRequest(prompt=prompt))
                generate.assert_called_once_with(HANDEDNESS.RIGHT_RAISE_CAPTION)
                self.assertEqual(result.prompt, prompt)
                self.assertEqual(result.backend, "custom-t2m")
                self.assertEqual(result.latency_seconds, 0.3)
                self.assertTrue(result.bvh.startswith(RAISE_HEADER))
                left, right, head = raised_fixture_positions(motion_rows(result.bvh)[1])
                active, inactive = (
                    (left, right) if requested == "left" else (right, left)
                )
                self.assertGreater(active[1] - head[1], 0.1)
                self.assertLess(inactive[1] - head[1], -0.2)

    def test_ambiguous_model_raise_fails_instead_of_returning_a_wrong_arm(self):
        values = LEFT_RAISE_VALUES.copy()
        values[1, 15] = -60
        with patch.dict("os.environ", {}, clear=True):
            backend = BACKENDS.CustomT2MBackend()
        with patch.object(
            backend, "_request_generation", return_value={"bvh_path": "generated.bvh"}
        ), patch.object(backend, "_download_bvh", return_value=raise_bvh(values)):
            with self.assertRaisesRegex(ValueError, "ambiguous hand raise"):
                backend.generate(BACKENDS.MotionRequest(prompt="左手を上げる"))


if __name__ == "__main__":
    unittest.main()
