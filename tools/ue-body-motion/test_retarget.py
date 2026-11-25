"""Exercise generated BVH with an A-pose, a helper bone, and a rotated mesh root."""

import math
import re
import unittest
from pathlib import Path

import retarget as rt

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (ROOT / "backend/app/assets/conversation_motion/skeleton.bvh").read_text()
ANGLE = math.radians(45)
A_POSE = (math.sin(ANGLE / 2), 0, 0, math.cos(ANGLE / 2))
# The upper arm's child offset is horizontal in bone-local space. In the mesh
# reference pose the parent's rotation points that arm down, away from the BVH T.
REFERENCE = {
    "root": (None, rt.ZERO, rt.IDENTITY),
    "pelvis": ("root", (0, 0, 95), rt.IDENTITY),
    "spine_01": ("pelvis", (0, 0, 8), rt.IDENTITY),
    "head": ("spine_01", (0, 0, 49), rt.IDENTITY),
    "clavicle_l": ("spine_01", (0, -5, 40), rt.IDENTITY),
    "upperarm_l": ("clavicle_l", (0, -12, 0), A_POSE),
    "helper_l": ("upperarm_l", (0, -10, 0), rt.IDENTITY),
    "lowerarm_l": ("helper_l", (0, -20, 0), rt.IDENTITY),
    "hand_l": ("lowerarm_l", (0, -25, 0), rt.IDENTITY),
    "thigh_l": ("pelvis", (0, -10, 0), rt.IDENTITY),
    "calf_l": ("thigh_l", (0, 0, -43), rt.IDENTITY),
    "foot_l": ("calf_l", (0, 0, -41), rt.IDENTITY),
    "thigh_r": ("pelvis", (0, 10, 0), rt.IDENTITY),
}


def generated_clip():
    shape = rt.parse_bvh(TEMPLATE)
    a = [0.0] * len(shape.rows[0])
    b = a.copy()
    a[:3], b[:3] = (19, 95, -30), (19.2, 95, -30)
    arm = next(j for j in shape.joints if j.name == "LeftArm")
    z_column = next(i for i, axis in arm.rotations if axis == 2)
    b[z_column] = 90
    header = TEMPLATE.partition("MOTION")[0]
    return (
        header
        + "MOTION\nFrames: 2\nFrame Time: 0.033333\n"
        + "\n".join(" ".join(map(str, row)) for row in (a, b))
    )


def posed_positions(ref, pose):
    rotations, pelvis = pose
    xyz, world = {}, {}
    for key, (parent, t, q) in ref.items():
        parent_q = world.get(parent, rt.IDENTITY)
        local = pelvis if key == "pelvis" else t
        xyz[key] = rt.add(xyz.get(parent, rt.ZERO), rt.rotate(parent_q, local))
        world[key] = rt.mul(parent_q, rotations.get(key, q))
    return xyz


class RetargetTests(unittest.TestCase):
    def assertVector(self, actual, expected, places=5):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def test_raised_arm_moves_up_on_left_through_unmapped_helper(self):
        clip = rt.parse_bvh(generated_clip())
        retarget = rt.prepare_retarget(clip, REFERENCE)
        before = posed_positions(REFERENCE, retarget.pose(0))
        after = posed_positions(REFERENCE, retarget.pose(1))

        def direction(positions):
            return rt.unit(rt.sub(positions["lowerarm_l"], positions["upperarm_l"]))

        self.assertVector(direction(before), (0, -1, 0))
        self.assertVector(direction(after), (0, 0, 1))
        self.assertNotIn("helper_l", retarget.pose(1)[0])
        self.assertVector(retarget.pose(0)[1], (0, 0, 95))
        self.assertVector(retarget.pose(1)[1], (0, -retarget.scale * 0.2, 95), places=3)

    def test_source_zxy_composes_in_declared_order(self):
        clip = rt.parse_bvh(generated_clip())
        root = clip.joints[0]
        # Z(90) * X(90) maps BVH up to BVH forward.
        for i, axis in root.rotations:
            clip.rows[0][i] = (90, 0, 90)[axis]
        self.assertVector(rt.rotate(clip.world_rotations(0)[0], (0, 1, 0)), (0, 0, 1))

    def test_root_motion_uses_parent_axes_and_relative_distance(self):
        rotated = {**REFERENCE, "root": (None, rt.ZERO, (0, 0, 0.5, math.sqrt(0.75)))}
        mapper = rt.prepare_retarget(rt.parse_bvh(generated_clip()), rotated)
        self.assertVector(mapper.pose(1)[1], (0, -0.2 * mapper.scale, 95), places=3)
        before = posed_positions(rotated, mapper.pose(0))
        after = posed_positions(rotated, mapper.pose(1))
        left = rt.unit(rt.sub(before["thigh_l"], before["thigh_r"]))
        self.assertVector(rt.unit(rt.sub(after["pelvis"], before["pelvis"])), left)

    def test_accepts_existing_text_to_motion_file(self):
        clip = rt.parse_bvh((ROOT / "100.bvh").read_text())
        retarget = rt.prepare_retarget(clip, REFERENCE)
        first = retarget.pose(0)[0]["upperarm_l"]
        last = retarget.pose(clip.frame_count - 1)[0]["upperarm_l"]
        self.assertLess(abs(rt.dot(first, last)), 0.99999)

    def test_ue_named_capture_includes_rest_rotations_in_frames(self):
        # A SuSu-style BVH has UE bone names and its first frame already
        # contains the rig's joint pre-rotations. Include a posed left arm.
        text = re.sub(
            r"((?:ROOT|JOINT) )(\w+)",
            lambda m: m[1] + rt.NAMES[m[2].lower()],
            generated_clip(),
        )
        clip = rt.parse_bvh(text)
        arm = next(j for j in clip.joints if j.name == "upperarm_l")
        z_channel = next(i for i, axis in arm.rotations if axis == 2)
        clip.rows[0][z_channel] = -30
        mapper = rt.prepare_retarget(clip, REFERENCE)
        first = posed_positions(REFERENCE, mapper.pose(0))
        after = posed_positions(REFERENCE, mapper.pose(1))
        self.assertVector(
            rt.unit(rt.sub(first["lowerarm_l"], first["upperarm_l"])),
            (0, -math.sqrt(0.75), -0.5),
        )
        self.assertVector(
            rt.unit(rt.sub(after["lowerarm_l"], after["upperarm_l"])), (0, 0, 1)
        )

    def test_rejects_incomplete_or_nonfinite_frames(self):
        text = generated_clip()
        header, *rows = text.split("Frame Time: 0.033333\n")
        values = rows[0].splitlines()[0].split()
        values[0] = "nan"
        invalid_frame = (
            header
            + "Frame Time: 0.033333\n"
            + " ".join(values)
            + "\n"
            + rows[0].splitlines()[1]
        )
        for bad in (text.replace("Frames: 2", "Frames: 3"), text[:-10], invalid_frame):
            with self.assertRaises(ValueError):
                rt.parse_bvh(bad)


if __name__ == "__main__":
    unittest.main()
