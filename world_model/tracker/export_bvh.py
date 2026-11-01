"""Write a simulated tracking trajectory back to the source BVH skeleton.

The result uses the hierarchy of the clip that was tracked, so it plays in
the existing BVH viewer on the gakucho GLB. Root and the 21 simulated
segments come from physics; fingers, which the body does not simulate, keep
the source clip's rotations. A merged segment's rotation is written to its
first BVH joint and identity to the others.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..references import _parse_bvh
from .body import ACTUATED, DISPLAY_TO_PHYSICS as C


def _hinge_rotation(angles, order):
    return Rotation.from_euler(order, angles)


def physics_to_bvh(source_text, qpos, *, root_scale, ground_shift, hz=50):
    bvh = _parse_bvh(source_text, target_height=1.8)
    names = [bone.name for bone in bvh.bones]
    T = len(qpos)
    source_index = np.clip(np.round(np.arange(T) / hz / bvh.frame_time).astype(int), 0, len(bvh.frames) - 1)
    frames = bvh.frames[source_index].copy()

    def write_rotation(joint, rotation):
        bone = bvh.bones[names.index(joint)]
        rot_channels = [c for c in bone.channels if c.endswith("rotation")]
        seq = "".join(c[0] for c in rot_channels)
        values = rotation.as_euler(seq, degrees=True)
        for k, channel in enumerate(rot_channels):
            frames[:, bone.channel_start + bone.channels.index(channel)] = values[:, k]

    # root
    root = bvh.bones[names.index("Hips")]
    position = qpos[:, :3].astype(np.float64).copy()
    position[:, 2] -= ground_shift
    source_position = (C.T @ position.T).T / root_scale / bvh.scale
    for axis, channel in enumerate(("Xposition", "Yposition", "Zposition")):
        if channel in root.channels:
            offset = root.offset[axis]
            frames[:, root.channel_start + root.channels.index(channel)] = source_position[:, axis] - offset
    root_matrix = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).as_matrix()
    write_rotation("Hips", Rotation.from_matrix(C.T @ root_matrix @ C))

    column = 7
    for seg in ACTUATED:
        angles = qpos[:, column:column + 3]
        column += 3
        matrix = _hinge_rotation(angles, seg.order).as_matrix()
        write_rotation(seg.bvh[0], Rotation.from_matrix(C.T @ matrix @ C))
        for joint in seg.bvh[1:]:
            write_rotation(joint, Rotation.identity(T))

    header = source_text.split("MOTION", 1)[0].replace("\r\n", "\n")
    lines = [header.rstrip("\n"), "MOTION", f"Frames: {T}", f"Frame Time: {1. / hz:.6f}"]
    lines += [" ".join(f"{value:.6f}" for value in row) for row in frames]
    return "\n".join(lines) + "\n"


def _source_bvh(library, clip):
    from pathlib import Path
    from .motion_library import DEFAULT_SOURCE
    path = clip["path"].split("#")[0]
    if path.startswith("t2m/"):
        return (Path(library) / "bvh" / path.split("/", 1)[1]).read_text()
    return (DEFAULT_SOURCE / path).read_text()


def main():
    """Convert a saved simulated (or reference) qpos trajectory to BVH."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="library whose clip supplies the skeleton")
    parser.add_argument("--clip", required=True, help="clip path in the library manifest")
    parser.add_argument("--trajectory", required=True, help="npz with qpos (and reference)")
    parser.add_argument("--key", default="qpos", choices=("qpos", "reference"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(args.library) / "manifest.json").read_text())
    clip = next(c for c in manifest["clips"] if c["path"] == args.clip)
    source = next(c for c in manifest["clips"] if c["path"] == clip["path"].split("#")[0])
    qpos = np.load(args.trajectory)[args.key]
    text = physics_to_bvh(_source_bvh(args.library, clip), qpos,
                          root_scale=source["rootScale"], ground_shift=source["groundShift"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(args.output, len(qpos), "frames")


if __name__ == "__main__":
    main()
