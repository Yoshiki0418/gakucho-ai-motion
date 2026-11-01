"""Kinematic clean-up of generated motions before tracking (G2).

Besides crouching part-way through, the T2M output differs from capture in
ways no tracker can follow: joint jitter (99th-percentile joint speed 20 rad/s
against 9 in the mocap), feet sliding over the floor (2.7 m/s against 0.04 at
the 90th percentile) and feet sinking below the floor. As in animation
clean-up, only the reference is changed:

1. low-pass the hinge angles, the root path and the root rotation
   (Gaussian, ``--sigma`` frames; 2 frames = 40 ms)
2. keep the stance foot planted: while the lower foot is near the floor, the
   root moves by minus that foot's horizontal slip (the legs' own motion then
   carries the body, as in a captured walk)
3. lift the root on frames where a foot or toe is below its standing height

Crouches are left as generated (``select_t2m`` rejects unintended ones).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

FOOT_REST, TOE_REST = .121, .069   # body-origin heights when standing (mocap, first frames)
STANCE = .03                       # m above rest height that still counts as on the floor


def clean(model, qpos, sigma=2., root_sigma=None):
    from . import body as B
    from .path_reference import kinematics
    names = [getattr(s, "name", s) for s in B.SEGMENTS]
    feet = [names.index("left_foot"), names.index("right_foot")]
    toes = [names.index("left_toe"), names.index("right_toe")]
    q = qpos.astype(np.float64).copy()
    root_sigma = sigma if root_sigma is None else root_sigma
    if root_sigma > 0:
        # T2M lunges, twists and tilts the root faster than any capture (failures follow them)
        q[:, :3] = gaussian_filter1d(q[:, :3], root_sigma, axis=0, mode="nearest")
        quat = q[:, 3:7].copy()
        quat[np.cumsum(np.r_[0, np.sign(np.sum(quat[1:] * quat[:-1], axis=1)) < 0]) % 2 == 1] *= -1
        quat = gaussian_filter1d(quat, root_sigma, axis=0, mode="nearest")
        q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    if sigma > 0:
        q[:, 7:] = gaussian_filter1d(q[:, 7:], sigma, axis=0, mode="nearest")
    xpos = kinematics(model, q)["xpos"]
    foot_z = xpos[:, feet, 2]
    stance = np.argmin(foot_z, axis=1)
    planted = foot_z[np.arange(len(q)), stance] < FOOT_REST + STANCE
    correction = np.zeros((len(q), 2))
    for t in range(len(q) - 1):
        slip = np.zeros(2)
        if planted[t] and planted[t + 1] and stance[t] == stance[t + 1]:
            slip = xpos[t + 1, feet[stance[t]], :2] - xpos[t, feet[stance[t]], :2]
        correction[t + 1] = correction[t] - slip
    q[:, :2] += correction
    xpos = kinematics(model, q)["xpos"]
    sink = np.maximum(FOOT_REST - xpos[:, feet, 2].min(axis=1), TOE_REST - xpos[:, toes, 2].min(axis=1))
    q[:, 2] += gaussian_filter1d(np.maximum(sink, 0.), max(sigma, 1.), mode="nearest")
    return kinematics(model, q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="t2m_samples output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sigma", type=float, default=2.)
    parser.add_argument("--root-sigma", type=float, default=None, help="root smoothing (frames); default --sigma")
    args = parser.parse_args()
    from . import body as B
    from .motion_library import ARRAYS
    model = B.build_model()
    source = Path(args.library)
    manifest = json.loads((source / "manifest.json").read_text())
    raw = np.load(source / "library.npz")
    qpos = raw["qpos"]
    arrays = {k: [] for k in ARRAYS}
    for clip in manifest["clips"]:
        result = clean(model, qpos[clip["start"]:clip["start"] + clip["length"]], args.sigma, args.root_sigma)
        for key in ARRAYS:
            arrays[key].append(result[key])
        clip["cleaned"] = {"sigmaFrames": args.sigma, "rootSigmaFrames": args.root_sigma or args.sigma, "stanceFootLocked": True, "grounded": True}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    manifest["summary"]["cleanedFrom"] = str(source)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"cleaned {len(manifest['clips'])} clips -> {out}")


if __name__ == "__main__":
    main()
