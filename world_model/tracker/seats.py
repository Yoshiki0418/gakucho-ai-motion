"""Estimate a seat box for clips that need a chair.

The mocap has no chair geometry. For a clip marked ``needsSeat`` the seated
frames are those whose pelvis is within 5 cm of its lowest height; the seat
top is placed at the pelvis capsule's lowest point there, centred slightly in
front of the pelvis (so the thighs rest on it) and aligned with the pelvis
heading. Clips without a seat get none. The seat is static during the clip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .body import load_profile

SEAT_HALF = (.22, .22, .5)   # box half extents; the box reaches below the floor
SEAT_FORWARD = .07
# Chair-height pelvis range. Sitting on the floor or lying keeps the pelvis at
# 0.1-0.3 m; on a chair it rests at 0.5-0.7 m; standing is about 1 m.
SEATED_PELVIS = (.40, .80)
SEATED_BEHIND_FEET = .22   # m; a squat keeps the pelvis over the feet
SEATED_HEAD = .90          # m; a push-up also has the pelvis at chair height
SEATED_SHARE = .9          # of the frames


def seated_share(xpos):
    """Share of frames in a sitting-on-a-chair posture (body positions (T, nb, 3))."""
    from .body import SEGMENTS
    names = [getattr(s, "name", s) for s in SEGMENTS]
    pelvis, head = xpos[:, names.index("pelvis")], xpos[:, names.index("head")]
    feet = (xpos[:, names.index("left_foot"), :2] + xpos[:, names.index("right_foot"), :2]) / 2
    behind = np.linalg.norm(pelvis[:, :2] - feet, axis=1)
    seated = ((pelvis[:, 2] >= SEATED_PELVIS[0]) & (pelvis[:, 2] < SEATED_PELVIS[1])
              & (behind > SEATED_BEHIND_FEET) & (head[:, 2] > SEATED_HEAD))
    return float(seated.mean())


def needs_seat(text, xpos):
    """Whether a clip needs a chair (``xpos``: reference body positions (T, nb, 3)).

    The first library used only the word 椅子. Some clips were captured on a
    chair without saying so and fail within a second without a seat:
    「座っている状態から、見回す。」 (the pelvis stays at 0.54-0.70 m) and desk
    tasks such as 「合掌(いただきますとか)」 or 「消しゴムで文字を消す」. A clip
    also needs one when it mentions sitting (座っ) and its pelvis never drops
    below chair height, or when it keeps a chair posture almost throughout.
    Floor sitting (床) never gets one.
    """
    if "椅子" in text:
        return True
    if "床" in text:
        return False
    if seated_share(xpos) >= SEATED_SHARE:
        return True
    low = float(np.min(xpos[:, 0, 2]))
    return "座っ" in text and SEATED_PELVIS[0] <= low < SEATED_PELVIS[1]


def estimate(xpos, xquat, profile):
    pelvis = profile["physical"]["pelvis"]
    radius, centre_z = float(pelvis["radius"]), float(pelvis["centerOffset"][2])
    z = xpos[:, 0, 2]
    seated = z <= z.min() + .05
    w, x, y, q = xquat[seated, 0].T
    heading = np.arctan2(2 * (x * y + w * q), 1 - 2 * (y * y + q * q))
    yaw = float(np.arctan2(np.sin(heading).mean(), np.cos(heading).mean()))
    forward = np.asarray([np.sin(yaw), -np.cos(yaw)])
    centre = xpos[seated, 0, :2].mean(axis=0) + SEAT_FORWARD * forward
    top = float(np.median(z[seated] + centre_z - radius))
    return {"x": float(centre[0]), "y": float(centre[1]), "top": top, "yaw": yaw,
            "seatedFrames": int(seated.sum())}


def relabel(source, output):
    """Copy a library with ``needsSeat`` recomputed by :func:`needs_seat`.

    The motion arrays are shared through a symlink (the clips do not change),
    so earlier runs that used ``source`` stay reproducible.
    """
    source, output = Path(source).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    xpos = np.load(source / "library.npz")["xpos"]
    changed = []
    for clip in manifest["clips"]:
        if clip["status"] != "ok":
            continue
        seat = needs_seat(clip["text"], xpos[clip["start"]:clip["start"] + clip["length"]])
        if seat != clip["needsSeat"]:
            changed.append({"path": clip["path"], "text": clip["text"], "split": clip["split"],
                            "needsSeat": seat})
        clip["needsSeat"] = seat
    usable = [c for c in manifest["clips"] if c["status"] == "ok"]
    manifest["summary"]["needsSeatUsable"] = sum(c["needsSeat"] for c in usable)
    manifest["summary"]["relabel"] = {"source": str(source), "rule": "seats.needs_seat",
                                      "seatedPelvis": SEATED_PELVIS, "seatedBehindFeet": SEATED_BEHIND_FEET,
                                      "seatedHead": SEATED_HEAD, "seatedShare": SEATED_SHARE,
                                      "changed": len(changed)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (output / "relabel.json").write_text(json.dumps(changed, ensure_ascii=False, indent=1))
    (output / "library.npz").symlink_to(source / "library.npz")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--relabel-from", default=None,
                        help="create --library from this library with needsSeat recomputed")
    args = parser.parse_args()
    library = Path(args.library)
    if args.relabel_from:
        changed = relabel(args.relabel_from, library)
        print(f"needsSeat changed for {len(changed)} clips")
    manifest = json.loads((library / "manifest.json").read_text())
    raw = np.load(library / "library.npz")
    data = {"xpos": raw["xpos"], "xquat": raw["xquat"]}   # npz members reload on every access
    profile = load_profile()
    seats = {}
    for clip in manifest["clips"]:
        if clip["status"] != "ok" or not clip["needsSeat"]:
            continue
        rows = slice(clip["start"], clip["start"] + clip["length"])
        seats[clip["path"]] = estimate(data["xpos"][rows], data["xquat"][rows], profile)
    tops = np.asarray([s["top"] for s in seats.values()])
    summary = {"seats": len(seats), "seatHalfExtents": SEAT_HALF, "forwardOffset": SEAT_FORWARD,
               "topP05": float(np.percentile(tops, 5)), "topMedian": float(np.median(tops)),
               "topP95": float(np.percentile(tops, 95))}
    (library / "seats.json").write_text(json.dumps({"summary": summary, "seats": seats}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
