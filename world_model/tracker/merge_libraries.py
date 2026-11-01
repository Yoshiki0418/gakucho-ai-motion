"""Append extra clips (e.g. cleaned T2M training references) to a motion library.

The tracker trains on one library directory. The result keeps the base
library's clips, seats and summary, and appends the extra library's clips with
their frame offsets moved past the base frames. Clip paths must be unique.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .motion_library import ARRAYS
    base = Path(args.base)
    manifest = json.loads((base / "manifest.json").read_text())
    data = np.load(base / "library.npz")
    arrays = {k: [data[k]] for k in ARRAYS}
    offset = int(arrays["qpos"][0].shape[0])
    paths = {c["path"] for c in manifest["clips"]}
    added = []
    for extra in map(Path, args.extra):
        extra_manifest = json.loads((extra / "manifest.json").read_text())
        extra_data = np.load(extra / "library.npz")
        for key in ARRAYS:
            arrays[key].append(extra_data[key])
        for clip in extra_manifest["clips"]:
            if clip["path"] in paths:
                raise SystemExit(f"duplicate clip path {clip['path']}")
            paths.add(clip["path"])
            if clip.get("status") == "ok":
                clip["start"] += offset
            manifest["clips"].append(clip)
        offset += int(extra_data["qpos"].shape[0])
        added.append({"library": str(extra), "clips": len(extra_manifest["clips"])})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    # seats of the base and of every extra library (clip paths are unique, checked above)
    seats = json.loads((base / "seats.json").read_text()) if (base / "seats.json").exists() else {"seats": {}}
    for extra in map(Path, args.extra):
        if (extra / "seats.json").exists():
            seats["seats"].update(json.loads((extra / "seats.json").read_text())["seats"])
    (out / "seats.json").write_text(json.dumps(seats, indent=1))
    manifest["summary"]["merged"] = {"base": str(base), "extra": added, "frames": offset}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest["summary"]["merged"], indent=1))


if __name__ == "__main__":
    main()
