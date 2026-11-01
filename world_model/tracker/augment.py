"""Add bent copies of locomotion clips to a motion library.

The captured walks are almost straight, while room navigation needs curved
walking. Each locomotion clip gets ``--variants`` copies bent with a random
smooth curvature profile (``path_reference.random_bend``). A copy keeps the
split of its source clip, so held-out takes stay held out.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .body import build_model
from .motion_library import ARRAYS
from .path_reference import kinematics, random_bend

LOCOMOTION_WORDS = ("歩", "走", "スキップ", "下がる")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--max-curvature", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    raw = np.load(source / "library.npz")
    arrays = {k: [raw[k]] for k in ARRAYS}
    offset = int(raw["qpos"].shape[0])
    model, rng = build_model(), np.random.default_rng(args.seed)
    added = []
    for clip in manifest["clips"]:
        if clip["status"] != "ok" or clip["needsSeat"] or not any(w in clip["text"] for w in LOCOMOTION_WORDS):
            continue
        qpos = raw["qpos"][clip["start"]:clip["start"] + clip["length"]].astype(np.float64)
        for v in range(args.variants):
            bent, curvature = random_bend(qpos, rng, max_curvature=args.max_curvature)
            data = kinematics(model, bent)
            entry = {k: clip[k] for k in ("dataset", "version", "subject", "take", "label", "text",
                                          "split", "needsSeat", "sha256")}
            entry.update(path=f'{clip["path"]}#bend{v}', status="ok", start=offset,
                         length=int(len(bent)), augmentation={"source": clip["path"], "kind": "bend",
                                                              "maxAbsCurvature": curvature})
            offset += len(bent)
            for k in ARRAYS:
                arrays[k].append(data[k])
            added.append(entry)
    np.savez(output / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    clips = manifest["clips"] + added
    summary = dict(manifest["summary"], clips=len(clips), frames=offset,
                   usable=manifest["summary"]["usable"] + len(added),
                   augmentation={"source": str(source), "bentClips": len(added),
                                 "variantsPerClip": args.variants, "maxCurvature": args.max_curvature,
                                 "seed": args.seed})
    (output / "manifest.json").write_text(json.dumps({"summary": summary, "clips": clips},
                                                     ensure_ascii=False, indent=1))
    if (source / "seats.json").exists():
        shutil.copy(source / "seats.json", output / "seats.json")
    print(json.dumps(summary["augmentation"], ensure_ascii=False), len(added))


if __name__ == "__main__":
    main()
