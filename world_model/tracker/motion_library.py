"""Build the retargeted motion library from the Custom-T2M-Lab mocap BVH tree.

Layout of the source tree: {dataset}/{version}/{subject}/{take}/{label}.bvh
with Japanese descriptions in {dataset}/{version}/texts/{label}.txt.

Output: one ``library.npz`` with all frames concatenated plus ``manifest.json``
holding per-clip provenance (path, SHA256, text, split, diagnostics). The
held-out split is a deterministic 10% per (dataset, version, label) chosen by
path hash, so every action class keeps unseen takes for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = Path("/gpu-server/yamamoto/Custom-T2M-Lab/data/BVH")
ARRAYS = ("qpos", "qvel", "xpos", "xquat", "linvel", "angvel")
MAX_SEGMENT_SPEED = 25.  # rad/s geodesic joint rotation speed; faster is a capture glitch
MAX_HINGE_SPEED = 40.    # rad/s per hinge; catches residual near-singular Euler motion
MIN_FRAMES = 25


def _text(path):
    try:
        return path.read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
    except (OSError, UnicodeDecodeError, IndexError):
        return ""


def scan(source):
    clips = []
    for bvh in sorted(Path(source).rglob("*.bvh")):
        rel = bvh.relative_to(source)
        parts = rel.parts
        if len(parts) != 5:
            continue
        dataset, version, subject, take, name = parts
        label = Path(name).stem
        text = _text(Path(source) / dataset / version / "texts" / f"{label}.txt")
        digest = int(hashlib.sha256(str(rel).encode()).hexdigest(), 16)
        clips.append({"path": str(rel), "dataset": dataset, "version": version, "subject": subject,
                      "take": take, "label": label, "text": text,
                      "split": "test" if digest % 10 == 0 else "train",
                      "needsSeat": "椅子" in text})
    return clips


def _work(args):
    source, clip = args
    from .body import build_model, load_profile
    from .retarget import retarget_bvh
    global _MODEL, _PROFILE
    if "_MODEL" not in globals():
        _PROFILE = load_profile()
        _MODEL = build_model(_PROFILE)
    raw = (Path(source) / clip["path"]).read_bytes()
    clip = dict(clip, sha256=hashlib.sha256(raw).hexdigest())
    try:
        result = retarget_bvh(raw.decode("utf-8"), _MODEL, _PROFILE)
    except Exception as error:  # keep going; the manifest records the failure
        return dict(clip, status=f"error: {error}"), None
    meta = result.pop("meta")
    status = "ok"
    if meta["frames"] < MIN_FRAMES:
        status = "too-short"
    elif meta["maxSegmentAngularSpeed"] > MAX_SEGMENT_SPEED:
        status = "capture-glitch"
    elif meta["maxHingeSpeed"] > MAX_HINGE_SPEED:
        status = "hinge-speed"
    return dict(clip, status=status, **{k: v for k, v in meta.items() if k != "segments"}), result


def build(source, output, workers=24):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    clips = scan(source)
    start = time.time()
    with get_context("spawn").Pool(workers) as pool:
        results = pool.map(_work, [(str(source), c) for c in clips], chunksize=8)
    from .seats import needs_seat
    manifest, arrays, offset = [], {k: [] for k in ARRAYS}, 0
    for meta, data in results:
        if data is not None and meta["status"] == "ok":
            meta["needsSeat"] = needs_seat(meta["text"], data["xpos"])
            meta["start"], meta["length"] = offset, int(data["qpos"].shape[0])
            offset += meta["length"]
            for key in ARRAYS:
                arrays[key].append(data[key])
        manifest.append(meta)
    from .body import build_xml, profile_sha256
    np.savez(output / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    usable = [m for m in manifest if m["status"] == "ok"]
    summary = {"version": 1, "source": str(source), "hz": 50, "clips": len(manifest),
               "usable": len(usable), "frames": offset,
               "statusCounts": {s: sum(m["status"] == s for m in manifest) for s in {m["status"] for m in manifest}},
               "trainUsable": sum(m["split"] == "train" for m in usable),
               "testUsable": sum(m["split"] == "test" for m in usable),
               "needsSeatUsable": sum(m["needsSeat"] for m in usable),
               "profileSha256": profile_sha256(),
               "bodyXmlSha256": hashlib.sha256(build_xml().encode()).hexdigest(),
               "elapsedSeconds": time.time() - start}
    (output / "manifest.json").write_text(json.dumps({"summary": summary, "clips": manifest},
                                                     ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    data = np.load(directory / "library.npz")
    return manifest, {k: data[k] for k in ARRAYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    build(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
