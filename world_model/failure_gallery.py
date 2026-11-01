"""Add explicitly excluded historical failures without rerunning scored evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .avatar_profile import avatar_profile_sha256


def register_failure_examples(index_path, output):
    index_path, output = Path(index_path), Path(output)
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_profile = manifest["avatarProfileSha256"]
    walking_model = manifest["trainingModels"]["walking"]
    expected_body = walking_model["bodyModelSha256"]
    selected_checkpoint = walking_model["checkpointSha256"]
    if index.get("version") != 1 or not index.get("examples"):
        raise ValueError("No versioned failure examples were supplied")
    if (index.get("avatarProfileSha256") != expected_profile
            or index.get("bodyModelSha256") != expected_body):
        raise ValueError("Failure index belongs to a different avatar body")

    pending, records, names = [], [], set()
    for entry in index["examples"]:
        filename = entry["filename"]
        if (Path(filename).name != filename or not filename.startswith("stage1-walking-fall-")
                or not filename.endswith(".json") or filename in names):
            raise ValueError("Invalid or duplicate failure trace filename")
        names.add(filename)
        source = index_path.parent / filename
        trace_bytes = source.read_bytes()
        trace_sha = hashlib.sha256(trace_bytes).hexdigest()
        if entry.get("traceSha256", trace_sha) != trace_sha:
            raise ValueError("Failure trace SHA256 differs from its index")
        trace = json.loads(trace_bytes)
        failure = trace.get("failureExample", {})
        if (trace.get("avatarProfileSha256") != expected_profile
                or avatar_profile_sha256(trace.get("avatarProfile")) != expected_profile
                or trace.get("bodyModelSha256") != expected_body):
            raise ValueError("Failure trace belongs to a different avatar body")
        if (failure.get("evaluationExcluded") is not True
                or failure.get("checkpointSha256") == selected_checkpoint
                or not failure.get("checkpointSha256")):
            raise ValueError("Historical failures must be excluded and identify their earlier checkpoint")
        for key in ("seed", "checkpointSha256", "fallTimeSeconds", "settlingSeconds"):
            if failure.get(key) != entry.get(key):
                raise ValueError(f"Failure index and trace disagree about {key}")
        frames = trace.get("frames", [])
        times = [frame.get("time") for frame in frames]
        if (trace.get("version") != 1 or trace.get("geometryScope") != "complete-world"
                or not trace.get("physicsRestBodies") or len(frames) < 2
                or any(not isinstance(time, (int, float)) or not math.isfinite(time) for time in times)
                or any(b <= a for a, b in zip(times, times[1:]))
                or any(not frame.get("bodies") or not frame.get("geoms") for frame in frames)):
            raise ValueError("Failure trace lacks a complete physical timeline")
        fall_time = failure["fallTimeSeconds"]
        fall_frame = min(frames, key=lambda frame: abs(frame["time"] - fall_time))
        if (not times[0] < fall_time < times[-1]
                or abs(fall_frame["time"] - fall_time) > trace.get("dt", .02) + 1e-6
                or fall_frame.get("stats", {}).get("fall") is not True):
            raise ValueError("Failure trace does not record the declared physical fall")
        target = output / filename
        if target.exists() and target.read_bytes() != trace_bytes:
            raise ValueError("An existing failure trace cannot be silently replaced")
        pending.append((target, trace_bytes))
        records.append({**entry, "url": f"/world-model/{filename}",
                        "traceSha256": trace_sha,
                        "evaluationExcluded": True})

    # Scored policy, gates, and metrics are preserved verbatim. These traces are a gallery.
    result = deepcopy(manifest)
    stage = next(stage for stage in result["stages"] if stage["stage"] == 1)
    urls = {entry["url"] for entry in records}
    stage["traceOptions"] = [entry for entry in stage.get("traceOptions", []) if entry["url"] not in urls]
    stage["traceOptions"] += [{"title": entry["title"], "url": entry["url"]} for entry in records]
    result["failureExamples"] = {"evaluationExcluded": True,
        "registeredAtUtc": datetime.now(timezone.utc).isoformat(),
        "indexSha256": hashlib.sha256(index_bytes).hexdigest(),
        "registrationSourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "examples": records}
    for target, contents in pending:
        if not target.exists():
            temporary = target.with_suffix(".json.pending")
            temporary.write_bytes(contents)
            temporary.replace(target)
    temporary = manifest_path.with_suffix(".json.pending")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, help="Historical failure index JSON")
    parser.add_argument("--output", default="frontend/public/world-model")
    args = parser.parse_args()
    result = register_failure_examples(args.examples, args.output)
    print(json.dumps({"examples": len(result["failureExamples"]["examples"]),
                      "evaluationExcluded": True, "gates": result["gates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
