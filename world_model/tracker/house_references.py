"""Training references for the house tour (v10): the legs that the G6 evaluation chains.

Stage-3 houses with TRAINING seeds (below 1,000,000; the evaluation uses 1,000,000+) get a
random task (2-4 rooms in random order, then the bedroom chair in 70% of them), a random
captured walk and a random captured sit. ``house_tour.build_tour`` composes the same
reference the evaluation tracks, and each leg becomes one library clip:

* walk legs: turn on the spot -> walk bent onto the A* route -> stand (``家の中で向きを変えて歩く``)
* the sit leg: walk -> turn round -> step back -> sit -> stay seated (``椅子まで歩いて振り向き、座る``),
  ``needsSeat`` with the chair seat as its seat, so training puts the seat box there.

Output has the motion-library layout (manifest.json, library.npz, seats.json; split "train")
and is appended to a library with ``merge_libraries``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np

WALK_TEXT = "家の中で向きを変えて歩く"
SIT_TEXT = "椅子まで歩いて振り向き、座る"


def _house(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from . import body as B
    from ..scenes import generate_scene
    from .depth_map import Grid
    from .home_navigation import _turn_clip, rasterise
    from .house_tour import _sit_clips, back_steps, build_tour
    from .navigation_eval import _walk_bank
    from .path_reference import kinematics
    seed, library, seat_library = args
    rng = random.Random(seed)
    scene = generate_scene(3, seed, rng.uniform(0., 1.))
    rooms = ["work-room", "kitchen", "bed-room", "living-room"]
    rng.shuffle(rooms)
    task = rooms[:rng.randint(2, 4)] + (["bedroom-chair-seat"] if rng.random() < .7 else [])
    scene["metadata"] = dict(scene["metadata"], taskSequence=task)
    bounds = scene["metadata"]["floorBounds"]
    grid = Grid(np.asarray(bounds[:2], float), np.asarray(bounds[2:], float), pad=.2)
    rasterise(grid, scene)
    bank = _walk_bank(library)
    walk_name, walk = bank[rng.randrange(len(bank))]
    sits = _sit_clips(seat_library)
    sit = sits[rng.randrange(len(sits))]
    model = B.build_model()
    try:
        reference, legs = build_tour(scene, walk, _turn_clip(library), sit, grid, .4, model, back_steps(library))
    except ValueError:
        return []
    chair = next(g for g in scene["geoms"] if g["id"] == "bedroom-chair/seat")
    seat_goal = next(g for g in scene["goals"] if g["kind"] == "sit")
    clips = []
    for index, leg in enumerate(legs):
        qpos = reference[leg["start"]:leg["end"] + 1]
        clip = {"path": f"house/{seed:06d}-{index}-{leg['goal']}", "dataset": "house", "label": leg["goal"],
                "text": SIT_TEXT if leg["kind"] == "sit" else WALK_TEXT, "split": "train",
                "needsSeat": leg["kind"] == "sit", "status": "ok", "sceneSeed": seed, "walk": walk_name,
                "sitClip": sit[0] if leg["kind"] == "sit" else None}
        seat = None
        if leg["kind"] == "sit":
            seat = {"x": float(seat_goal["position"][0]), "y": float(seat_goal["position"][1]),
                    "top": float(seat_goal["position"][2]), "yaw": float(chair.get("yaw", 0.)), "seatedFrames": 100}
        clips.append((clip, kinematics(model, qpos), seat))
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="walk bank and turn/backward clips (motion-library-v1)")
    parser.add_argument("--seat-library", required=True, help="sit clips with seats.json (motion-library-v3)")
    parser.add_argument("--houses", type=int, default=150)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed_base + args.houses > 1_000_000:
        raise SystemExit("training houses must use seeds below the evaluation range (1,000,000+)")
    from .motion_library import ARRAYS
    jobs = [(args.seed_base + i, args.library, args.seat_library) for i in range(args.houses)]
    with get_context("spawn").Pool(args.workers) as pool:
        houses = pool.map(_house, jobs, chunksize=1)
    manifest, arrays, seats, offset = [], {k: [] for k in ARRAYS}, {}, 0
    for clips in houses:
        for clip, data, seat in clips:
            clip["start"], clip["length"] = offset, int(len(data["qpos"]))
            offset += clip["length"]
            for key in ARRAYS:
                arrays[key].append(data[key])
            manifest.append(clip)
            if seat:
                seats[clip["path"]] = seat
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    summary = {"version": 1, "source": "house_tour.build_tour on stage-3 training houses", "hz": 50,
               "clips": len(manifest), "usable": len(manifest), "frames": offset,
               "houses": args.houses, "seedBase": args.seed_base,
               "walkLegs": sum(not c["needsSeat"] for c in manifest), "sitLegs": len(seats)}
    (out / "manifest.json").write_text(json.dumps({"summary": summary, "clips": manifest}, ensure_ascii=False, indent=1))
    (out / "seats.json").write_text(json.dumps({"summary": {"seats": len(seats)}, "seats": seats}, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
