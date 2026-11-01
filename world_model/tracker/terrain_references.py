"""Training references on steps and short stairs (v11): captured walks adapted by IK.

Stage-4 scenes with TRAINING seeds (below 1,000,000) and difficulty below 2/3: one low step
(6-12 cm) or three steps (9.5-11.5 cm, 38 cm treads) up to a landing. For each scene a random
captured walk goes up (spawn -> goal) and, in a second clip, down (goal -> spawn), adapted to
the terrain by ``terrain_reference.adapt``, then stands 1.5 s. Each clip records its terrain as
boxes in the shapes the training environment has (``mjx_tracking.TERRAIN_BOXES``):
platform 2.6 x 1.8, three treads 1.8 x 0.38, landing 3.2 x 1.6, each given by centre and top.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def _boxes(scene):
    """Terrain boxes of a stage-4 scene in the training environment's slots."""
    out = []
    for g in scene["geoms"]:
        kind = {"step": "platform", "stair": "tread", "landing": "landing"}.get(g.get("semantic"))
        if kind:
            out.append({"kind": kind, "x": float(g["position"][0]), "y": float(g["position"][1]),
                        "top": float(g["position"][2] + g["size"][2] / 2), "size": [float(v) for v in g["size"][:2]]})
    return out


def _scene_clips(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from . import body as B
    from ..scenes import generate_scene
    from .navigation_eval import _walk_bank
    from .path_reference import follow
    from .terrain_reference import adapt, terrain_height
    seed, library = args
    rng = random.Random(seed)
    scene = generate_scene(4, seed, rng.uniform(0., 2 / 3 - 1e-3))
    height = terrain_height(scene)
    bank = _walk_bank(library)
    model = B.build_model()
    spawn = np.asarray(scene["spawn"][:2], float)
    goal = np.asarray(scene["goals"][-1]["position"][:2], float)
    clips = []
    for direction, (a, b) in (("up", (spawn, goal)), ("down", (goal, spawn))):
        walk_name, walk = bank[rng.randrange(len(bank))]
        # a little sideways spread so the feet do not always land on the same spots
        offset = np.asarray([rng.uniform(-.4, .4), 0.])
        route = np.stack([a + offset, b + offset])
        flat, _ = follow(walk, route, start_direction=float(np.arctan2(*(route[1] - route[0])[::-1])))
        data = adapt(model, flat, height)
        hold = 75
        data = {k: np.concatenate([v, np.repeat(v[-1:], hold, axis=0)]) for k, v in data.items()}
        data["qvel"][-hold:] = 0.
        data["linvel"][-hold:] = 0.
        data["angvel"][-hold:] = 0.
        clip = {"path": f"terrain/{seed:06d}-{direction}", "dataset": "terrain", "label": direction,
                "text": f"{scene['name']}を{'上る' if direction == 'up' else '下りる'}", "split": "train",
                "needsSeat": False, "status": "ok", "sceneSeed": seed, "walk": walk_name,
                "terrain": _boxes(scene), "rise": scene["metadata"].get("rise")}
        clips.append((clip, data))
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="walk bank (motion-library-v1)")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seed_base + args.scenes > 1_000_000:
        raise SystemExit("training scenes must use seeds below the evaluation range (1,000,000+)")
    from .motion_library import ARRAYS
    with get_context("spawn").Pool(args.workers) as pool:
        scenes = pool.map(_scene_clips, [(args.seed_base + i, args.library) for i in range(args.scenes)], chunksize=1)
    manifest, arrays, offset = [], {k: [] for k in ARRAYS}, 0
    for clips in scenes:
        for clip, data in clips:
            clip["start"], clip["length"] = offset, int(len(data["qpos"]))
            offset += clip["length"]
            for key in ARRAYS:
                arrays[key].append(data[key].astype(np.float32))
            manifest.append(clip)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    summary = {"version": 1, "source": "terrain_reference.adapt on stage-4 training scenes", "hz": 50,
               "clips": len(manifest), "usable": len(manifest), "frames": offset, "scenes": args.scenes,
               "byTerrain": {t: sum(c["text"].startswith(t) for c in manifest) for t in ("低い段差", "短い階段")}}
    (out / "manifest.json").write_text(json.dumps({"summary": summary, "clips": manifest}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
