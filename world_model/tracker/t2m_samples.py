"""Generate motions with the running Japanese T2M service and retarget them.

The service is only asked to generate (one request at a time); its loaded
model and settings are not changed. Generated BVHs are copied with their
response metadata and retargeted into a library directory with the same
layout as ``motion_library`` (split "test"), so the tracker evaluators can
run on them unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

SERVICE = "http://127.0.0.1:8100/api/realtime/generate"
RESULTS_ROOT = Path("/gpu-server/yamamoto/Custom-T2M-Lab")

# Standing whole-body prompts from the dataset vocabulary (no chair needed),
# plus a few free phrasings that are not verbatim training labels.
PROMPTS = [
    "立っている状態から、歩く。", "立っている状態から、走る。", "立っている状態から、ジャンプする。",
    "立っている状態から、しゃがむ。", "立っている状態から、スクワットをする。", "立っている状態から、手を振る。",
    "立っている状態から、拍手する。", "立っている状態から、お辞儀する。", "立っている状態から、敬礼する。",
    "立っている状態から、伸びをする。", "立っている状態から、うなずく。", "立っている状態から、首を横に振る。",
    "立っている状態から、見回す。", "立っている状態から、指をさす。", "立っている状態から、電話をかける。",
    "立っている状態から、頭を触る。", "立っている状態から、腰に手をあてる。", "立っている状態から、祈る。",
    "立っている状態から、片膝をつく。", "立っている状態から、ほうきで掃く。", "立っている状態から、スキップする。",
    "立っている状態から、（ボールなしで）蹴る。", "立っている状態から、（ボールなしで）投げる。",
    "立っている状態から、バットを振る。", "立っている状態から、ゴルフクラブを振る。",
    "立っている状態から、小さい段ボールを持ち上げる。", "立っている状態から、大きい物を渡す。",
    "立っている状態から、ハイタッチする。", "立っている状態から、握手する。", "立っている状態から、うちわであおぐ。",
    "前向きに歩く", "後ろ向きに歩く", "ポケットに手を入れたまま歩く。", "スマホを見ながら歩く。",
    "ゆっくり歩いて止まる。", "少し歩いてから振り返る。", "両手を上げて喜ぶ。", "腕を組んで考える。",
    "手を振りながら歩く。", "深くお辞儀をしてから顔を上げる。",
]


def generate(text, seed, **sampling):
    body = json.dumps({"text": text, "seed": seed, **sampling}).encode()
    request = urllib.request.Request(SERVICE, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    # Several candidates per prompt (e.g. "1-10"): the tracker then picks one it can execute.
    parser.add_argument("--seeds", default=None, help="seed list such as 1-10 or 1,3,5 (overrides --seed)")
    # Training references for the tracker come from other prompts than the G2 evaluation (PROMPTS).
    parser.add_argument("--prompts-file", default=None, help="JSON list of prompt texts instead of PROMPTS")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    seeds = [args.seed]
    if args.seeds:
        seeds = []
        for part in args.seeds.split(","):
            low, _, high = part.partition("-")
            seeds += list(range(int(low), int(high or low) + 1))
    sampling = {k: v for k, v in (("temperature", args.temperature), ("top_k", args.top_k)) if v is not None}
    from .body import build_model, load_profile
    from .motion_library import ARRAYS
    from .retarget import retarget_bvh
    output = Path(args.output)
    (output / "bvh").mkdir(parents=True, exist_ok=False)
    profile, model = load_profile(), build_model()
    clips, arrays, offset = [], {k: [] for k in ARRAYS}, 0
    prompts = json.loads(Path(args.prompts_file).read_text()) if args.prompts_file else PROMPTS
    if args.prompts_file and set(prompts) & set(PROMPTS):
        raise SystemExit("training prompts overlap the G2 evaluation prompts")
    jobs = [(index, text, seed) for index, text in enumerate(prompts) for seed in seeds]
    for index, text, seed in jobs:
        start = time.time()
        response = generate(text, seed, **sampling)
        source = RESULTS_ROOT / response["bvh_path"]
        name = f"t2m-{index:03d}.bvh" if args.seeds is None else f"t2m-{index:03d}-s{seed:02d}.bvh"
        shutil.copy(source, output / "bvh" / name)
        raw = (output / "bvh" / name).read_bytes()
        result = retarget_bvh(raw.decode("utf-8"), model, profile)
        meta = result.pop("meta")
        clip = {"path": f"t2m/{name}", "dataset": "t2m", "version": response["model_version"],
                "subject": "generated", "take": str(seed), "label": f"{index:03d}", "text": text,
                "split": args.split, "needsSeat": "椅子" in text, "sha256": hashlib.sha256(raw).hexdigest(),
                "status": "ok", "start": offset, "length": meta["frames"],
                "t2m": {k: response.get(k) for k in ("model_version", "checkpoint_id", "seed", "duration_seconds",
                                                     "token_count", "quality_flags", "cfg_scale", "temperature")},
                **{k: v for k, v in meta.items() if k != "segments"}}
        offset += meta["frames"]
        for key in ARRAYS:
            arrays[key].append(result[key])
        clips.append(clip)
        print(f"{index:02d} s{seed} {text} {meta['frames'] / 50:.1f}s ({time.time() - start:.1f}s)", flush=True)
    np.savez(output / "library.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    summary = {"version": 1, "source": SERVICE, "hz": 50, "clips": len(clips), "usable": len(clips),
               "frames": offset, "seeds": seeds, "sampling": sampling or "service defaults"}
    (output / "manifest.json").write_text(json.dumps({"summary": summary, "clips": clips},
                                                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
