"""Pick, for each prompt, one T2M candidate the tracker can execute (G2).

The T2M service drops most standing motions into a crouch part-way through
(pelvis 0.2-0.6 m; 38/40 with its default sampling), which no tracker should
follow. Several seeds per prompt are generated (``t2m_samples --seeds``), all
of them are played once on the GPU (``sweep_clips``), and per prompt the
candidate is chosen that

1. the tracker follows to the end (GPU sweep), and
2. keeps a standing pelvis unless the prompt asks for a low posture,

preferring the lowest mean body error. The choice is then re-checked by the
independent CPU evaluator (``evaluate_tracker --paths-from``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Prompts that legitimately bring the pelvis low (crouch, kneel, sit, lie, lift from the floor, jump take-off).
LOW_POSTURE_WORDS = ("しゃがむ", "スクワット", "座", "寝", "膝", "床", "拾", "腕立て", "腹筋", "這", "転",
                     "持ち上げ", "ジャンプ")
STANDING_PELVIS = .75   # m; standing is about 0.95 m


def unintended_crouch(text, xpos):
    if any(word in text for word in LOW_POSTURE_WORDS):
        return False
    return bool(np.min(xpos[:, 0, 2]) < STANDING_PELVIS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="t2m_samples output with several seeds per prompt")
    parser.add_argument("--sweep", required=True, help="sweep_clips output over that library")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(args.library) / "manifest.json").read_text())
    xpos = np.load(Path(args.library) / "library.npz")["xpos"]
    swept = {c["path"]: c for c in json.loads(Path(args.sweep).read_text())["clips"]}
    prompts = {}
    for clip in manifest["clips"]:
        result = swept[clip["path"]]
        crouch = unintended_crouch(clip["text"], xpos[clip["start"]:clip["start"] + clip["length"]])
        prompts.setdefault(clip["label"], {"text": clip["text"], "candidates": []})["candidates"].append(
            {"path": clip["path"], "seed": int(clip["take"]), "tracked": not result["failed"],
             "unintendedCrouch": crouch, "meanBodyErrorMeters": result["meanBodyErrorMeters"],
             "seconds": round(clip["length"] / 50, 2)})
    selected, rows = [], []
    for label, prompt in sorted(prompts.items()):
        good = [c for c in prompt["candidates"] if c["tracked"] and not c["unintendedCrouch"]]
        pick = min(good, key=lambda c: c["meanBodyErrorMeters"]) if good else None
        rows.append({"label": label, "text": prompt["text"], "selected": pick and pick["path"],
                     "tracked": sum(c["tracked"] for c in prompt["candidates"]),
                     "noCrouch": sum(not c["unintendedCrouch"] for c in prompt["candidates"]),
                     "both": len(good), "candidates": prompt["candidates"]})
        if pick:
            selected.append({"path": pick["path"]})
    every = [c for p in prompts.values() for c in p["candidates"]]
    summary = {"prompts": len(prompts), "candidatesPerPrompt": len(every) // max(len(prompts), 1),
               "candidates": len(every),
               "candidateTracked": sum(c["tracked"] for c in every),
               "candidateNoCrouch": sum(not c["unintendedCrouch"] for c in every),
               "candidateBoth": sum(c["tracked"] and not c["unintendedCrouch"] for c in every),
               "promptsWithSelection": len(selected)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps({"summary": summary, "prompts": rows},
                                                   ensure_ascii=False, indent=1))
    # evaluate_tracker --paths-from reads {"clips": [{"path": ...}]}
    (out / "selected-paths.json").write_text(json.dumps({"clips": selected}, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
