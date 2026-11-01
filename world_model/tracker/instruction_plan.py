"""G6: break a Japanese instruction into the house tour's sub-goals with an LLM.

Layer 1 of the architecture: the instruction becomes an ordered list of steps
(walk to a place / sit on a seat) over the place IDs of the scene; ``house_tour``
then plans, builds the reference and tracks it. The LLM only chooses among the
listed IDs; its answer is validated and normalised here:

* a walk to a seat right before sitting on it is dropped (sitting includes the approach)
* repeated consecutive steps collapse; unknown IDs or sitting on a non-seat are errors
* sitting must be the last step (standing up again is not built yet)
* a request outside the listed places (stairs, upstairs, outside) returns no steps

Offline evaluation uses a local model through ollama (``--model``), no external
service; in the product the conversation LLM would make the same call.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11888/api/chat"

# Stage-3 house (scenes.generate_scene(3, ...)): goal id -> Japanese name, kind
PLACES = {
    "work-room": ("作業部屋", "walk"),
    "kitchen": ("キッチン", "walk"),
    "bed-room": ("寝室", "walk"),
    "living-room": ("リビング", "walk"),
    "bedroom-chair-seat": ("寝室の椅子", "sit"),
}

SYSTEM = """あなたは家庭用ロボット(学長アバター)の行動計画係です。
ユーザーの日本語の指示を、順番に実行する部分目標に分けてください。
使える場所 ID(これ以外は使わない):
{places}
規則:
- action は "walk"(その場所へ歩いて止まる)か "sit"(その椅子に座る)だけ。sit は椅子の ID にだけ使う。
- 指示に出てくる順番どおりに並べる(「まず」「〜してから」「最後に」に従う)。行かないよう言われた場所は入れない。
- 「全部の部屋」などは、作業部屋・キッチン・寝室・リビングの順。
- 一覧に無い場所や、できないこと(階段・2階・外・物を運ぶ など)を求められたら steps を空にし、unsupported に理由を書く。
- 返すのは JSON {{"steps": [{{"action": "walk" | "sit", "target": "ID"}}], "unsupported": null | "理由"}} だけ。"""


def ask(instruction, model, keep_alive="120s"):
    places = "\n".join(f"- {pid}: {name}({'座れる椅子' if kind == 'sit' else '場所'})"
                       for pid, (name, kind) in PLACES.items())
    body = {"model": model, "stream": False, "think": False, "keep_alive": keep_alive, "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM.format(places=places)},
                         {"role": "user", "content": instruction}]}
    request = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        return json.loads(json.loads(response.read())["message"]["content"])


def normalise(answer):
    """Validated step list [(action, target)], or raise ValueError."""
    steps = [(s.get("action"), s.get("target")) for s in answer.get("steps") or []]
    for action, target in steps:
        if target not in PLACES or action not in ("walk", "sit"):
            raise ValueError(f"unknown step {action} {target}")
        if action == "sit" and PLACES[target][1] != "sit":
            raise ValueError(f"cannot sit on {target}")
    out = []
    for i, step in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if step[0] == "walk" and nxt == ("sit", step[1]):
            continue   # sitting includes walking to the chair
        if step[0] == "walk" and PLACES[step[1]][1] == "sit":
            step = ("walk", step[1])
        if out and out[-1] == step:
            continue
        out.append(step)
    if any(action == "sit" for action, _ in out[:-1]):
        raise ValueError("standing up again after sitting is not supported")
    return out


# (instruction, expected steps or None when it must be refused)
CASES = [
    ("キッチンに寄ってから、寝室の椅子に座って", [("walk", "kitchen"), ("sit", "bedroom-chair-seat")]),
    ("リビングへ行って", [("walk", "living-room")]),
    ("作業部屋に行ってください", [("walk", "work-room")]),
    ("寝室の椅子に座ってください", [("sit", "bedroom-chair-seat")]),
    ("まず作業部屋、次にキッチン、最後にリビングへ", [("walk", "work-room"), ("walk", "kitchen"), ("walk", "living-room")]),
    ("全部の部屋を回ってから、寝室の椅子で休んで",
     [("walk", "work-room"), ("walk", "kitchen"), ("walk", "bed-room"), ("walk", "living-room"),
      ("sit", "bedroom-chair-seat")]),
    ("台所を見てきて", [("walk", "kitchen")]),
    ("居間に移動して、それから寝室に行って", [("walk", "living-room"), ("walk", "bed-room")]),
    ("キッチンには行かずに、作業部屋だけ見てきて", [("walk", "work-room")]),
    ("寝室に行ったら椅子に腰かけて", [("walk", "bed-room"), ("sit", "bedroom-chair-seat")]),
    ("リビングとキッチンを順に見て回って", [("walk", "living-room"), ("walk", "kitchen")]),
    ("書斎に行って", None),
    ("階段を上って2階へ行って", None),
    ("外に出て郵便受けを見てきて", None),
    ("キッチンから水を持ってきて", None),
    ("作業部屋で少し待ってから寝室へ", [("walk", "work-room"), ("walk", "bed-room")]),
    ("最後にリビングに行く前に、キッチンに寄って", [("walk", "kitchen"), ("walk", "living-room")]),
    ("椅子に座りたいので寝室の椅子までお願い", [("sit", "bedroom-chair-seat")]),
    ("寝室とリビングとキッチンと作業部屋をこの順で回って",
     [("walk", "bed-room"), ("walk", "living-room"), ("walk", "kitchen"), ("walk", "work-room")]),
    ("座ってからキッチンへ行って", None),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.8:27b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, start = [], time.time()
    for index, (instruction, expected) in enumerate(CASES):
        began = time.time()
        keep = "0" if index == len(CASES) - 1 else "120s"   # free the GPU after the last call
        try:
            answer = ask(instruction, args.model, keep)
            steps, error = normalise(answer), None
        except Exception as exc:  # noqa: BLE001 - the case is recorded as an error
            answer, steps, error = None, None, str(exc)
        refused = not steps
        correct = refused if expected is None else steps == expected
        rows.append({"instruction": instruction, "expected": expected, "answer": answer,
                     "steps": steps, "error": error, "correct": correct, "seconds": round(time.time() - began, 1)})
        print(f"{'OK' if correct else 'NG'} {instruction} -> {steps or error or 'refused'}", flush=True)
    summary = {"model": args.model, "cases": len(rows), "correct": sum(r["correct"] for r in rows),
               "refusalCases": sum(r["expected"] is None for r in rows),
               "refusedCorrectly": sum(r["correct"] for r in rows if r["expected"] is None),
               "elapsedSeconds": round(time.time() - start, 1)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
