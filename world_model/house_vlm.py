"""G7: what does the local VLM (Qwen3-VL-30B through ollama) understand from the head camera?

    python -m world_model.house_vlm --vision .runtime/world-model/house2f/vision --output .../vlm

Uses the views of ``house_vision`` (every walk place of ``house2f``, eight headings, the depth camera's
pose and field of view). Two questions, both answered as JSON:
* room: standing at a place, the four images at 0/90/180/270 degrees -> which room am I in (one of
  ROOM_CHOICES). Scored against the room of the place (LDK places against their zone).
* objects: one image -> the main objects in it (Japanese). Scored by recall of the listed COCO-class
  objects of the ground truth that are at least half in the frame, by keyword (OBJECT_WORDS).
Latency of every call is recorded (the executor will decide about once a second).
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOM_CHOICES = ["リビング", "キッチン", "ダイニング", "畳コーナー", "玄関", "廊下・ホール", "階段", "トイレ", "洗面所",
                "浴室", "寝室", "子ども部屋", "ウォークインクローゼット", "スタディコーナー", "わからない"]
# place -> answers counted as right
EXPECTED = {
    "living": {"リビング"}, "kitchen": {"キッチン"}, "dining": {"ダイニング"}, "tatami": {"畳コーナー", "リビング"},
    "entrance": {"玄関", "廊下・ホール"}, "hall-1f": {"廊下・ホール"}, "toilet-1f": {"トイレ"}, "washroom": {"洗面所"},
    "bathroom": {"浴室"}, "hall-2f": {"廊下・ホール", "階段"}, "study-corner": {"スタディコーナー", "廊下・ホール"},
    "master-bedroom": {"寝室"}, "walk-in-closet": {"ウォークインクローゼット"}, "kids-room-1": {"子ども部屋", "寝室"},
    "kids-room-2": {"子ども部屋", "寝室"}, "toilet-2f": {"トイレ"},
}
# COCO class -> words that count as naming it in the Japanese list
OBJECT_WORDS = {
    "refrigerator": ["冷蔵庫"], "oven": ["コンロ", "IH", "オーブン", "調理器"], "sink": ["シンク", "流し", "洗面台", "洗面ボウル"],
    "microwave": ["電子レンジ", "レンジ"], "toaster": ["トースター"], "dining table": ["テーブル", "食卓"],
    "chair": ["椅子", "いす", "イス", "チェア"], "couch": ["ソファ"], "tv": ["テレビ", "モニター", "ディスプレイ", "画面"],
    "potted plant": ["植物", "観葉", "鉢"], "vase": ["花瓶", "花"], "bowl": ["器", "ボウル", "皿", "果物"],
    "clock": ["時計"], "toilet": ["トイレ", "便器", "便座"], "bed": ["ベッド"], "laptop": ["パソコン", "ノート"],
    "backpack": ["ランドセル", "かばん", "鞄", "リュック", "バッグ"], "teddy bear": ["ぬいぐるみ", "くま", "クマ"],
    "sports ball": ["ボール"], "book": ["本"], "umbrella": ["傘"], "hair drier": ["ドライヤー"], "toothbrush": ["歯ブラシ"],
}
ROOM_PROMPT = ("あなたは家庭用ロボットの目です。いま家の中のある場所に立ち、その場で向きを変えながら頭のカメラで4枚撮りました"
               "（順に 0°・90°・180°・270°）。あなたが今立っている部屋はどこですか。次から1つ選んでください: "
               + "、".join(ROOM_CHOICES) + "。JSON だけで答えてください: {\"room\": \"...\", \"reason\": \"短い理由\"}")
OBJECT_PROMPT = ("あなたは家庭用ロボットの目です。頭のカメラの画像に写っている主な物（家具・家電・日用品）を日本語で挙げてください。"
                 "JSON だけで答えてください: {\"objects\": [\"...\", ...]}")


def ask(url, model, prompt, images, timeout=600):
    body = {"model": model, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(Path(i).read_bytes()).decode() for i in images]}]}
    request = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    seconds = time.time() - started
    text = reply.get("message", {}).get("content", "")
    try:
        answer = json.loads(text)
    except ValueError:
        answer = {"raw": text}
    return answer, round(seconds, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", required=True, help="house_vision output (views.json, images/)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11999")
    parser.add_argument("--model", default="qwen3-vl:30b-a3b-instruct")
    parser.add_argument("--objects", type=int, default=0, help="views for the object question (0 = all)")
    args = parser.parse_args()
    vision, out = Path(args.vision), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    views = json.loads((vision / "views.json").read_text())
    by_place = defaultdict(dict)
    for v in views:
        by_place[v["place"]][round(v["heading"] * 180 / 3.141592653589793) % 360] = v

    ask(args.url, args.model, "こんにちは。JSON で {\"ok\": true} と答えて", [])   # load the model before timing
    rooms = []
    for place, headings in by_place.items():
        images = [vision / "images" / f"{headings[d]['name']}.jpg" for d in (0, 90, 180, 270) if d in headings]
        answer, seconds = ask(args.url, args.model, ROOM_PROMPT, images)
        room = str(answer.get("room", "")).strip()
        rooms.append({"place": place, "answer": room, "right": room in EXPECTED.get(place, set()), "reason": answer.get("reason"),
                      "seconds": seconds, "images": len(images)})
        print(json.dumps(rooms[-1], ensure_ascii=False), flush=True)
    objects, found, total = [], defaultdict(int), defaultdict(int)
    for v in views[:args.objects or len(views)]:
        answer, seconds = ask(args.url, args.model, OBJECT_PROMPT, [vision / "images" / f"{v['name']}.jpg"])
        named = " ".join(str(o) for o in answer.get("objects", []))
        hits = []
        for t in v["truth"]:
            whole = t.get("inside", 1) >= 0.5 and (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) >= 1500
            if not whole:
                continue
            total[t["coco"]] += 1
            if any(word in named for word in OBJECT_WORDS.get(t["coco"], [])):
                found[t["coco"]] += 1
                hits.append(t["coco"])
        objects.append({"view": v["name"], "named": answer.get("objects", answer.get("raw")), "found": hits, "seconds": seconds})
    secs = [r["seconds"] for r in rooms] + [o["seconds"] for o in objects]
    report = {"model": args.model,
              "room": {"right": sum(r["right"] for r in rooms), "places": len(rooms),
                       "secondsMean": round(sum(r["seconds"] for r in rooms) / max(len(rooms), 1), 2), "answers": rooms},
              "objects": {"found": sum(found.values()), "visible": sum(total.values()),
                          "rate": round(sum(found.values()) / max(sum(total.values()), 1), 3),
                          "byClass": {c: [found[c], total[c]] for c in sorted(total, key=lambda c: -total[c])},
                          "secondsMean": round(sum(o["seconds"] for o in objects) / max(len(objects), 1), 2), "views": objects},
              "secondsMax": max(secs) if secs else None}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"room": {k: report["room"][k] for k in ("right", "places", "secondsMean")},
                      "objects": {k: report["objects"][k] for k in ("found", "visible", "rate", "byClass", "secondsMean")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
