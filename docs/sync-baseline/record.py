#!/usr/bin/env python3
"""応答と行動の同期を評価するためのベースライン収録スクリプト。

Playwright で Chromium を起動し、シナリオごとに
  - 画面の動画 (webm)
  - ブラウザコンソールログ (jsonl)
を recordings/<run>/<scenario-id>/ 以下に残す。

改修の前後で同じシナリオを流し、analyze.py で数値を突き合わせて比較する。

使い方:
    python3 docs/sync-baseline/record.py --label before
    python3 docs/sync-baseline/record.py --label before --only 01-greeting
    python3 docs/sync-baseline/record.py --label before --url http://localhost:4000

前提:
    pip install playwright && python3 -m playwright install chromium
    フロントエンドが起動していること。backend:true のシナリオはバックエンドも必要。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit(
        "playwright が見つかりません。\n"
        "  pip install playwright && python3 -m playwright install chromium"
    )

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SCENARIOS = ROOT / "scenarios.json"
VIEWPORT = {"width": 1600, "height": 900}
LOCK = ROOT / "recordings" / ".record.lock"


class RecordingLock:
    """収録の多重起動を防ぐ。

    2 つの収録が同時に走ると、同じバックエンドへ並行してリクエストが飛び、
    対話履歴のリセットも衝突して、シナリオの出力が複数ディレクトリへ分裂する。
    セッションが切れて再実行するときに起きやすいので、明示的にロックを取る。
    """

    def __enter__(self) -> "RecordingLock":
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = LOCK.read_text(encoding="utf-8").strip() or "(不明)"
            alive = owner.isdigit() and Path(f"/proc/{owner}").exists()
            if alive:
                sys.exit(
                    f"すでに収録が実行中です (PID {owner})。\n"
                    f"止める場合: kill {owner} && rm {LOCK}"
                )
            print(f"[warn] 前回の収録({owner})の残骸ロックを引き継ぎます")
            LOCK.unlink(missing_ok=True)
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *_exc) -> None:
        LOCK.unlink(missing_ok=True)


def to_mp4(webm: Path) -> Path | None:
    """Playwright が出す webm を mp4(H.264) へ変換する。

    webm はブラウザやプレゼン資料での扱いが悪いので mp4 を既定にする。
    ffmpeg が無い環境では webm のまま残す。
    """
    if not shutil.which("ffmpeg"):
        print("  [warn] ffmpeg が無いため webm のまま残します")
        return None

    mp4 = webm.with_suffix(".mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(webm),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            # 収録は可変フレームレートなので固定して再生互換を上げる
            "-vsync", "cfr", "-r", "25",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not mp4.exists():
        print(f"  [warn] mp4 変換に失敗: {result.stderr.strip()[:200]}")
        return None

    webm.unlink(missing_ok=True)
    return mp4


def mux_audio(video: Path, chunks: list[dict], perf_origin: float,
              video_origin: float, work_dir: Path) -> bool:
    """ブラウザが鳴らした音声を、再生スケジュールどおりに動画へ合成する。

    Playwright の画面収録には音声が入らない。一方で SSE の音声(base64 WAV)と
    「AudioContext 上のどの時刻に鳴らしたか」は両方とも取得できるので、
    ループバック録音より正確に元の再生を再構成できる。

    video_time(perf) = (perf - perf_origin) / 1000 + video_origin
    実際の再生開始は AudioContext 側で startAt にスケジュールされているので、
    ログ時点(ctxNow)からの先送り分 (startAt - ctxNow) を足す。
    """
    if not chunks or not shutil.which("ffmpeg"):
        return False

    work_dir.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []

    for i, chunk in enumerate(chunks):
        wav = work_dir / f"{i:03d}.wav"
        try:
            wav.write_bytes(base64.b64decode(chunk["base64"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 音声{i}のデコードに失敗: {exc}")
            continue

        at = (
            (chunk["perf"] - perf_origin) / 1000
            + video_origin
            + (chunk["startAt"] - chunk["ctxNow"])
        )
        delay_ms = max(0, round(at * 1000))
        inputs += ["-i", str(wav)]
        # adelay はチャンネルごとに指定が要る。all=1 で全チャンネルへ適用する。
        filters.append(f"[{len(labels) + 1}:a]adelay={delay_ms}:all=1[a{len(labels)}]")
        labels.append(f"[a{len(labels)}]")

    if not labels:
        return False

    filter_complex = ";".join(filters) + ";" + "".join(labels)
    filter_complex += f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[aout]"

    out = video.with_name(video.stem + "-with-audio.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), *inputs,
         "-filter_complex", filter_complex,
         "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out.exists():
        print(f"  [warn] 音声合成に失敗: {result.stderr.strip()[:200]}")
        return False

    video.unlink(missing_ok=True)
    out.rename(video)
    shutil.rmtree(work_dir, ignore_errors=True)
    return True


def load_scenarios(only: str | None) -> list[dict]:
    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    scenarios = data["scenarios"]
    if only:
        scenarios = [s for s in scenarios if s["id"] == only]
        if not scenarios:
            sys.exit(f"シナリオ '{only}' が見つかりません")
    return scenarios


# チャット入力欄。styled-components 製の <input> で type 属性が付いていないため、
# input[type=text] では引っかからない。placeholder を第一候補にする。
INPUT_PLACEHOLDER = "話しかけるか入力してください"
INPUT_FALLBACK = (
    "textarea:visible, "
    "input[placeholder]:visible, "
    "input:not([type=file]):not([type=checkbox]):not([type=radio]):visible"
)


def find_input(page):
    box = page.get_by_placeholder(INPUT_PLACEHOLDER)
    if box.count() > 0:
        return box.first
    return page.locator(INPUT_FALLBACK).first


def send_message(page, text: str) -> None:
    """チャット入力欄にテキストを入れて送信する。"""
    box = find_input(page)
    box.wait_for(state="visible", timeout=15000)
    box.click()
    box.fill(text)
    page.keyboard.press("Enter")


def switch_to_character(page) -> None:
    """アバター表示を『キャラ』モードへ切り替える。"""
    for label in ("🧸 キャラ", "キャラ"):
        button = page.get_by_role("button", name=label)
        if button.count() > 0:
            button.first.click()
            return
    print("  [warn] キャラ切替ボタンが見つかりませんでした")


def run_scenario(p, scenario: dict, out_dir: Path, url: str, avatar: str | None, headed: bool) -> dict:
    sid = scenario["id"]
    print(f"▶ {sid}: {scenario['title']}")
    scenario_dir = out_dir / sid
    scenario_dir.mkdir(parents=True, exist_ok=True)

    logs: list[dict] = []
    # 学長アバターは 20-30MB あり、ソフトウェア描画で連続再生するとメモリが積み上がり
    # ブラウザごと落ちる。シナリオごとに作り直して切り離す。
    browser = p.chromium.launch(
        headless=not headed,
        # 自動再生をブロックさせない (音声が鳴らないと同期計測ができない)
        args=["--autoplay-policy=no-user-gesture-required", "--use-gl=swiftshader"],
    )
    context = browser.new_context(
        viewport=VIEWPORT,
        record_video_dir=str(scenario_dir / "_video"),
        record_video_size=VIEWPORT,
    )
    page = context.new_page()
    started = time.time()

    page.on(
        "console",
        lambda msg: logs.append(
            {"t": round(time.time() - started, 4), "type": msg.type, "text": msg.text}
        ),
    )
    page.on(
        "pageerror",
        lambda err: logs.append(
            {"t": round(time.time() - started, 4), "type": "pageerror", "text": str(err)}
        ),
    )

    # シナリオ同士が履歴で干渉しないよう、対話履歴をリセットしてから始める
    try:
        reset = page.request.post(f"{url.rstrip('/')}/api/text-chat/reset")
        print(f"  履歴リセット: {reset.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 履歴リセットに失敗(バックエンド未起動?): {exc}")

    # 音声キャプチャを有効化する(このフラグが無ければアプリ側は何もしない)
    page.add_init_script("window.__syncCapture = { chunks: [] }")

    page.goto(url, wait_until="networkidle")
    # 動画の t=0 とページの performance.now() を対応付ける
    perf_origin = page.evaluate("performance.now()")
    video_origin = time.time() - started
    page.wait_for_timeout(2500)
    switch_to_character(page)
    page.wait_for_timeout(1500)

    if avatar:
        select = page.locator("select").first
        if select.count() > 0:
            select.select_option(value=f"/avatars/{avatar}")
            # 学長アバターは 20-30MB あり、ソフトウェア描画だと読み込みに時間がかかる
            page.wait_for_timeout(14000)

    if scenario.get("bvh_file"):
        bvh_path = REPO / scenario["bvh_file"]
        if bvh_path.exists():
            page.locator("input[type=file][accept*='.bvh']").first.set_input_files(str(bvh_path))
        else:
            print(f"  [warn] BVH が見つかりません: {bvh_path}")

    inputs = scenario.get("input")
    if inputs:
        if isinstance(inputs, str):
            inputs = [inputs]
        per_turn = max(scenario.get("duration_sec", 30) // len(inputs), 8)
        for text in inputs:
            print(f"  送信: {text}")
            send_message(page, text)
            page.wait_for_timeout(per_turn * 1000)
    else:
        page.wait_for_timeout(scenario.get("duration_sec", 30) * 1000)

    page.wait_for_timeout(1500)

    try:
        audio_chunks = page.evaluate(
            "(window.__syncCapture && window.__syncCapture.chunks) || []"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 音声の回収に失敗: {exc}")
        audio_chunks = []

    # 応答本文を控える。意味的な一致(応答内容と動作が噛み合っているか)の判断に要る。
    try:
        transcript = page.evaluate(
            "Array.from(document.querySelectorAll('main, body'))"
            ".map((n) => n.innerText).sort((a, b) => b.length - a.length)[0] || ''"
        )
    except Exception:  # noqa: BLE001
        transcript = ""
    (scenario_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    video = page.video
    page.close()  # 動画は close 時に確定する

    video_path = None
    if video:
        try:
            saved = scenario_dir / "screen.webm"
            video.save_as(str(saved))
            video_path = to_mp4(saved) or saved
            if video_path.suffix == ".mp4" and audio_chunks:
                if mux_audio(
                    video_path, audio_chunks, perf_origin, video_origin,
                    scenario_dir / "_audio",
                ):
                    print(f"  音声 {len(audio_chunks)} 区間を合成")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] 動画の保存に失敗: {exc}")

    context.close()
    browser.close()
    shutil.rmtree(scenario_dir / "_video", ignore_errors=True)

    (scenario_dir / "console.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in logs) + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "scenario.json").write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sync_lines = [entry for entry in logs if "[SYNC]" in entry["text"]]
    print(f"  ログ {len(logs)} 行 (うち [SYNC] {len(sync_lines)} 行)")
    if video_path:
        print(f"  動画 {video_path.relative_to(REPO)}")

    return {
        "id": sid,
        "logs": len(logs),
        "sync": len(sync_lines),
        "video": bool(video_path),
        "audio_chunks": len(audio_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="収録ラベル (例: before / after)")
    parser.add_argument("--url", default="http://localhost:4000")
    parser.add_argument("--only", default=None, help="1シナリオだけ実行する")
    parser.add_argument("--avatar", default=None, help="使用する glb (例: gakucho_v3_suit.glb)")
    parser.add_argument("--headed", action="store_true", help="ブラウザを表示して実行")
    args = parser.parse_args()

    scenarios = load_scenarios(args.only)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "recordings" / f"{stamp}-{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"収録先: {out_dir.relative_to(REPO)}\n")

    results = []
    with RecordingLock(), sync_playwright() as p:
        for scenario in scenarios:
            # 1シナリオが落ちても残りは収録する。部分的な結果でも比較には使える。
            try:
                results.append(
                    run_scenario(p, scenario, out_dir, args.url, args.avatar, args.headed)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] {scenario['id']} が失敗: {type(exc).__name__}: {exc}")
                results.append(
                    {"id": scenario["id"], "error": f"{type(exc).__name__}: {exc}"}
                )

    (out_dir / "run.json").write_text(
        json.dumps(
            {"label": args.label, "url": args.url, "avatar": args.avatar,
             "recorded_at": stamp, "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n完了。次に:\n  python3 docs/sync-baseline/analyze.py {out_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
