#!/usr/bin/env python3
"""収録した mp4 に、発話区間とモーション発火のタイムライン帯を焼き込む。

Playwright の画面収録には音声トラックが入らないため、動画を見ただけでは
「喋っているタイミング」と「動いたタイミング」の関係が分からない。
コンソールの [SYNC] ログから両者を復元し、画面下部に帯として描き込む。

  緑の帯 SPEAKING   … 音声が鳴っている区間
  桃色  MOTION      … モーションが適用された瞬間(1.2秒だけ点灯)

使い方:
    python3 docs/sync-baseline/annotate.py docs/sync-baseline/recordings/<run>
    python3 docs/sync-baseline/annotate.py <run>/01-greeting     # 単体でも可
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import AUDIO_RE, MOTION_RE  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MOTION_FLASH_SEC = 1.2


def extract_events(log_file: Path) -> tuple[list[tuple[float, float]], list[float]]:
    """console.jsonl から (発話区間, モーション発火時刻) を動画時間で返す。"""
    audio: list[tuple[float, float]] = []
    motion: list[float] = []
    offset: float | None = None  # 動画時間 = perf/1000 + offset

    for line in log_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        text = entry["text"]

        m = AUDIO_RE.search(text)
        if m:
            perf, start, duration, ctx_now = (float(g) for g in m.groups())
            if offset is None:
                offset = entry["t"] - perf / 1000
            begin = perf / 1000 + (start - ctx_now) + offset
            audio.append((begin, begin + duration))
            continue

        m = MOTION_RE.search(text)
        if m:
            perf = float(m.group(1))
            if offset is None:
                offset = entry["t"] - perf / 1000
            motion.append(perf / 1000 + offset)

    return audio, motion


def build_filter(audio: list[tuple[float, float]], motion: list[float]) -> str:
    """発話帯・モーション帯・経過時間を描く ffmpeg フィルタを組み立てる。"""
    parts = [
        # 帯の下地
        "drawbox=x=0:y=ih-64:w=iw:h=64:color=black@0.55:t=fill",
        f"drawtext=fontfile={FONT}:text='t=%{{pts\\:hms}}':"
        "x=16:y=h-46:fontsize=22:fontcolor=white@0.85",
    ]

    for begin, end in audio:
        parts.append(
            f"drawbox=x=200:y=ih-52:w=250:h=18:color=0x22c55e@0.95:t=fill:"
            f"enable='between(t,{begin:.3f},{end:.3f})'"
        )
    parts.append(
        f"drawtext=fontfile={FONT}:text='SPEAKING':x=210:y=h-50:"
        "fontsize=15:fontcolor=black:"
        + "enable='" + "+".join(f"between(t,{b:.3f},{e:.3f})" for b, e in audio) + "'"
        if audio
        else "null"
    )

    for at in motion:
        parts.append(
            f"drawbox=x=480:y=ih-52:w=250:h=18:color=0xec4899@0.95:t=fill:"
            f"enable='between(t,{at:.3f},{at + MOTION_FLASH_SEC:.3f})'"
        )
    if motion:
        parts.append(
            f"drawtext=fontfile={FONT}:text='MOTION':x=492:y=h-50:"
            "fontsize=15:fontcolor=black:enable='"
            + "+".join(
                f"between(t,{a:.3f},{a + MOTION_FLASH_SEC:.3f})" for a in motion
            )
            + "'"
        )

    # 常時表示の凡例
    parts.append(
        f"drawtext=fontfile={FONT}:text='speech':x=210:y=h-26:"
        "fontsize=13:fontcolor=0x22c55e"
    )
    parts.append(
        f"drawtext=fontfile={FONT}:text='motion':x=492:y=h-26:"
        "fontsize=13:fontcolor=0xec4899"
    )
    return ",".join(p for p in parts if p != "null")


def annotate(scenario_dir: Path) -> bool:
    log_file = scenario_dir / "console.jsonl"
    source = scenario_dir / "screen.mp4"
    if not source.exists():
        source = scenario_dir / "screen.webm"
    if not log_file.exists() or not source.exists():
        return False

    audio, motion = extract_events(log_file)
    if not audio and not motion:
        print(f"  {scenario_dir.name}: [SYNC] イベントが無いので注釈なし")
        return False

    out = scenario_dir / "screen-annotated.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(source),
        "-vf", build_filter(audio, motion),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p",
        # 合成済みの音声を落とさないようそのまま通す(音声が無い動画でも失敗しない)
        "-c:a", "copy", "-map", "0:v", "-map", "0:a?",
        "-movflags", "+faststart",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {scenario_dir.name}: 注釈に失敗 {result.stderr.strip()[:200]}")
        return False

    print(
        f"  {scenario_dir.name}: 発話 {len(audio)} 区間 / モーション {len(motion)} 回 "
        f"→ {out.name}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="収録ディレクトリ(run 全体 or シナリオ単体)")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg が必要です")

    if (args.target / "console.jsonl").exists():
        annotate(args.target)
        return

    for scenario_dir in sorted(p for p in args.target.iterdir() if p.is_dir()):
        annotate(scenario_dir)


if __name__ == "__main__":
    main()
