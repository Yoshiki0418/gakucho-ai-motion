#!/usr/bin/env python3
"""収録したコンソールログから、応答と行動の同期ズレを数値化する。

record.py が残した console.jsonl を読み、[SYNC] 行から
  - 音声がタイムライン上のいつ鳴る予定だったか (audio_scheduled)
  - モーションがいつ適用されたか (motion_applied)
を突き合わせて、遅れを算出する。

使い方:
    python3 docs/sync-baseline/analyze.py docs/sync-baseline/recordings/<run>
    python3 docs/sync-baseline/analyze.py <before-run> --compare <after-run>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AUDIO_RE = re.compile(
    r"\[SYNC\] audio_scheduled perf=([\d.]+) startAt=([\d.]+) "
    r"duration=([\d.]+) ctxNow=([\d.]+)"
)
# motion_applied のログは改修前後で項目が異なる。
#   改修前: perf / ctxNow / scheduledUpTo / prompt / latency_ms
#   改修後: perf / ctxNow / startAt / lag / prompt
# 遅れの算出には perf と prompt しか要らないので、その 2 つだけを共通に拾う。
MOTION_RE = re.compile(r'\[SYNC\] motion_applied perf=([\d.]+).*?prompt="([^"]*)"')
MOTION_LATENCY_RE = re.compile(r"latency_ms=(\S+)")
MOTION_LAG_RE = re.compile(r"lag=(-?[\d.]+)ms")
# 実際に再生されたモーションの尺。バックエンドによって大きく違う
# (Custom-T2M は約2.57秒固定、SentiAvatar は発話長そのもの)ので、
# 無動作率はこの実測値から出す。
PLAYED_RE = re.compile(r"\[SYNC\] motion_playing .*?played=([\d.]+)")
# 収録環境に必ず出るノイズ。アプリの不具合ではないので件数から除く。
#   - WebGL: headless はソフトウェア描画になるため性能警告が出る
#   - styled-components: React の dev ビルドだけで出る再レンダー時の警告
# モーション 1 本がおおよそカバーする秒数。Custom-T2M-Lab は入力に関わらず
# 常に約 2.57 秒を返すため、これを既定値にしている。
MOTION_COVER_SEC = 2.57

NOISE = (
    "GL Driver Message",
    "GPU stall due to ReadPixels",
    "WebGL-",
    "a style property during rerender",
)


def parse_scenario(scenario_dir: Path) -> dict | None:
    log_file = scenario_dir / "console.jsonl"
    if not log_file.exists():
        return None

    audio: list[dict] = []
    motion: list[dict] = []
    played: list[float] = []
    errors = 0

    for line in log_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        text = entry["text"]
        if entry["type"] in ("error", "pageerror") and not any(n in text for n in NOISE):
            errors += 1

        m = AUDIO_RE.search(text)
        if m:
            perf, start, duration, ctx_now = (float(g) for g in m.groups())
            # AudioContext の時刻を壁時計へ写す。モーションが最初の音声より先に来ると
            # その時点では AudioContext が無いため、両者は壁時計上で突き合わせる。
            audio.append(
                {
                    "start": start,
                    "duration": duration,
                    "perf_start": perf + (start - ctx_now) * 1000,
                    "perf_end": perf + (start + duration - ctx_now) * 1000,
                }
            )
            continue

        m = PLAYED_RE.search(text)
        if m:
            played.append(float(m.group(1)))
            continue

        m = MOTION_RE.search(text)
        if m:
            latency = MOTION_LATENCY_RE.search(text)
            lag = MOTION_LAG_RE.search(text)
            motion.append(
                {
                    "perf": float(m.group(1)),
                    "prompt": m.group(2),
                    "latency_ms": latency.group(1) if latency else "n/a",
                    # 改修後は「スケジュールした時刻からの遅れ」も出している
                    "schedule_lag_ms": float(lag.group(1)) if lag else None,
                }
            )

    speech_start = min((a["perf_start"] for a in audio), default=None)
    speech_end = max((a["perf_end"] for a in audio), default=None)

    rows = []
    for item in motion:
        now = item["perf"]
        row = {
            "prompt": item["prompt"],
            "t2m_latency_ms": item["latency_ms"],
            "schedule_lag_ms": item.get("schedule_lag_ms"),
        }
        if speech_start is not None:
            row["lag_from_speech_start_ms"] = round(now - speech_start)
            row["after_speech_end_ms"] = round(now - speech_end)
            segment = next(
                (i for i, a in enumerate(audio)
                 if a["perf_start"] <= now < a["perf_end"]),
                None,
            )
            if segment is not None:
                row["segment_playing"] = f"#{segment + 1}"
            elif now < speech_start:
                row["segment_playing"] = "発話開始前"
            else:
                row["segment_playing"] = "発話終了後"

            # 「担当した発話の開始」からのズレ。改修後はモーションが自分の発話へ
            # 紐づくので、第1発話基準の値では良し悪しが判定できない。
            # 発話中に出たならその発話、外に出たなら最も近い発話を基準にする。
            if segment is not None:
                anchor = audio[segment]["perf_start"]
            elif now < speech_start:
                anchor = speech_start
            else:
                anchor = audio[-1]["perf_start"]
            row["lag_from_own_segment_ms"] = round(now - anchor)
        rows.append(row)

    # 「喋っているのに動いていない」時間の割合。
    # タイミングが合っていても、発話の大半が無動作なら不自然に見える。
    # モーション 1 本がカバーする時間は再生尺で近似する(既定 2.57s = Custom-T2M の固定長)。
    speech_seconds = (speech_end - speech_start) / 1000 if audio else 0.0
    # 実測の再生尺があればそれを使う。無い場合(古い収録)だけ既定値で近似する。
    covered_raw = sum(played) if played else len(motion) * MOTION_COVER_SEC
    covered = min(covered_raw, speech_seconds)
    idle_ratio = (
        round((speech_seconds - covered) / speech_seconds * 100, 1)
        if speech_seconds > 0 else None
    )

    return {
        "scenario": scenario_dir.name,
        "idle_body_ratio": idle_ratio,
        "played_seconds": round(sum(played), 2) if played else None,
        "audio_segments": len(audio),
        "speech_duration_s": round((speech_end - speech_start) / 1000, 3) if audio else 0.0,
        "motions": len(motion),
        "console_errors": errors,
        "rows": rows,
    }


def summarize(run_dir: Path) -> list[dict]:
    results = []
    for scenario_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        parsed = parse_scenario(scenario_dir)
        if parsed:
            results.append(parsed)
    return results


def print_run(run_dir: Path, results: list[dict]) -> None:
    print(f"\n===== {run_dir.name} =====")
    for r in results:
        print(f"\n■ {r['scenario']}")
        idle = r.get("idle_body_ratio")
        idle_text = f" / 無動作率 {idle}%" if idle is not None else ""
        print(
            f"   発話セグメント {r['audio_segments']} 個 / 発話長 {r['speech_duration_s']}s / "
            f"モーション {r['motions']} 回{idle_text} / コンソールエラー {r['console_errors']}"
        )
        if not r["rows"]:
            print("   モーションなし")
            continue
        for row in r["rows"]:
            lag = row.get("lag_from_speech_start_ms")
            after = row.get("after_speech_end_ms")
            if lag is None:
                print(f"   - \"{row['prompt']}\" (音声なし / T2M {row['t2m_latency_ms']}ms)")
                continue
            if after is not None and after > 0:
                verdict = "★発話が終わってから発火"
            elif lag < 0:
                verdict = "★発話が始まる前に発火"
            else:
                verdict = ""
            sched = row.get("schedule_lag_ms")
            sched_text = f"、予定からの遅れ {sched:+.0f}ms" if sched is not None else ""
            print(
                f"   - \"{row['prompt']}\": 発話開始から {lag:+}ms、"
                f"発話終了に対し {after:+}ms、再生中={row['segment_playing']}、"
                f"T2M {row['t2m_latency_ms']}ms{sched_text} {verdict}"
            )


def overall(results: list[dict]) -> dict:
    lags = [
        row["lag_from_speech_start_ms"]
        for r in results for row in r["rows"]
        if "lag_from_speech_start_ms" in row
    ]
    own = [
        row["lag_from_own_segment_ms"]
        for r in results for row in r["rows"]
        if "lag_from_own_segment_ms" in row
    ]
    late = [
        row for r in results for row in r["rows"]
        if row.get("after_speech_end_ms", -1) > 0
    ]
    early = [
        row for r in results for row in r["rows"]
        if row.get("lag_from_speech_start_ms", 0) < 0
    ]
    idles = [r["idle_body_ratio"] for r in results
             if r.get("idle_body_ratio") is not None and r["audio_segments"] > 0]
    speech = sum(r["speech_duration_s"] for r in results)
    total_played = sum(r["played_seconds"] or 0 for r in results)
    covered_raw = total_played or sum(r["motions"] for r in results) * MOTION_COVER_SEC
    covered = min(covered_raw, speech) if speech else 0
    return {
        "idle_body_ratio": round((speech - covered) / speech * 100, 1) if speech else None,
        "per_scenario_idle": idles,
        "motion_count": sum(r["motions"] for r in results),
        "measured": len(lags),
        "median_lag_ms": sorted(lags)[len(lags) // 2] if lags else None,
        "median_own_lag_ms": sorted(own)[len(own) // 2] if own else None,
        "abs_max_own_lag_ms": max((abs(v) for v in own), default=None),
        "abs_max_lag_ms": max((abs(v) for v in lags), default=None),
        "fired_after_speech_ended": len(late),
        "fired_before_speech_started": len(early),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()

    base = summarize(args.run)
    print_run(args.run, base)
    stats = overall(base)
    print(f"\n--- {args.run.name} 総括 ---")
    print(f"   モーション {stats['motion_count']} 回 (計測できたもの {stats['measured']} 回)")
    print(
        f"   担当発話の開始からのズレ: 中央値 {stats['median_own_lag_ms']}ms / "
        f"絶対値最大 {stats['abs_max_own_lag_ms']}ms  ← 主指標"
    )
    print(f"   発話が始まる前に発火した回数: {stats['fired_before_speech_started']}")
    print(f"   発話が終わってから発火した回数: {stats['fired_after_speech_ended']}")
    print(f"   無動作率(喋っているのに動いていない時間の割合): {stats['idle_body_ratio']}%  ← 自然さの主指標")

    if args.compare:
        other = summarize(args.compare)
        print_run(args.compare, other)
        other_stats = overall(other)
        print(f"\n===== 比較 =====")
        print(f"{'指標':<28}{args.run.name:>22}{args.compare.name:>22}")
        for key, label in [
            ("median_own_lag_ms", "担当発話からのズレ(中央値)"),
            ("abs_max_own_lag_ms", "担当発話からのズレ(最大)"),
            ("fired_before_speech_started", "発話前に発火した回数"),
            ("fired_after_speech_ended", "発話終了後に発火した回数"),
            ("idle_body_ratio", "無動作率(%)"),
        ]:
            print(f"{label:<28}{str(stats[key]):>22}{str(other_stats[key]):>22}")


if __name__ == "__main__":
    main()
