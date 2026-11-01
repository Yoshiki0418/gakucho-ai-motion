"""学習の様子を画面(/world-model/training)から見られるようにする書き出し係。

30 秒ごとに `.runtime/world-model/tracker/runs/` を読み、次を `.runtime/world-model/monitor/public/` に書く。
このディレクトリを静的配信コンテナ(training-monitor)が配り、フロントエンドが `/training-monitor/` 経由で読む。

- status.json   各学習の進み具合(ステップ・速度・残り時間・状態)と、途中評価の推移
- manifest.json /world-model の「比較する記録」に足す、最新の重みでの再生記録の一覧
- records/      新しいチェックポイントが出るたびに、決まった代表動作を CPU MuJoCo で再生した記録

GPU は使わない(方策の推論も物理も CPU)。学習の邪魔をしない。

    .runtime/world-model/mjx-venv/bin/python -m world_model.tracker.monitor            # 常駐
    .runtime/world-model/mjx-venv/bin/python -m world_model.tracker.monitor --once     # 1 回だけ
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / ".runtime/world-model/tracker/runs"
OUT = ROOT / ".runtime/world-model/monitor/public"
URL_PREFIX = "/training-monitor/"

# 学習ごとの説明(無い学習は引数から作る)
LABELS = {
    "v6": "全身追従の本学習(立つ・歩く・身振り。家具回避の土台)",
    "v7-seated": "椅子に座る・立つ動作を加えた追加学習(v6 の続き)",
    "v7-seated-b": "椅子に座る・立つ動作を加えた追加学習(1.54 億ステップで GPU メモリ不足により停止 → そこから再開)",
    "v7-seated-c": "椅子に座る・立つ動作を加えた追加学習の仕上げ(3.16 億ステップから。途中評価を 3 回に減らして再開)",
    "v8": "立つ・座るを両方保つ学習(v6 の続き。毎回、全動作を再生して失敗した動作を多めに練習。椅子ありでも体重 64.7 kg に修正)",
    "v9": "T2M の生成動作にも強くする学習(v8 の続き。評価とは別の 248 文の生成 744 本を後処理して練習に追加)",
    "v10": "家の中の流れを練習(v9 の続き。向きを変えて歩く・振り向いて下がって椅子に座る 547 本を追加)",
    "v11": "段差と短い階段の上り下りを練習(v10 の続き。歩行を段に合わせた手本 400 本、学習環境に段の箱を置く)",
    "v12": "段差と短い階段の練習を続ける(v11 の続き。v11 では段の手本の失敗が 4 ラウンドで 219→97/400 と減り続けていた)",
    "v13": "段差と短い階段の練習をさらに続ける(v12 の続き。v12 で短い階段 3/16→7/16、段の手本の失敗 91→67/400)",
    "v14": "段差と短い階段の練習を続ける(v13 の続き。家の階段に合わせた手本ができるまでのつなぎ)",
    "v15": "家の階段(16段・蹴上19cm・踏面25cm)の上り下りを加える(v14 の続き、手本 240 本)",
}
# 再開した学習は、元の学習の何ステップ目から続いているか
RESUMED_FROM = {"v7-seated-b": 154_140_672, "v7-seated-c": 315_621_376}

# 毎回同じ動作で比べる(学習に使っていない保留モーキャプ)
SHOWCASE = [
    ("walk", "0001/01/002/002/001.bvh", "歩く"),
    ("wave", "0001/01/008/002/055.bvh", "手を振る"),
    ("sit", "0001/02/004/002/003.bvh", "椅子に座る"),
    ("stand", "0001/01/001/001/005.bvh", "椅子から立ち上がる"),
]
KEEP_CHECKPOINTS = 2  # 学習ごとに再生記録を残すチェックポイントの数


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _state(run: Path, rows: list[dict]) -> tuple[str, str | None]:
    if (run / "done.json").exists():
        return "finished", None
    progress = run / "progress.jsonl"
    age = time.time() - progress.stat().st_mtime if progress.exists() else 1e9
    log = RUNS / f"{run.name}.log"
    tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
    error = next((line.strip() for line in reversed(tail) if re.search(r"Error: |Traceback \(most recent", line)), None)
    if error and age > 120:
        if "Failed to allocate" in error:
            error = "GPU メモリ不足で停止(" + error.split("Failed to allocate", 1)[1].split("[")[0].strip() + ")"
        return "stopped", error[:200]
    if age < 600:  # 評価の間は 1〜2 分書かれないことがある
        return "running", None
    return "stopped", "進捗が 10 分以上更新されていません"


def summarize(run: Path) -> dict | None:
    meta_path = run / "run.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    args = meta.get("args", {})
    rows = _rows(run / "progress.jsonl")
    state, reason = _state(run, rows)
    # 回に分けた学習(adaptive_train.sh)は、全体の予定と通算のステップを run.json・進捗に持つ
    total = int(meta.get("plannedTimesteps") or args.get("timesteps", 0))
    step = int(rows[-1]["step"]) if rows else 0
    elapsed = float(rows[-1].get("elapsedSeconds", 0)) if rows else 0.0
    plain = [r for r in rows if "eval/episode_reward" not in r]
    rate = None
    if len(plain) >= 2 and plain[-1]["elapsedSeconds"] > plain[-2]["elapsedSeconds"]:
        rate = (plain[-1]["step"] - plain[-2]["step"]) / (plain[-1]["elapsedSeconds"] - plain[-2]["elapsedSeconds"])
    offset = RESUMED_FROM.get(run.name, 0)
    evals = [{"step": int(r["step"]) + offset, "failed": round(r.get("eval/episode_failed", 0.0), 4),
              "positionErrorCm": round(r.get("eval/episode_position_error", 0.0), 2),
              "reward": round(r.get("eval/episode_reward", 0.0), 2)}
             for r in rows if "eval/episode_reward" in r]
    final = None
    summary_path = RUNS / f"{run.name}-eval-final" / "summary.json"
    if summary_path.exists():
        final = json.loads(summary_path.read_text())
    label = LABELS.get(run.name) or ("座る動作を含む学習" if args.get("include_seated") else "全身追従の学習")
    started = meta_path.stat().st_mtime
    return {
        "name": run.name, "label": label, "state": state, "reason": reason,
        "startedAt": dt.datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "updatedAt": dt.datetime.fromtimestamp((run / "progress.jsonl").stat().st_mtime).isoformat(timespec="seconds")
        if (run / "progress.jsonl").exists() else None,
        "step": step + offset, "totalSteps": total + offset, "resumedFrom": offset or None,
        "elapsedSeconds": round(elapsed), "stepsPerSecond": round(rate) if rate else None,
        "etaSeconds": round((total - step) / rate) if rate and state == "running" else None,
        "includeSeated": bool(args.get("include_seated")), "evals": evals, "finalEvaluation": final,
    }


def latest_checkpoint(run: Path) -> Path | None:
    candidates = sorted((run / "checkpoints").glob("params_*.pkl"))
    final = run / "final_params.pkl"
    if final.exists():
        return final
    return candidates[-1] if candidates else None


def _rollout(args):
    """1 本の代表動作を最新の重みで再生する(別プロセス、CPU)。"""
    run, params, library, clip_path, seated, work = args
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TRACKER_EVAL_SEATED"] = "1" if seated else "0"
    from .evaluate_tracker import _evaluate_clip
    return _evaluate_clip((run, params, library, clip_path, 12., work))


def render_checkpoint(run: Path, params: Path, step: int, summary: dict) -> list[dict]:
    """代表動作を再生し、/world-model で再生できる記録にする。"""
    import numpy as np

    from . import body as B
    from .mjx_tracking import load_clips, seat_pose
    from .publish_viewer import trace_from_qpos

    meta = json.loads((run / "run.json").read_text())
    library = str(ROOT / meta["args"]["library"]) if not os.path.isabs(meta["args"]["library"]) else meta["args"]["library"]
    seated = bool(meta["args"].get("include_seated"))
    chosen = [item for item in SHOWCASE if seated or item[0] not in ("sit", "stand")]
    work = OUT.parent / "work" / run.name / str(step)
    jobs = [(str(run), str(params), library, path, seated and key in ("sit", "stand"), str(work)) for key, path, _ in chosen]
    with get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_rollout, jobs, chunksize=1)

    target = OUT / "records" / run.name / str(step)
    target.mkdir(parents=True, exist_ok=True)
    seat_clips = {}
    if seated:
        _, clips, _ = load_clips(library, split="all", include_seated=True, keys=("qpos",))
        seat_clips = {clip["path"]: clip for clip in clips if clip.get("seat") is not None}
    options = []
    for (key, path, title), result in zip(chosen, results):
        traj = np.load(work / f"{path.replace('/', '_').replace('#', '_')}.npz")
        use_seat = key in ("sit", "stand") and path in seat_clips
        if use_seat:
            from .mjx_tracking import seat_xml
            model = B.build_model(extra_world=[seat_xml()])
            mocap = [seat_pose(seat_clips[path]["seat"])]
        else:
            model, mocap = B.build_model(), None
        model.opt.iterations, model.opt.ls_iterations = 6, 10
        seconds = (len(traj["qpos"]) - 1) / 50.
        outcome = (f"最後まで追従({seconds:.1f}秒)" if result["success"]
                   else f"{result['failedAtSeconds']:.1f}秒で追従失敗(体節の平均誤差 0.5m 超)")
        status = f"学習中の重み {run.name}・{step / 1e8:.2f}億ステップ・{title}・{outcome}・平均誤差{result['meanBodyErrorMeters'] * 100:.0f}cm"
        trace = trace_from_qpos(model, traj["qpos"], mocap=mocap, meta={
            "controller": f"全身追従器 {run.name}(学習中・{step / 1e8:.2f}億ステップ時点の重み)",
            "motionReference": f"保留モーキャプ(学習に未使用):{result['text']}", "viewerStatus": status,
            "tracker": {"run": run.name, "step": step, "params": params.name, "clip": path, "result": result},
        })
        name = f"{key}.json"
        (target / name).write_text(json.dumps(trace, ensure_ascii=False))
        options.append({"stage": 1, "url": f"{URL_PREFIX}records/{run.name}/{step}/{name}",
                        "title": f"【学習中 {run.name} {step / 1e8:.2f}億】{title}({'成功' if result['success'] else '失敗'})",
                        "key": key, "success": result["success"], "meanErrorCm": round(result["meanBodyErrorMeters"] * 100, 1),
                        "failedAtSeconds": result["failedAtSeconds"]})
    (target / "index.json").write_text(json.dumps(options, ensure_ascii=False, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    # 古いチェックポイントの記録は消す
    steps = sorted((int(p.name) for p in (OUT / "records" / run.name).iterdir() if p.name.isdigit()), reverse=True)
    for old in steps[KEEP_CHECKPOINTS:]:
        shutil.rmtree(OUT / "records" / run.name / str(old), ignore_errors=True)
    return options


def _write(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    tmp.replace(path)


def collect_records() -> list[dict]:
    options = []
    root = OUT / "records"
    if not root.exists():
        return options
    for run in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        for step_dir in sorted((p for p in run.iterdir() if p.name.isdigit()), key=lambda p: -int(p.name)):
            index = step_dir / "index.json"
            if index.exists():
                options += json.loads(index.read_text())
    return options


def tick(render: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 動作確認用の短い学習(sanity)は出さない
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and (p / "run.json").exists() and "sanity" not in p.name),
                  key=lambda p: (p / "run.json").stat().st_mtime, reverse=True)[:3]
    summaries = [s for s in (summarize(run) for run in runs) if s]
    if render:
        for run, summary in zip(runs, summaries):
            if summary["state"] != "running":
                continue
            params = latest_checkpoint(run)
            if params is None:
                continue
            step = int(re.findall(r"\d+", params.stem)[0]) + RESUMED_FROM.get(run.name, 0) if "params_" in params.name else summary["step"]
            if (OUT / "records" / run.name / str(step) / "index.json").exists():
                continue
            print(f"[monitor] rendering {run.name} step={step}", flush=True)
            started = time.time()
            render_checkpoint(run, params, step, summary)
            print(f"[monitor] rendered in {time.time() - started:.0f}s", flush=True)
            summaries = [s for s in (summarize(r) for r in runs) if s]
    records = collect_records()
    for summary in summaries:
        summary["records"] = [r for r in records if r["url"].startswith(f"{URL_PREFIX}records/{summary['name']}/")]
    now = dt.datetime.now().isoformat(timespec="seconds")
    _write(OUT / "status.json", {"version": 1, "updatedAt": now, "runs": summaries})
    _write(OUT / "manifest.json", {"version": 1, "updatedAt": now,
                                   "traceOptions": [{"stage": r["stage"], "title": r["title"], "url": r["url"]} for r in records]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=30.)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            tick(render=not args.no_render)
        except Exception as error:  # noqa: BLE001  (書き出しの失敗で常駐を止めない)
            print(f"[monitor] error: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
