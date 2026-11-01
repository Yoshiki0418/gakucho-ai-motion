"""会話用「意味の層」のクリップライブラリを作る(ビルド時のみ実行)。

本番の応答中に T2M を呼ぶ代わりに、行動タグごとの動作を事前に作って品質を確かめておく。

- コミュニケーション動作カタログ(MOT_7xx)のうち、モーキャプ収録があるもの
  (Custom-T2M-Lab の 0010/01、被験者 005)は実データをそのまま使う。
- 収録が無いタグは T2M で候補を複数生成し、崩れ検出で最良の 1 本を選ぶ。

各クリップは T2M 出力と同じ標準骨格(55 関節)・30fps にそろえ、
動作区間の切り出し、収録ノイズ(1 フレームだけ跳ぶ回転)の補修、
ストローク(動作の山場)時刻と動く部位の判定をここで済ませておく。
立ち姿は会話画面の待機姿勢に合わせ、収録はその立ち姿からの変化として載せ直す(neutral.py)。

実行例(ホストの numpy/scipy 入り環境で):

    PYTHONPATH=backend python -m app.agent.general_conversation.tools.conversation_motion.build_library \\
        --output backend/app/assets/conversation_motion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from .bvh import Motion, Skeleton, forward_kinematics, parse_bvh, qangle, qinv, qmean, qmul, qslerp, resample, write_bvh
from .groups import group_joint_indices
from .library import mirror_motion
from .neutral import onto_neutral, viewer_neutral

SOURCE = Path("/gpu-server/yamamoto/Custom-T2M-Lab/data")
LABELS = SOURCE / "DatasetLabels/0010/01/label.md"
MOCAP_DIR = SOURCE / "BVH/0010/01/005/001"
T2M_SERVICE = "http://127.0.0.1:8100/api/realtime/generate"
T2M_RESULTS = Path("/gpu-server/yamamoto/Custom-T2M-Lab")
FPS = 30.0

# 収録の無いタグを T2M で作るときの文。学習ラベルに一致する文ほど安定する。
T2M_ONLY = {
    "MOT_783": "右手を上げる",
    "MOT_774": "右手の人差し指を立てて「ちょっと待って」と示すジェスチャーをする",
}
# 本番の T2M と同じ生成設定(docker-compose.startup.yml)
T2M_SAMPLING = {"cfg_scale": 7.0, "top_k": 50, "temperature": 0.8}
T2M_CANDIDATES = 10

# 1 フレームで回転がこれ以上跳ぶのは収録ノイズとみなして補修する
GLITCH_RAD_PER_S = 30.0
ACTIVE_THRESHOLD_DEG = {"right_arm": 12.0, "left_arm": 12.0, "head": 6.0, "spine": 6.0, "hips": 8.0}


def load_labels() -> dict[str, str]:
    text = LABELS.read_text(encoding="utf-8-sig")
    return {number: label.strip() for number, label in re.findall(r"(\d{3})\s*\n\s*(.+)", text)}


def leg_length(skeleton: Skeleton) -> float:
    return float(sum(np.linalg.norm(skeleton.joints[skeleton.index(n)].offset) for n in ("LeftLeg", "LeftFoot")))


def to_canonical(motion: Motion, canonical: Skeleton) -> Motion:
    """同じ関節名の骨格へ。回転はそのまま、ルート位置は脚の長さの比で縮める。"""
    index = [motion.skeleton.index(name) for name in canonical.names]
    ratio = leg_length(canonical) / leg_length(motion.skeleton)
    return Motion(canonical, motion.root * ratio, motion.quats[:, index].copy(), motion.fps)


def repair_glitches(motion: Motion) -> tuple[Motion, int]:
    """1〜数フレームだけ跳ぶ回転を、前後の正常フレームの slerp で埋める。"""
    motion = motion.copy()
    speed = qangle(qmul(motion.quats[1:], qinv(motion.quats[:-1]))) * motion.fps  # (T-1, J)
    repaired = 0
    for j in range(motion.quats.shape[1]):
        bad = np.zeros(motion.frames, dtype=bool)
        jumps = np.nonzero(speed[:, j] > GLITCH_RAD_PER_S)[0]
        for k in jumps:
            bad[k + 1] = True
        if not bad.any():
            continue
        t = 0
        while t < motion.frames:
            if not bad[t]:
                t += 1
                continue
            start = t
            while t < motion.frames and bad[t]:
                t += 1
            # 跳んだ先から戻ってくるまでを含めて埋める
            left, right = start - 1, min(t, motion.frames - 1)
            if left < 0:
                motion.quats[:right, j] = motion.quats[right, j]
            else:
                span = right - left
                alpha = np.arange(1, span) / span
                motion.quats[left + 1:right, j] = qslerp(motion.quats[left, j], motion.quats[right, j], alpha)
            repaired += right - left - 1
    return motion, repaired


def analyse(motion: Motion) -> dict:
    """動く部位、動作区間、ストローク時刻を求める。"""
    skeleton = motion.skeleton
    reference = motion.quats[0]
    angle = np.degrees(qangle(qmul(motion.quats, qinv(reference)[None])))  # (T, J)
    groups = {}
    signals = []
    for group, (main, _all) in group_joint_indices(skeleton).items():
        per_joint = angle[:, main]
        value = float(np.percentile(per_joint.max(axis=1), 98))
        active = value >= ACTIVE_THRESHOLD_DEG[group]
        groups[group] = {"rangeDeg": round(value, 1), "active": bool(active)}
        if active:
            signal = per_joint.sum(axis=1)
            signals.append(signal / max(signal.max(), 1e-6))
    if not signals:
        signals = [np.zeros(motion.frames)]
    activity = np.max(np.stack(signals), axis=0)
    above = np.nonzero(activity > 0.12)[0]
    onset = int(above[0]) if len(above) else 0
    offset = int(above[-1]) if len(above) else motion.frames - 1
    stroke = int(np.nonzero(activity >= 0.7)[0][0]) if (activity >= 0.7).any() else onset
    return {"groups": groups, "onset": onset, "offset": offset, "stroke": stroke, "activity": activity}


def quality(motion: Motion) -> dict:
    """崩れ検出に使う量。"""
    positions, _ = forward_kinematics(motion)
    hips = motion.root
    drop = float(hips[0, 1] - hips[:, 1].min())
    travel = float(np.linalg.norm(hips[:, [0, 2]] - hips[0, [0, 2]], axis=1).max())
    speed = qangle(qmul(motion.quats[1:], qinv(motion.quats[:-1]))) * motion.fps
    names = motion.skeleton.names
    feet = [names.index("LeftToeBase"), names.index("RightToeBase")]
    foot_slide = float(np.abs(np.diff(positions[:, feet][:, :, [0, 2]], axis=0)).sum(axis=0).max())
    return {"hipDropM": round(drop, 3), "rootTravelM": round(travel, 3),
            "maxAngularSpeed": round(float(speed.max()), 1), "footSlideM": round(foot_slide, 3)}


def trim(motion: Motion, info: dict) -> tuple[Motion, float]:
    start = max(info["onset"] - int(0.25 * FPS), 0)
    stop = min(info["offset"] + int(0.4 * FPS), motion.frames - 1) + 1
    clip = motion.slice(start, stop)
    clip.root[:, [0, 2]] -= clip.root[0, [0, 2]]
    return clip, (info["stroke"] - start) / FPS


def t2m_generate(text: str, seed: int) -> str:
    body = json.dumps({"text": text, "seed": seed, **T2M_SAMPLING}).encode()
    request = urllib.request.Request(T2M_SERVICE, data=body, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        result = json.loads(response.read())
    return (T2M_RESULTS / result["bvh_path"]).read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--canonical", required=True, help="T2M 出力の BVH(標準骨格の見本)")
    parser.add_argument("--preview", help="確認用に各クリップの BVH を書き出す先")
    parser.add_argument("--skip-t2m", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    canonical_motion = parse_bvh(Path(args.canonical).read_text())
    canonical = canonical_motion.skeleton
    labels = load_labels()

    clips: dict[str, Motion] = {}
    entries = []
    sources = []
    for path in sorted(MOCAP_DIR.glob("*.bvh")):
        raw = parse_bvh(path.read_text())
        raw, repaired = repair_glitches(raw)
        sources.append((path, resample(to_canonical(raw, canonical), FPS), repaired))

    # 立ち姿: 会話画面の待機姿勢。指だけは収録の自然に曲げた形(各収録の 1 フレーム目の平均)
    fingers = qmean(np.stack([motion.quats[0] for _, motion, _ in sources]), axis=0)
    neutral = viewer_neutral(canonical, fingers)
    # 腰の高さ: T2M 出力の先頭フレームと同じ接地(最も低い関節の高さ)にそろえる。
    # 会話画面は動作の 1 フレーム目の腰を「立っているときの腰」とみなし、待機に戻るときは
    # 足を床に着ける。ここがずれていると、ターンのたびに浮いて沈む。
    canonical_positions, _ = forward_kinematics(canonical_motion.slice(0, 1))
    lowest = float(canonical_positions[0, :, 1].min())
    at_zero, _ = forward_kinematics(Motion(canonical, np.zeros((1, 3)), neutral[None], FPS))
    root_height = lowest - float(at_zero[0, :, 1].min())

    for path, source, repaired in sources:
        number = path.stem
        catalog_id = f"MOT_{700 + int(number)}"
        motion = onto_neutral(source, source.quats[0], neutral)
        info = analyse(motion)
        clip, stroke = trim(motion, info)
        clips[catalog_id] = clip
        entries.append({
            "id": catalog_id, "label": labels.get(number, ""), "source": "mocap",
            "sourcePath": str(path.relative_to(SOURCE)),
            "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration": round(clip.duration, 3), "stroke": round(stroke, 3),
            "groups": info["groups"], "repairedFrames": repaired, "quality": quality(clip),
            "reference": "clipStartOntoViewerNeutral",
        })
        print(f"{catalog_id} {clip.duration:5.2f}s stroke={stroke:4.2f}s repaired={repaired:3d} "
              f"active={[g for g, v in info['groups'].items() if v['active']]} {labels.get(number, '')[:24]}")

    if not args.skip_t2m:
        for catalog_id, text in T2M_ONLY.items():
            best = None
            for seed in range(1, T2M_CANDIDATES + 1):
                started = time.time()
                motion = resample(parse_bvh(t2m_generate(text, seed)), FPS)
                motion = to_canonical(motion, canonical)
                motion, repaired = repair_glitches(motion)
                # T2M は左右を取り違えやすい(「右手を上げる」で左手が上がる)。
                # 本番の motion_handedness と同じく、鏡映した版も候補に入れる。
                for mirrored, candidate in ((False, motion), (True, mirror_motion(motion))):
                    info = analyse(candidate)
                    q = quality(candidate)
                    arm = info["groups"]["right_arm"]["rangeDeg"]
                    other = info["groups"]["left_arm"]["rangeDeg"]
                    # しゃがみ込み・移動・跳びは減点。右腕が動き、左腕は大きく動かない
                    score = (arm - other - 400 * max(q["hipDropM"] - 0.03, 0) - 200 * q["rootTravelM"]
                             - 2 * max(q["maxAngularSpeed"] - 15, 0))
                    ok = arm >= 30 and other <= 0.4 * arm and q["hipDropM"] < 0.08 and q["rootTravelM"] < 0.25
                    print(f"  {catalog_id} seed={seed} mirrored={mirrored} right={arm:.0f}deg left={other:.0f}deg "
                          f"{q} score={score:.1f} ok={ok} ({time.time() - started:.1f}s)")
                    if ok and (best is None or score > best[0]):
                        best = (score, seed, candidate, info, q, repaired, mirrored)
            if best is None:
                print(f"{catalog_id}: 合格候補なし(T2M のまま本番に出さない)")
                continue
            score, seed, motion, info, q, repaired, mirrored = best
            motion = onto_neutral(motion, motion.quats[0], neutral)
            info = analyse(motion)
            clip, stroke = trim(motion, info)
            clips[catalog_id] = clip
            entries.append({
                "id": catalog_id, "label": text, "source": "t2m", "t2mSeed": seed, "t2mSampling": T2M_SAMPLING,
                "t2mMirrored": mirrored,
                "duration": round(clip.duration, 3), "stroke": round(stroke, 3), "groups": info["groups"],
                "repairedFrames": repaired, "quality": quality(clip), "candidates": T2M_CANDIDATES,
                "reference": "clipStartOntoViewerNeutral",
            })

    order = [e["id"] for e in entries]
    lengths = np.asarray([clips[i].frames for i in order])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez_compressed(
        output / "library.npz",
        root=np.concatenate([clips[i].root for i in order]).astype(np.float32),
        quats=np.concatenate([clips[i].quats for i in order]).astype(np.float32),
        offsets=offsets, neutral=neutral.astype(np.float32),
    )
    for entry, start in zip(entries, offsets[:-1]):
        entry["frameStart"] = int(start)
    neutral_motion = Motion(canonical, np.array([[0.0, root_height, 0.0]]), neutral[None], FPS)
    (output / "skeleton.bvh").write_text(write_bvh(neutral_motion))
    catalog = {"version": 1, "fps": FPS, "rootHeight": root_height, "clips": entries,
               "builtAt": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "MOT_7xx は Custom-T2M-Lab 0010/01 のモーキャプ。t2m は崩れ検出で選んだ生成。"}
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + "\n")
    if args.preview:
        preview = Path(args.preview)
        preview.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            (preview / f"{entry['id']}.bvh").write_text(write_bvh(clips[entry["id"]]))
    print(f"{len(entries)} clips → {output}")


if __name__ == "__main__":
    main()
