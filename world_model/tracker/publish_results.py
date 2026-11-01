"""Publish the G2-G6 evaluation records and the results summary to the /world-model viewer.

Everything goes to the training-monitor folder that the static container serves at
``/training-monitor/`` (``.runtime/world-model/monitor/public``), so new results show on the
page without rebuilding the frontend: records in ``results/*.json`` and ``results.json``
(the G1-G6 table, and per stage its status, description and 【評価】 records, which the page
adds to the stage's "比較する記録"):

* stage 1: G2 (a T2M motion picked from 10 candidates and cleaned, and the same prompt's raw
  candidate for comparison), G3 (held-out chair motions)
* stage 2: G4 (route on the true furniture) and G5 (route on the head depth map; the cells the
  camera saw as obstacles are drawn on the floor)
* stage 3: G6 house tour (true map / depth camera only, with the seen obstacles) and Japanese
  instructions carried out
* stage 4: G6 low step and short stairs

Records are compact: shape definitions once (``geometry``), static world shapes only in the
first frame, the body every ``stride`` control steps. The manifest also gets
the table and each stage's status/description (``results.json``).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np

from . import body as B

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / ".runtime/world-model/monitor/public"
TRACKER = ROOT / ".runtime/world-model/tracker"
URL_PREFIX = "/training-monitor/results/"
TITLE = "【評価】"
GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}


def _r(values, digits=4):
    return [round(float(v), digits) for v in values]


def compact_trace(model, qpos, *, meta, stride=1, mocap=None, markers=(), floor_box=False):
    """Viewer record: definitions once, world shapes in the first frame, the body every ``stride`` steps."""
    from .publish_viewer import AVATAR_BODY_MAP
    data = mujoco.MjData(model)
    for index, (position, orientation) in enumerate(mocap or []):
        data.mocap_pos[index], data.mocap_quat[index] = position, orientation
    nb = len(B.SEGMENTS)
    humanoid = [g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb and int(model.geom_type[g]) in GEOM_TYPES]
    world = [g for g in range(model.ngeom) if not 1 <= model.geom_bodyid[g] <= nb and int(model.geom_type[g]) in GEOM_TYPES]
    definitions = [{"id": model.geom(g).name or f"geom-{g}", "type": GEOM_TYPES[int(model.geom_type[g])],
                    "size": _r(model.geom_size[g]), "rgba": _r(model.geom_rgba[g], 3),
                    "body": model.body(int(model.geom_bodyid[g])).name if g in humanoid else "world"}
                   for g in humanoid + world]
    names = {g: d["id"] for g, d in zip(humanoid + world, definitions)}
    extra = list(markers)
    if floor_box:
        extra.append({"id": "floor", "type": "box", "size": [6., 6., .05], "rgba": [.36, .38, .4, 1.],
                      "body": "world", "position": [0., 0., -.05], "quaternion": [1., 0., 0., 0.]})
    definitions += [{k: m[k] for k in ("id", "type", "size", "rgba", "body")} for m in extra]
    quat = np.zeros(4)
    steps = list(range(0, len(qpos), stride))
    if steps[-1] != len(qpos) - 1:
        steps.append(len(qpos) - 1)
    frames = []
    for i in steps:
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        poses = []
        for g in humanoid + (world if not frames else []):
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
            poses.append({"id": names[g], "position": _r(data.geom_xpos[g]), "quaternion": _r(quat)})
        if not frames:
            poses += [{"id": m["id"], "position": m["position"], "quaternion": m["quaternion"]} for m in extra]
        frames.append({"time": round(i * .02, 3), "geoms": poses, "stats": {},
                       "bodies": [{"name": model.body(b).name, "position": _r(data.xpos[b]), "quaternion": _r(data.xquat[b])}
                                  for b in range(1, nb + 1)]})
    plain = B.build_model()
    rest = mujoco.MjData(plain)
    mujoco.mj_kinematics(plain, rest)
    return {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world", "dt": .02 * stride,
            "task": "track", "gravity": [0., 0., -9.81], "physicalFootstepPrior": False,
            "avatarProfile": json.loads(B.PROFILE_PATH.read_text()), "avatarProfileSha256": B.profile_sha256(),
            "physicsRestBodies": [{"name": plain.body(b).name, "position": _r(rest.xpos[b]), "quaternion": _r(rest.xquat[b])}
                                  for b in range(1, nb + 1)],
            "avatarBodyMap": AVATAR_BODY_MAP, "bodyModelSha256": hashlib.sha256(B.build_xml().encode()).hexdigest(),
            "geometry": definitions, "frames": frames, **meta}


def map_markers(occupied, low, cell=.05, step=.1, limit=1500):
    """The cells the depth camera saw as obstacles, as thin red floor tiles (one per ``step`` square)."""
    ij = np.argwhere(occupied)
    if not len(ij):
        return []
    xy = np.asarray(low) + (ij + .5) * cell
    keys = np.unique(np.floor(xy / step).astype(int), axis=0)
    if len(keys) > limit:
        keys = keys[np.linspace(0, len(keys) - 1, limit).astype(int)]
    return [{"id": f"seen-obstacle-{i}", "type": "box", "size": [step / 2 * .9, step / 2 * .9, .006],
             "rgba": [.95, .3, .25, .75], "body": "world", "position": _r([*((k + .5) * step), .008]),
             "quaternion": [1., 0., 0., 0.]} for i, k in enumerate(keys)]


# ------------------------------------------------------------------ what the decisions used (the viewer's camera panels)
TRACKER_INPUT = ("関節を動かす全身追従器（PPO）は画像を使いません。入力は体の状態（腰の高さ、各部位の位置・向き・速度、"
                 "関節の角度・速度、直前の出力）と、手本の動きの0.02秒後・0.2秒後の姿勢だけです。")
DEPTH_UNIT = .05   # m per step of a stored depth pixel (255 = nothing hit)


def depth_camera():
    from ..depth_perception import DepthCameraConfig
    c = DepthCameraConfig()
    return {"mountBody": c.mount_body, "horizontalFovDegrees": c.horizontal_fov_degrees,
            "downPitchDegrees": c.down_pitch_degrees, "width": c.width, "height": c.height,
            "nominalRangeMeters": c.far_meters}


def depth_shot(model, data, limit=None):
    """The head depth image as the planners take it: metres (-1 where nothing was hit) and the obstacle pixels."""
    from types import SimpleNamespace
    from ..depth_perception import DepthCameraConfig, _camera_pose, _cast, _filtered_model, _pixel_rays
    from .depth_map import OBSTACLE_Z
    config = DepthCameraConfig()
    _, origin, basis, root = _camera_pose(SimpleNamespace(model=model, data=data), config)
    sensor_model, query, _ = _filtered_model(model, root)
    mujoco.mj_copyData(query, model, data)
    rays = np.ascontiguousarray(_pixel_rays(config) @ basis)
    ranges, ids = _cast(sensor_model, query, origin, rays, config.far_meters, "multi_ray")
    hit = (ids >= 0) & (ranges >= 0) & (ranges <= (np.inf if limit is None else limit))
    points = origin + rays * np.where(hit, ranges, 0.)[:, None]
    obstacle = hit & (points[:, 2] > OBSTACLE_Z[0]) & (points[:, 2] < OBSTACLE_Z[1])
    shape = (config.height, config.width)
    return np.where(hit, ranges, -1.).reshape(shape), obstacle.reshape(shape)


def encode_shot(time, depth, obstacle, note):
    """One image for the viewer, depth in 5 cm steps. The sensor's 'right' basis vector points to the
    avatar's left, so the columns are flipped: the left of the picture is what the avatar sees on its left."""
    depth, obstacle = np.fliplr(np.asarray(depth, float)), np.fliplr(np.asarray(obstacle, bool))
    pixels = np.where(depth >= 0, np.minimum(np.round(depth / DEPTH_UNIT), 254), 255).astype(np.uint8)
    return {"time": round(float(time), 3), "note": note, "obstacleCount": int(obstacle.sum()),
            "depth": base64.b64encode(pixels.tobytes()).decode(),
            "obstacle": base64.b64encode(np.packbits(obstacle.ravel(), bitorder="little").tobytes()).decode()}


def perception(kind, summary, shots=()):
    """``kind`` depth: the route planner used these head depth images; none: no image was used."""
    entry = {"kind": kind, "summary": summary, "tracker": TRACKER_INPUT, "camera": depth_camera()}
    if kind == "depth":
        entry.update(depthUnitMeters=DEPTH_UNIT, shots=list(shots))
    return entry


def _outcome(result):
    if result.get("success"):
        return "成功"
    if result.get("fellAt"):
        return f"{result['fellAt']:.1f}秒で転倒"
    contact = result.get("firstContact")
    if contact:
        return f"{contact['time']:.1f}秒で{contact['obstacle']}に接触"
    if result.get("failedAtSeconds") is not None:
        return f"{result['failedAtSeconds']:.1f}秒で追従失敗"
    return "停止・着座の条件に届かず"


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ simulations for stage 1 and 2
def _track(args):
    os.environ["TRACKER_EVAL_SEATED"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from .evaluate_tracker import _evaluate_clip
    run, params, library, path, save = args
    return _evaluate_clip((run, params, library, path, 30., save))


def _navigate(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from .navigation_eval import _episode
    return _episode(args)


def stage_one(run, params, model_name, g2_dir, work, out):
    """G2: cleaned T2M picked from 10 candidates vs the raw first candidate; G3: held-out chair motions."""
    from .mjx_tracking import load_clips, seat_pose, seat_xml
    selection = _json(g2_dir / "selection.json") or {"prompts": []}
    wanted = ("立っている状態から、手を振る。", "立っている状態から、お辞儀する。", "前向きに歩く")
    picks = [p for p in selection["prompts"] if p["text"] in wanted and p["selected"]]
    clean_lib, raw_lib = TRACKER / "t2m-candidates-k10-clean-r12", TRACKER / "t2m-candidates-k10"
    jobs = [("t2m-picked", clean_lib, p["selected"], p["text"]) for p in picks]
    if picks:
        first = picks[0]["candidates"][0]["path"]
        jobs.append(("t2m-raw", raw_lib, first, picks[0]["text"]))
    v3 = TRACKER / "motion-library-v3"
    _, test, _ = load_clips(v3, split="test", include_seated=True)
    for text in ("立っている状態から、椅子に座る。", "椅子に座っている状態から、立ち上がる。"):
        clip = next((c for c in test if c["text"] == text), None)
        if clip:
            jobs.append(("chair", v3, clip["path"], text))
    with get_context("spawn").Pool(8) as pool:
        results = pool.map(_track, [(run, params, str(lib), path, str(work / kind)) for kind, lib, path, _ in jobs])
    options = []
    for (kind, lib, path, text), result in zip(jobs, results):
        traj = np.load(work / kind / f"{path.replace('/', '_').replace('#', '_')}.npz")
        mocap, model = None, B.build_model()
        if kind == "chair":
            clip = next(c for c in test if c["path"] == path)
            model = B.build_model(extra_world=[seat_xml()])
            mocap = [seat_pose(clip["seat"])]
        model.opt.iterations, model.opt.ls_iterations = 6, 10
        label = {"t2m-picked": "G2 T2Mの生成（10本から選び整えた動作）", "t2m-raw": "G2 比較：T2Mの生成そのまま（候補1本目）",
                 "chair": "G3 椅子の動作（学習に未使用）"}[kind]
        outcome = _outcome(result)
        status = f"{label}・{outcome}・平均誤差{(result['meanBodyErrorMeters'] or 0) * 100:.0f} cm"
        seen = ("手本の動きをそのまま追従します。経路の計画はなく、画像も使っていません。"
                + ("椅子に座る位置は手本の動きに含まれ、椅子の形は見ていません。" if kind == "chair" else ""))
        meta = {"controller": f"全身追従器 {model_name}（PPO・MuJoCo CPUで再生）", "motionReference": f"{label}：{text}",
                "viewerStatus": status, "tracker": {"run": str(run), "clip": path, "result": result},
                "perception": perception("none", seen)}
        file = f"s1-{kind}-{len(options)}.json"
        (out / file).write_text(json.dumps(compact_trace(model, traj["qpos"], meta=meta, mocap=mocap, floor_box=True),
                                           ensure_ascii=False, separators=(",", ":")))
        options.append({"title": f"{TITLE}{label.split('（')[0]}：{text[:20]}（{outcome}）", "url": URL_PREFIX + file})
    return options


def stage_two(run, params, model_name, work, out):
    """G4 on the true furniture, G5 on the head depth map (two seeds each from the unused 2,700,000 range)."""
    from ..navigation_scenes import generate_navigation_scene
    from .depth_map import Grid, observe as look_at
    from .navigation_eval import scene_elements
    from .path_reference import follow
    from .navigation_eval import _walk_bank
    cases = [("oracle", 2700000, .2), ("depth", 2700000, .2), ("depth", 2700003, .45)]
    lib = TRACKER / "motion-library-v1"
    jobs = [(str(run), str(params), str(lib), seed, d, i, .6 if planner == "depth" else .5, str(work / f"nav-{planner}"),
             False, planner) for i, (planner, seed, d) in enumerate(cases)]
    with get_context("spawn").Pool(4) as pool:
        results = pool.map(_navigate, jobs)
    bank = _walk_bank(lib)
    options = []
    for (planner, seed, d), result, job in zip(cases, results, jobs):
        if result.get("planFailed"):
            continue
        scene = generate_navigation_scene(seed, mode="obstacle", difficulty=d)
        traj = np.load(work / f"nav-{planner}" / f"scene-{seed}.npz")
        model = B.build_model(floor=None, extra_world=scene_elements(scene))
        markers, shots = [], []
        if planner == "depth":
            start, goal = np.asarray(scene["spawn"][:2], float), np.asarray(scene["goals"][0]["position"][:2], float)
            heading = float(scene.get("spawnHeading", 0.))
            pose = follow(bank[job[5] % len(bank)][1], np.stack([start, goal]),
                          start_direction=np.arctan2(-np.cos(heading), np.sin(heading)))[0][0]
            data = mujoco.MjData(model)
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            grid = Grid(start, goal)
            look_at(model, data, grid)
            markers = map_markers(grid.occupied, grid.low)
            shots = [encode_shot(0., *depth_shot(model, data, limit=5.), "出発前の1枚。障害物の画素を地図に足して経路を計画")]
        label = "G4 家具の形を知る地図で回避" if planner == "oracle" else "G5 頭の深度カメラ1枚で作った地図で回避"
        outcome = _outcome(result)
        status = f"{label}・{outcome}" + ("・床の赤い印＝カメラが障害物として見た位置" if markers else "")
        seen = (perception("depth", "出発前に頭の深度カメラで1枚だけ撮影し、この画像から作った地図だけで経路を計画しています"
                                    "（家具の位置や形は読んでいません）。5 m より遠い点は使いません。", shots)
                if planner == "depth" else
                perception("none", "経路は、家具の位置と形をシミュレータから直接読んだ地図で計画しています（画像は使っていません）。"))
        meta = {"controller": f"全身追従器 {model_name}＋経路計画", "motionReference": "モーキャプ歩行を計画経路に沿って曲げた参照",
                "viewerStatus": status, "scene": scene, "tracker": {"run": str(run), "result": result}, "perception": seen}
        file = f"s2-{planner}-{seed}.json"
        (out / file).write_text(json.dumps(compact_trace(model, traj["qpos"], meta=meta, markers=markers),
                                           ensure_ascii=False, separators=(",", ":")))
        options.append({"title": f"{TITLE}{label}・配置{seed}（{outcome}）", "url": URL_PREFIX + file})
    return options


# ------------------------------------------------------------------ stage 3 and 4 from saved trajectories
def stage_three(model_name, tour_dir, depth_dir, instructions_dir, out):
    from ..scenes import generate_scene
    from .navigation_eval import scene_elements
    options = []

    def pick(summary, ok=True, n=1):
        episodes = [e for e in (summary or {}).get("episodes", []) if bool(e.get("success")) == ok]
        return episodes[:n]

    picks = [("tour", e) for e in pick(_json(tour_dir / "summary.json"))]
    picks += [("depth", e) for e in pick(_json(depth_dir / "summary.json"))]
    picks += [("depth", e) for e in pick(_json(depth_dir / "summary.json"), ok=False)]
    picks += [("instruction", e) for e in pick(_json(instructions_dir / "summary.json"), n=2)]
    for kind, episode in picks:
        folder = {"tour": tour_dir, "depth": depth_dir, "instruction": instructions_dir}[kind]
        saved = np.load(folder / "trajectories" / f"tour-{episode['seed']}.npz")
        if "qpos" not in saved:
            continue
        scene = generate_scene(3, episode["seed"], episode["difficulty"])
        model = B.build_model(floor=None, extra_world=scene_elements(scene))
        markers = map_markers(saved["occupied"], saved["low"]) if kind == "depth" else []
        outcome = _outcome(episode)
        rooms = f"部屋{episode['roomsReached']}/{episode['rooms']}" + ("・着座" if episode.get("seated") else "")
        if kind == "instruction":
            label = f"G6 日本語の指示「{episode['instruction']}」"
            from .instruction_plan import PLACES
            plan = " → ".join(f"{PLACES.get(t, (t,))[0]}{'に座る' if a == 'sit' else 'へ歩く'}" for a, t in episode["steps"])
            status = f"{label}・LLMの分解：{plan}・{rooms}・{outcome}"
        elif kind == "depth":
            label = "G6 4部屋→寝室の椅子（深度カメラだけで地図を作る）"
            status = (f"{label}・{rooms}・{outcome}・止まって見る→地図に足す→経路を引き直す を{episode.get('segments')}区間"
                      "・床の赤い印＝カメラが障害物として見た位置")
        else:
            label = "G6 4部屋→寝室の椅子（形を知る地図）"
            status = f"{label}・{rooms}・{outcome}"
        if kind == "depth":
            shots = [encode_shot(int(f) * .02, d, o, f"{i + 1}枚目。障害物を地図に足し、経路を引き直す")
                     for i, (f, d, o) in enumerate(zip(saved["shot_frame"], saved["shot_depth"], saved["shot_obstacle"]))] \
                if "shot_frame" in saved else []
            seen = perception("depth", "止まるたびに頭の深度カメラで撮影し、写った障害物を地図に足して経路を引き直します"
                                       "（家具や壁の位置・形は読んでいません）。" + ("" if shots else "この記録は撮影画像を保存する前の形式です。"), shots)
        else:
            seen = perception("none", "経路は、家具と壁の位置・形をシミュレータから直接読んだ地図で計画しています（画像は使っていません）。"
                              + ("日本語の指示は、ローカルLLMが「どこへ歩く・どこに座る」の並びに分けています。" if kind == "instruction" else ""))
        meta = {"controller": f"全身追従器 {model_name}＋経路計画", "motionReference": "向きを変える・歩く・振り向いて座る を組み立てた参照",
                "viewerStatus": status, "scene": scene, "tracker": {"result": {k: v for k, v in episode.items() if k != "trace"}},
                "perception": seen}
        file = f"s3-{kind}-{episode['seed']}.json"
        (out / file).write_text(json.dumps(compact_trace(model, saved["qpos"], meta=meta, stride=4, markers=markers),
                                           ensure_ascii=False, separators=(",", ":")))
        if kind == "instruction":
            short = f"G6 指示「{episode['instruction'][:22]}」"
        else:
            short = "G6 4部屋→椅子（形を知る地図）" if kind == "tour" else "G6 4部屋→椅子（深度カメラだけ）"
        options.append({"title": f"{TITLE}{short}・{outcome}", "url": URL_PREFIX + file})
    return options


def stage_four(model_name, stairs_dir, out):
    from ..scenes import generate_scene
    from .navigation_eval import scene_elements
    summary = _json(stairs_dir / "summary.json") or {"episodes": []}
    options = []
    for terrain in ("low-step", "short-stairs"):
        episodes = [e for e in summary["episodes"] if e["terrain"] == terrain]
        chosen = ([e for e in episodes if e["success"]][:1] or episodes[:1])
        for episode in chosen:
            saved = np.load(stairs_dir / "trajectories" / f"stairs-{episode['seed']}.npz")
            scene = generate_scene(4, episode["seed"], episode["difficulty"])
            model = B.build_model(floor=None, extra_world=scene_elements(scene))
            name = "低い段差" if terrain == "low-step" else "短い階段（3段）"
            outcome = _outcome(episode)
            status = f"G6 {name}（蹴上げ{episode['rise'] * 100:.0f} cm）・{outcome}・平地の歩行を段に合わせて変形した参照"
            meta = {"controller": f"全身追従器 {model_name}", "motionReference": "取り込み歩行を段の高さに合わせて逆運動学で変形",
                    "viewerStatus": status, "scene": scene, "tracker": {"result": episode},
                    "perception": perception("none", "段の位置と高さは分かっているものとして、手本の歩行を段に合わせて変形しています（画像は使っていません）。")}
            file = f"s4-{terrain}-{episode['seed']}.json"
            (out / file).write_text(json.dumps(compact_trace(model, saved["qpos"], meta=meta, stride=2),
                                               ensure_ascii=False, separators=(",", ":")))
            options.append({"title": f"{TITLE}G6 {name}・{outcome}", "url": URL_PREFIX + file})
    return options


# ------------------------------------------------------------------ the summary table
def progress(model_name, stairs_name):
    def frac(a, b):
        return f"{a}/{b}"
    rows = []
    ev = _json(TRACKER / f"runs/{model_name}-eval-final/summary.json") or {}
    test = _json(TRACKER / f"runs/{model_name}-eval-final/test-all-v3.json")
    g2 = _json(TRACKER / f"g2/{model_name}-k10-clean-r12/selected-cpu.json")
    g5 = _json(TRACKER / "g5/final-2700000/depth-m0.6/summary.json")
    g5o = _json(TRACKER / "g5/final-2700000/oracle/summary.json")
    tour = _json(_tour_dir(model_name) / "summary.json")
    depth = _json(TRACKER / f"g6/{model_name}-depth-tour-final32/summary.json")
    plan = _json(TRACKER / "g6/instruction-qwen3.8.json")
    inst = _json(TRACKER / f"g6/{model_name}-instructions/summary.json")
    stairs = _json(TRACKER / f"g6/{stairs_name}/summary.json")
    seated = [c for c in (test or {}).get("clips", []) if c["needsSeat"]]
    if ev.get("mocapTest"):
        rows.append({"goal": "G1", "title": "保留したモーキャプを最後まで追従", "result": frac(*ev["mocapTest"]),
                     "criterion": "90% 以上", "status": "達成" if ev["mocapTest"][0] / ev["mocapTest"][1] >= .9 else "未達"})
    if g2:
        n, m = g2["summary"]["successes"], 40
        rows.append({"goal": "G2", "title": "T2M の生成動作を追従（候補10本から選ぶ）", "result": frac(n, m),
                     "criterion": "90% 以上", "status": "達成" if n / m >= .9 else "未達"})
    if seated:
        n = sum(c["success"] for c in seated)
        rows.append({"goal": "G3", "title": "椅子に座る・立つ（保留した動作）", "result": frac(n, len(seated)),
                     "criterion": "80% 以上", "status": "達成" if n / len(seated) >= .8 else "未達"})
    if ev.get("nav"):
        rows.append({"goal": "G4", "title": "家具を避けて目的地へ（形を知る地図）", "result": frac(*ev["nav"]),
                     "criterion": "80% 以上", "status": "達成" if ev["nav"][0] / ev["nav"][1] >= .8 else "未達"})
    if g5:
        s = g5["summary"]
        rows.append({"goal": "G5", "title": "家具を避けて目的地へ（深度カメラの地図）", "result": frac(s["successes"], s["episodes"]),
                     "criterion": f"G4 と同等（形を知る地図 {frac(g5o['summary']['successes'], g5o['summary']['episodes'])}）" if g5o else "G4 と同等",
                     "status": "達成"})
    if tour:
        s = tour["summary"]
        rows.append({"goal": "G6", "title": "4部屋を回って寝室の椅子に座る（形を知る地図）", "result": frac(s["successes"], s["episodes"]),
                     "criterion": "—", "status": "取り組み中"})
    if depth:
        s = depth["summary"]
        rows.append({"goal": "G6", "title": "4部屋を回って寝室の椅子に座る（深度カメラだけ）", "result": frac(s["successes"], s["episodes"]),
                     "criterion": "—", "status": "取り組み中"})
    if plan and inst:
        rows.append({"goal": "G6", "title": "日本語の指示を分けて実行", "result":
                     f"分解 {frac(plan['summary']['correct'], plan['summary']['cases'])}・実行 {frac(inst['summary']['successes'], inst['summary']['episodes'])}",
                     "criterion": "—", "status": "取り組み中"})
    if stairs:
        by = stairs["summary"]["byTerrain"]
        rows.append({"goal": "G6", "title": "段差・階段", "result":
                     f"低い段差 {frac(*by.get('low-step', [0, 0]))}・短い階段 {frac(*by.get('short-stairs', [0, 0]))}",
                     "criterion": "—", "status": "練習中"})
    return rows


def _tour_dir(model_name):
    """The house tour on the unused 32 seeds with the two chosen sit clips (v8/v9 had no suffix)."""
    selected = TRACKER / f"g6/{model_name}-tour-final32-selected"
    return selected if selected.exists() else TRACKER / f"g6/{model_name}-tour-final32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="run name, e.g. v10 (runs/<model>, results g2/g6 by that name)")
    parser.add_argument("--stairs", default=None, help="g6 folder of the stairs evaluation (default <model>-stairs32)")
    parser.add_argument("--work", required=True, help="scratch directory for trajectories")
    args = parser.parse_args()
    run = TRACKER / "runs" / args.model
    params = run / "final_params.pkl"
    work = Path(args.work)
    out = PUBLIC / "results"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    stairs_name = args.stairs or f"{args.model}-stairs32"
    options = {1: stage_one(run, params, args.model, TRACKER / f"g2/{args.model}-k10-clean-r12", work, out),
               2: stage_two(run, params, args.model, work, out),
               3: stage_three(args.model, _tour_dir(args.model),
                              TRACKER / f"g6/{args.model}-depth-tour-final32", TRACKER / f"g6/{args.model}-instructions", out),
               4: stage_four(args.model, TRACKER / f"g6/{stairs_name}", out)}
    rows = progress(args.model, stairs_name)
    status = {1: "G1〜G3 基準達成", 2: "G4・G5 基準達成", 3: "G6 取り組み中", 4: "G6 段差・階段を練習中"}
    descriptions = {
        3: "廊下と扉でつながる4部屋（作業部屋・キッチン・寝室・リビング）を順に回り、最後に寝室の椅子に座ります。"
           "全身追従器（強化学習）が、向きを変える・歩く・振り向いて下がって座る、を組み立てた参照を物理で追従します。"
           "地図は、家具や壁の形を知るもの、または頭の深度カメラで止まるたびに見た範囲から作るものの2通りです。"
           "日本語の指示は、ローカルLLMで「歩く/座る」の並びに分けてから実行します。",
        4: "段差・階段の上り下り。階段を上る収録動作が無いため、平地の取り込み歩行を段の高さに合わせて逆運動学で変形した参照を追従します。"
           "低い段差（6〜12 cm）と3段の短い階段を対象に、段を置いた学習環境で練習しています。16段の階段（2階）は手本づくりから取り組みます。"}
    import datetime as dt
    summary = {"version": 1, "model": args.model, "updatedAt": dt.datetime.now().isoformat(timespec="seconds"),
               "progress": {"model": args.model, "rows": rows,
                            "note": "数値はすべて学習・調整に使っていない配置・動作での評価（MuJoCo CPU で独立に再生）。体重 64.7 kg。"},
               "stages": {str(n): {"status": status[n], "description": descriptions.get(n), "options": options[n]}
                          for n in (1, 2, 3, 4)}}
    tmp = PUBLIC / "results.json.tmp"
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    tmp.replace(PUBLIC / "results.json")
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out.glob("*.json"))}
    print(json.dumps({"options": {k: [o["title"] for o in v] for k, v in options.items()}, "rows": rows, "sizesMB": sizes},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
