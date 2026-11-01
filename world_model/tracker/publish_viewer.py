"""Publish tracker rollouts to the /world-model viewer (frontend/public/world-model).

Each selected clip or navigation scene is simulated in native MuJoCo with the
given checkpoint (exactly as in ``evaluate_tracker`` / ``navigation_eval``),
and the per-frame body and geometry poses are written as a complete-world
record. Records carry a bone-to-body map so the viewer drives the gakucho GLB
from the 22 tracking bodies. Earlier tracker entries in the manifest are
replaced; all other manifest entries are left as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np

from . import body as B

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "frontend/public/world-model"
URL_PREFIX = "/world-model/tracker/"
TITLE_PREFIX = "【新・全身追従】"

AVATAR_BODY_MAP = {
    "hips": "pelvis", "spine": "spine", "spine1": "spine", "spine2": "spine2", "spine3": "spine2",
    "spine4": "chest", "neck": "neck", "neck1": "neck", "head": "head",
    **{f"{side}{bone}": f"{side}_{body}" for side in ("left", "right") for bone, body in (
        ("Shoulder", "clavicle"), ("UpperArm", "upper_arm"), ("LowerArm", "forearm"), ("Hand", "hand"),
        ("UpLeg", "thigh"), ("LowerLeg", "shin"), ("Foot", "foot"), ("Toe", "toe"))},
}
GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_PLANE): "plane"}

MOCAP_LABELS = ("001", "002", "011", "012", "014", "024", "055", "080", "087", "095")
T2M_INDICES = ("000", "005", "007")


def _round(values, digits=5):
    return [round(float(v), digits) for v in values]


def trace_from_qpos(model, qpos, *, meta, floor_box=True, mocap=None):
    """mocap: (位置, 四元数) の組の並び。椅子など mocap で置いた物体を記録に含めるときに渡す。"""
    data = mujoco.MjData(model)
    nb = len(B.SEGMENTS)
    quat = np.zeros(4)
    for index, (position, orientation) in enumerate(mocap or []):
        data.mocap_pos[index], data.mocap_quat[index] = position, orientation

    def frame(q, time):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        bodies = [{"name": model.body(b).name, "position": _round(data.xpos[b]), "quaternion": _round(data.xquat[b])}
                  for b in range(1, nb + 1)]
        geoms = []
        for g in range(model.ngeom):
            kind = GEOM_TYPES.get(int(model.geom_type[g]))
            if kind is None or (kind == "plane" and floor_box):
                continue
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
            # 椅子など mocap で置いた物体は身体ではなく環境(world)として扱う(アバター表示で隠されないように)
            body_id = int(model.geom_bodyid[g])
            body = model.body(body_id).name if body_id and model.body_mocapid[body_id] < 0 else "world"
            geoms.append({"id": model.geom(g).name, "type": kind, "position": _round(data.geom_xpos[g]),
                          "quaternion": _round(quat), "size": _round(model.geom_size[g]),
                          "rgba": _round(model.geom_rgba[g], 3), "body": body})
        if floor_box:
            geoms.append({"id": "floor", "type": "box", "position": [0., 0., -.05], "quaternion": [1., 0., 0., 0.],
                          "size": [6., 6., .05], "rgba": [.36, .38, .4, 1.], "body": "world"})
        return {"time": round(time, 4), "geoms": geoms, "bodies": bodies, "stats": {}}

    rest_data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, rest_data)
    rest = [{"name": model.body(b).name, "position": _round(rest_data.xpos[b]), "quaternion": _round(rest_data.xquat[b])}
            for b in range(1, nb + 1)]
    profile = json.loads(B.PROFILE_PATH.read_text())
    return {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world", "dt": .02,
            "task": "track", "gravity": [0., 0., -9.81], "physicalFootstepPrior": False,
            "avatarProfile": profile, "avatarProfileSha256": B.profile_sha256(),
            "physicsRestBodies": rest, "avatarBodyMap": AVATAR_BODY_MAP,
            "bodyModelSha256": hashlib.sha256(B.build_xml().encode()).hexdigest(),
            "frames": [frame(q, i * .02) for i, q in enumerate(qpos)], **meta}


def _track(args):
    from .evaluate_tracker import _evaluate_clip
    run, params, library, clip, save = args
    return _evaluate_clip((run, params, library, clip, 30., save))


def _navigate(args):
    from .navigation_eval import _episode
    return _episode(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--label", required=True, help="short checkpoint label shown in titles")
    parser.add_argument("--work", required=True, help="scratch directory for trajectories")
    args = parser.parse_args()
    work = Path(args.work)
    mocap_lib = ROOT / ".runtime/world-model/tracker/motion-library-v1"
    t2m_lib = ROOT / ".runtime/world-model/tracker/t2m-samples"
    manifest = json.loads((mocap_lib / "manifest.json").read_text())
    chosen = []
    for label in MOCAP_LABELS:
        clip = next((c for c in manifest["clips"] if c["status"] == "ok" and c["split"] == "test"
                     and c["dataset"] == "0001" and c["label"] == label and not c["needsSeat"]), None)
        if clip:
            chosen.append(("mocap", mocap_lib, clip))
    t2m = json.loads((t2m_lib / "manifest.json").read_text())["clips"]
    chosen += [("t2m", t2m_lib, c) for c in t2m if c["label"] in T2M_INDICES]
    cohort = json.loads((ROOT / ".runtime/world-model/tracker/nav-dev/codex-new32-scenes.json").read_text())
    nav = [e for e in cohort if e["r5Success"]][:2] + [e for e in cohort if not e["r5Success"]][:2]

    with get_context("spawn").Pool(16) as pool:
        tracked = pool.map(_track, [(args.run, args.params, str(lib), c["path"], str(work / kind))
                                    for kind, lib, c in chosen], chunksize=1)
        navigated = pool.map(_navigate, [(args.run, args.params, str(mocap_lib), e["sceneSeed"], e["difficulty"],
                                          i, .5, str(work / "nav"), False) for i, e in enumerate(nav)], chunksize=1)

    out = PUBLIC / "tracker"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    model = B.build_model()
    model.opt.iterations, model.opt.ls_iterations = 6, 10
    stage1, stage2 = [], []
    for (kind, lib, clip), result in zip(chosen, tracked):
        name = clip["path"].replace("/", "_")
        traj = np.load(work / kind / f"{name}.npz")
        seconds = (len(traj["qpos"]) - 1) / 50.
        source = "保留モーキャプ（学習に未使用）" if kind == "mocap" else "日本語T2Mの生成動作"
        outcome = (f"最後まで追従（{seconds:.1f}秒）" if result["success"]
                   else f"{result['failedAtSeconds']:.1f}秒で追従失敗（体節の平均誤差0.5 m超）")
        status = f"{source}・{outcome}・平均誤差{result['meanBodyErrorMeters'] * 100:.0f} cm"
        meta = {"controller": f"全身追従器 {args.label}（PPO・MuJoCo CPUで再生）",
                "motionReference": f"{source}：{clip['text']}", "viewerStatus": status,
                "tracker": {"run": args.run, "paramsSha256": hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
                            "clip": clip["path"], "result": result}}
        file = f"{kind}-{clip['label']}.json"
        (out / file).write_text(json.dumps(trace_from_qpos(model, traj["qpos"], meta=meta), ensure_ascii=False))
        stage1.append({"title": f"{TITLE_PREFIX}{'T2M' if kind == 't2m' else 'モーキャプ'}：{clip['text'][:22]}"
                                f"（{'成功' if result['success'] else '失敗'}）", "url": URL_PREFIX + file})
    from .navigation_eval import scene_elements
    from ..navigation_scenes import generate_navigation_scene
    for entry, result in zip(nav, navigated):
        scene = generate_navigation_scene(entry["sceneSeed"], mode="obstacle", difficulty=entry["difficulty"])
        traj = np.load(work / "nav" / f"scene-{entry['sceneSeed']}.npz")
        scene_model = B.build_model(floor=None, extra_world=scene_elements(scene))
        outcome = ("到着・停止（通過）" if result["success"] else
                   f"{result['fellAt']:.1f}秒で転倒" if result["fellAt"] else
                   f"{result['firstContact']['time']:.1f}秒で{result['firstContact']['obstacle']}に接触"
                   if result["firstContact"] else f"停止条件未達（目標まで{result['finalGoalDistance']:.2f} m）")
        status = (f"経路計画（全形状を知る特権情報）＋経路沿いの歩行参照＋追従・{outcome}"
                  f"／旧R4・5aaは{'成功' if entry['r5Success'] else '失敗'}")
        meta = {"controller": f"全身追従器 {args.label}＋経路計画（視覚は未使用）",
                "motionReference": "モーキャプ歩行を計画経路に沿って曲げた参照", "viewerStatus": status,
                "scene": scene, "tracker": {"run": args.run, "result": result}}
        trace = trace_from_qpos(scene_model, traj["qpos"], meta=meta, floor_box=False)
        file = f"nav-{entry['sceneSeed']}.json"
        (out / file).write_text(json.dumps(trace, ensure_ascii=False))
        stage2.append({"title": f"{TITLE_PREFIX}家具回避・配置{entry['sceneSeed']}（{'通過' if result['success'] else '未通過'}）",
                       "url": URL_PREFIX + file})

    manifest_path = PUBLIC / "manifest.json"
    viewer = json.loads(manifest_path.read_text())
    for stage in viewer["stages"]:
        options = [o for o in stage.get("traceOptions", []) if not o["url"].startswith(URL_PREFIX)]
        stage["traceOptions"] = options + (stage1 if stage["stage"] == 1 else stage2 if stage["stage"] == 2 else [])
    viewer["fullBodyTracker"] = {"checkpoint": args.label, "run": args.run,
                                 "note": "新しい全身追従器（22体節・63自由度、学長の実寸）の記録。旧20関節身体の記録とは別系統。"}
    manifest_path.write_text(json.dumps(viewer, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage1": stage1, "stage2": stage2}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
