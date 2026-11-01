"""Export four physical-world stages and keep task proficiency gates explicit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .environment import HumanoidRoomEnv
from .humanoid import build_humanoid_xml
from .scenes import generate_scene
from .training import dump_json, evaluate
from .body_contract import trace_body_metadata, assert_policy_body

TITLES = ["身体・重力・接触", "家具配置の変化", "複数の部屋と廊下", "段差・階段・2階"]


def record_unpowered(scene, path, seconds=3, avatar_profile=None):
    env = HumanoidRoomEnv(scene=scene, randomize=False, avatar_profile=avatar_profile)
    env.reset(seed=15000)
    trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world", "dt": .02, "controller": "unpowered",
             "task": "passive", "scene": scene, "gravity": env.model.opt.gravity.tolist(),
             "frames": [env.snapshot()]}
    trace.update(trace_body_metadata(env))
    for _ in range(round(seconds/.02)):
        env.data.ctrl[:] = 0
        for _ in range(10):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        trace["frames"].append(env.snapshot())
    dump_json(path, trace)
    env.close()


def verify_worlds(seeds=range(20), avatar_profile=None):
    reports = []
    for stage in range(1, 5):
        for seed in seeds:
            difficulty = (seed % 3 + .5) / 3 if stage == 4 else .5
            scene = generate_scene(stage, seed=seed, difficulty=difficulty)
            model = mujoco.MjModel.from_xml_string(build_humanoid_xml(scene, avatar_profile))
            assert np.isfinite(model.body_mass).all()
            assert model.opt.gravity[2] == -9.81
            assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
            assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), "Infinite ground hides falls/holes"
            env = HumanoidRoomEnv(scene=scene, randomize=False, max_episode_seconds=1,
                                  avatar_profile=avatar_profile)
            _, info = env.reset(seed=seed)
            initial_height = float(env.data.qpos[2])
            for _ in range(50):
                _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
                if terminated:
                    break
            reports.append({"stage": stage, "seed": seed, "difficulty": difficulty,
                            "geomCount": model.ngeom, "dynamicDofs": model.nv,
                            "initialHeight": initial_height, "finalHeight": float(env.data.qpos[2]),
                            "finiteState": bool(np.isfinite(env.data.qpos).all()),
                            "standingProbeFell": bool(terminated), "contactCount": int(env.data.ncon)})
            assert np.isfinite(env.data.qpos).all(), "Invalid physical state"
            assert not terminated and env.data.ncon > 0, "Spawn does not support the standing body"
            env.close()
    return reports


def export(output, policy_path=None, evaluation_path=None, verify=True, avatar_profile=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads(Path(evaluation_path).read_text()) if evaluation_path else None
    policy = None
    if policy_path:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path, device="cpu")
    contract_env = HumanoidRoomEnv(avatar_profile=avatar_profile)
    assert_policy_body(policy, contract_env)
    if evaluation:
        provenance = evaluation.get("provenance", {})
        if provenance.get("avatarProfileSha256") != contract_env.avatar_profile_sha256:
            raise ValueError("Standing evaluation belongs to a different avatar profile")
        if avatar_profile is not None and provenance.get("bodyModelSha256") != contract_env.body_model_sha256:
            raise ValueError("Standing evaluation belongs to a different physical body")
        if policy_path and provenance.get("checkpointSha256") != hashlib.sha256(Path(policy_path).read_bytes()).hexdigest():
            raise ValueError("Standing evaluation belongs to a different policy checkpoint")
    contract_env.close()
    reports = verify_worlds(avatar_profile=avatar_profile) if verify else []
    manifest = {"version": 1, "physics": "MuJoCo", "gravity": [0, 0, -9.81],
                "coordinateSystem": "Z-up", "display": "recorded physics at 1x",
                "onlineAvatarControl": False, "neuralWorldModelTrained": False,
                "stageLearningOrder": [1, 2, 3, 4],
                "stages": [], "gates": {"standing": "pending", "walking": "pending",
                                         "sitting": "pending", "unseenLayoutNavigation": "pending",
                                         "multiRoomTasks": "pending", "stairs": "pending"}}
    if evaluation:
        success = evaluation["policy"]["successRate"]
        manifest["gates"]["standing"] = "passed" if success >= .9 else "failed"
        manifest["training"] = evaluation["provenance"]
    for stage in range(1, 5):
        scene = generate_scene(stage, seed=4100+stage, difficulty=.85 if stage == 4 else .5)
        dump_json(output / f"stage{stage}-scene.json", scene)
        item = {"stage": stage, "title": TITLES[stage-1],
                "status": "立位方策を検証済み・歩行と着座は検証中" if stage == 1 and policy else "物理環境を作成・課題の学習は未完了",
                "sceneUrl": f"/world-model/stage{stage}-scene.json",
                "geometryVerified": bool(reports), "proficiencyPassed": False,
                "gates": {"standing": manifest["gates"]["standing"] if stage == 1 else "pending",
                          "walking": "pending", "sitting": "pending"},
                "metrics": {"stageWorldsChecked": sum(r["stage"] == stage for r in reports)},
                "description": scene.get("metadata", {}).get("description", "")}
        if policy:
            result = evaluate(policy, task="stand", scene=scene, seeds=[12000+stage], randomize=False,
                              avatar_profile=avatar_profile,
                              trace_path=output / f"stage{stage}-rollout.json")
            item["traceUrl"] = f"/world-model/stage{stage}-rollout.json"
            item["metrics"]["standingProbe"] = result
            item["traceTask"] = "stand"
            if stage == 1:
                evaluate(None, scene=scene, seeds=[12001], randomize=False,
                         avatar_profile=avatar_profile,
                         trace_path=output / "stage1-baseline.json")
                record_unpowered(scene, output / "stage1-unpowered.json", avatar_profile=avatar_profile)
                item["traceOptions"] = [
                    {"title": "学習した立位", "url": item["traceUrl"]},
                    {"title": "基準の立位制御", "url": "/world-model/stage1-baseline.json"},
                    {"title": "制御なし・重力による転倒", "url": "/world-model/stage1-unpowered.json"},
                ]
        manifest["stages"].append(item)
    dump_json(output / "physics-world-verification.json", reports)
    dump_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="frontend/public/world-model")
    parser.add_argument("--policy")
    parser.add_argument("--evaluation")
    parser.add_argument("--skip-world-checks", action="store_true")
    parser.add_argument("--avatar-profile")
    args = parser.parse_args()
    manifest = export(args.output, args.policy, args.evaluation, not args.skip_world_checks, args.avatar_profile)
    print(json.dumps({"stages": len(manifest["stages"]), "gates": manifest["gates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
