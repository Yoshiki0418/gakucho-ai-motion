"""Held-out physical navigation audit, with arrival, contact and gait gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .body_contract import trace_body_metadata
from .gait_audit import audit_gait
from .training import dump_json


def audit_navigation(trace):
    """Audit the whole trajectory; a late arrival cannot erase a collision."""
    gait = audit_gait(trace)
    frames = trace["frames"]
    collided = any(frame["stats"].get("collision_occurred", False)
                   or frame["stats"].get("obstacle_collision", False)
                   or frame["stats"].get("navigation_collision", False)
                   or frame["stats"].get("navigation_collision_occurred", False)
                   or frame["stats"].get("collision_force_newtons", 0) > 1
                   for frame in frames)
    final = frames[-1]["stats"]
    supported = final.get("navigation_floor_support") == [True, True]
    stopped = float(np.linalg.norm(final["root_velocity"])) < .25
    arrived = final["goal_distance"] < .25
    stop_window = [frame for frame in frames if frame["time"] > frames[-1]["time"]-.5+1e-9]
    stop_samples = math.ceil(.5/trace["dt"]-1e-9)
    def valid_stop_sample(frame):
        stats = frame["stats"]
        metrics = stats.get("navigation", {}).get("metrics", {})
        loads = metrics.get("floor_support_forces_newtons", [])
        return (stats["goal_distance"] < .25 and float(np.linalg.norm(stats["root_velocity"])) < .25
            and stats["upright"] > .8 and len(loads) == 2 and min(loads) >= 20
            and metrics.get("goal_floor_matches") is True
            and metrics.get("root_height_error_m", math.inf) < .15)
    held = (final.get("navigation_success_seconds", 0) >= .5-1e-9
            and len(stop_window) >= stop_samples and all(valid_stop_sample(frame) for frame in stop_window))
    mode = trace.get("navigation", {}).get("mode", "turn")
    scene = trace.get("scene", {})
    furniture = next((geom for geom in scene.get("geoms", []) if geom.get("entityId") == "navigation-furniture"), None)
    from .navigation_scenes import segment_intersects_box
    blocked = bool(furniture is not None and scene.get("metadata", {}).get("directRouteBlocked", False)
                   and segment_intersects_box(scene["spawn"], scene["goals"][0]["position"], furniture))
    gates = {"survived": not any(frame["stats"]["fall"] for frame in frames),
             "collisionFree": not collided, "goalReached": arrived,
             "stopped": stopped, "bothFeetSupport": supported,
             "sustainedStop": held, "physicalRepeatedGait": gait["walkingVerified"],
             "obstacleActuallyBlocksDirectRoute": mode != "obstacle" or blocked}
    return {"version": 1, "navigationVerified": all(gates.values()),
            "gates": gates, "failedGates": [name for name, value in gates.items() if not value],
            "gaitQuality": gait,
            "scope": "one-goal flat-floor task; turning or a single randomized blocking obstacle",
            "contactAudit": "actual solver contacts observed every 0.002 s; sticky collision history",
            "stopAudit": "final 0.5 s actual control-frame samples at 0.02 s; both floor loads and speed rechecked",
            "anyRoomNavigationVerified": False, "stairsVerified": False}


def evaluate(policy, *, mode="turn", scene_seeds=range(1_100_000, 1_100_008),
             physics_seed_base=3_100_000, difficulty_range=(0., .25),
             seconds=30., avatar_profile=None, output=None, checkpoint_sha256=None):
    from .navigation import EpisodeNavigationEnv
    from .navigation_training import assert_navigation_policy

    scene_seeds = list(scene_seeds)
    if not scene_seeds or len(set(scene_seeds)) != len(scene_seeds):
        raise ValueError("Navigation evaluation requires distinct held-out scene seeds")
    final_cohort = all(1_800_000 <= seed < 2_000_000 for seed in scene_seeds)
    output = Path(output) if output else None
    episodes, recorded = [], []
    for index, scene_seed in enumerate(scene_seeds):
        env = EpisodeNavigationEnv(mode=mode, split="heldout", base_seed=scene_seed,
                    difficulty_range=difficulty_range, randomize=True,
                    max_episode_seconds=seconds, avatar_profile=avatar_profile)
        try:
            observation, info = env.reset(seed=physics_seed_base+index, options={
                "scene_seed": int(scene_seed), "physics_seed": physics_seed_base+index})
            assert_navigation_policy(policy, env)
            body = env.unwrapped
            scene = body.scene
            trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
                "dt": body.dt, "task": "walk", "controller": "PPO navigation residual" if policy else "PD navigation baseline",
                "motionReference": "cached T2M BVH", "physicalFootstepPrior": True,
                "scene": scene, "gravity": body.model.opt.gravity.tolist(),
                "navigation": {"mode": mode, "sceneSeed": int(scene_seed),
                    "physicsSeed": physics_seed_base+index, "checkpointSha256": checkpoint_sha256,
                    "difficultyRange": list(difficulty_range), "split": "heldout",
                    "pathController": "none; learned joint residual with goal-heading physical prior",
                    "rootPositionOverwrittenDuringStep": False,
                    "evaluationRole": "development" if int(scene_seed) < 1_800_000 else "final"},
                "frames": [env.snapshot()]}
            trace.update(trace_body_metadata(body))
            terminated = truncated = False
            rewards = []
            maximum_external_force = 0.
            while not (terminated or truncated):
                action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy.predict(observation, deterministic=True)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                maximum_external_force = max(maximum_external_force,
                    float(np.max(np.abs(body.data.xfrc_applied))),
                    float(np.max(np.abs(body.data.qfrc_applied))))
                rewards.append(float(reward))
                trace["frames"].append(env.snapshot())
            if maximum_external_force != 0:
                raise ValueError("Navigation evaluation unexpectedly received an external force")
            quality = audit_navigation(trace)
            trace["navigationQuality"] = quality
            final = trace["frames"][-1]["stats"]
            episode = {"sceneSeed": int(scene_seed), "physicsSeed": physics_seed_base+index,
                "seconds": float(body.data.time), "return": sum(rewards),
                "fell": bool(final["fall"]), "arrivalStopped": all(quality["gates"][key] for key in
                    ("goalReached", "stopped", "bothFeetSupport", "sustainedStop")),
                "collisionFree": quality["gates"]["collisionFree"],
                "walkingVerified": quality["gates"]["physicalRepeatedGait"],
                "navigationVerified": quality["navigationVerified"],
                "final": final, "quality": quality, "externalForceMaximum": maximum_external_force,
                "sceneSha256": hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
            episodes.append(episode)
            # Keep the first example and the first verified/failure example,
            # declaring every scene in the aggregate without selecting its score.
            kind = "verified" if quality["navigationVerified"] else "failed"
            should_record = index == 0 or not any(item["kind"] == kind for item in recorded)
            if output and should_record:
                filename = f"stage2-{mode}-{scene_seed}-{kind}.json"
                dump_json(output/filename, trace)
                recorded.append({"kind": kind, "sceneSeed": int(scene_seed), "filename": filename,
                                 "traceSha256": hashlib.sha256((output/filename).read_bytes()).hexdigest()})
            print(json.dumps({"sceneSeed": int(scene_seed), "mode": mode,
                "seconds": episode["seconds"], "goalDistance": final["goal_distance"],
                "fell": episode["fell"], "navigationVerified": episode["navigationVerified"],
                "failedGates": quality["failedGates"]}, ensure_ascii=False), flush=True)
        finally:
            env.close()
    if not episodes:
        raise ValueError("Navigation evaluation requires at least one held-out world")
    result = {"version": 1, "mode": mode, "episodeCount": len(episodes),
        "checkpointSha256": checkpoint_sha256,
        "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
        "survivalRate": sum(not e["fell"] for e in episodes)/len(episodes),
        "collisionFreeRate": sum(e["collisionFree"] for e in episodes)/len(episodes),
        "arrivalStoppedRate": sum(e["arrivalStopped"] for e in episodes)/len(episodes),
        "physicalGaitPassRate": sum(e["walkingVerified"] for e in episodes)/len(episodes),
        "navigationVerifiedRate": sum(e["navigationVerified"] for e in episodes)/len(episodes),
        "evaluationRole": "final" if final_cohort else "development",
        "proficiencyPassed": final_cohort and len(episodes) >= 32 and sum(e["navigationVerified"] for e in episodes)/len(episodes) >= .9,
        "difficultyRange": list(difficulty_range), "seconds": seconds,
        "episodes": episodes, "recordings": recorded,
        "sourceSha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ("navigation_evaluation.py", "navigation.py", "navigation_scenes.py", "steering.py", "gait_audit.py")},
        "scope": "flat floor, one goal, single blocking furniture obstacle or small-angle turning",
        "anyRoomNavigationVerified": False, "stairsVerified": False}
    if output:
        dump_json(output/"evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", help="110-input policy ZIP; omitted means zero residual baseline")
    parser.add_argument("--mode", choices=("turn", "obstacle"), default="turn")
    parser.add_argument("--scene-seed-base", type=int, default=1_100_000)
    parser.add_argument("--physics-seed-base", type=int, default=3_100_000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.25)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0 or not 0 <= args.difficulty_min <= args.difficulty_max <= 1:
        parser.error("Episodes and seconds must be positive, with difficulty in [0,1]")
    import torch
    torch.set_num_threads(1)
    policy = None
    checkpoint_sha = None
    if args.policy:
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy, device="cpu")
        checkpoint_sha = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    result = evaluate(policy, mode=args.mode, scene_seeds=range(args.scene_seed_base,args.scene_seed_base+args.episodes),
        physics_seed_base=args.physics_seed_base, difficulty_range=(args.difficulty_min,args.difficulty_max),
        seconds=args.seconds, avatar_profile=args.avatar_profile, output=args.output,
        checkpoint_sha256=checkpoint_sha)
    print(json.dumps({key:result[key] for key in ("episodeCount", "survivalRate", "collisionFreeRate", "arrivalStoppedRate", "navigationVerifiedRate", "proficiencyPassed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
