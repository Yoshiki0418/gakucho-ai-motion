"""Audit learned command navigation at every actual 20ms motor-control frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .body_contract import trace_body_metadata
from .generalization import _difficulty_range, validate_scene_seed
from .navigation_evaluation import audit_navigation
from .training import dump_json


def _scene_seeds(seeds):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation requires distinct held-out scene seeds")
    return [validate_scene_seed(seed, "heldout") for seed in seeds]


def append_control_frames(trace, frames, *, expected_count, control_dt):
    """Reject downsampled, reordered or duplicated replay/audit frames."""
    if len(frames) != expected_count or expected_count < 1:
        raise ValueError("each command decision must retain every executed motor-control frame")
    previous = float(trace["frames"][-1]["time"])
    for frame in frames:
        timestamp = float(frame["time"])
        if not math.isfinite(timestamp) or not math.isclose(timestamp-previous, control_dt, abs_tol=1e-8):
            raise ValueError("navigation audit requires uninterrupted actual 20ms control samples")
        previous = timestamp
    trace["frames"].extend(frames)


def evaluate(policy, *, motor_checkpoint, mode="obstacle",
             scene_seeds=range(1_160_000, 1_160_008), physics_seed_base=3_160_000,
             difficulty_range=(0., .15), seconds=30., avatar_profile=None, output=None,
             checkpoint_sha256=None, decision_steps=5, max_heading_radians=.45,
             max_lookahead_m=.5, heading_slew_radians_per_second=.175,
             motor_hip_yaw_action_limit=None):
    from .command_navigation import CommandNavigationEnv, assert_command_policy

    scene_seeds = _scene_seeds(scene_seeds)
    difficulty_range = _difficulty_range(difficulty_range)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("evaluation seconds must be positive and finite")
    if policy is not None and getattr(policy, "num_timesteps", 0) < 1:
        raise ValueError("command evaluation requires a trained high-level checkpoint")
    motor_checkpoint = Path(motor_checkpoint)
    if motor_checkpoint.suffix != ".zip":
        motor_checkpoint = motor_checkpoint.with_suffix(".zip")
    motor_sha = hashlib.sha256(motor_checkpoint.read_bytes()).hexdigest()
    final_cohort = all(1_800_000 <= seed < 2_000_000 for seed in scene_seeds)
    output = Path(output) if output else None
    episodes, recorded = [], []
    control_contract = None
    for index, scene_seed in enumerate(scene_seeds):
        env = CommandNavigationEnv(motor_checkpoint=motor_checkpoint,
            decision_steps=decision_steps, max_heading_radians=max_heading_radians,
            max_lookahead_m=max_lookahead_m,
            heading_slew_radians_per_second=heading_slew_radians_per_second,
            motor_hip_yaw_action_limit=motor_hip_yaw_action_limit,
            record_trajectory=True,
            mode=mode, split="heldout", base_seed=scene_seed, difficulty_range=difficulty_range,
            randomize=True, max_episode_seconds=seconds, avatar_profile=avatar_profile)
        try:
            observation, _ = env.reset(seed=physics_seed_base+index, options={
                "scene_seed": scene_seed, "physics_seed": physics_seed_base+index})
            assert_command_policy(policy, env)
            body, scene = env.unwrapped, env.navigation_env.current_scene
            control_contract = env.control_contract
            trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
                "dt": float(body.dt), "task": "walk",
                "controller": "PPO command navigator with frozen PPO joint motor" if policy else "zero command baseline with frozen PPO joint motor",
                "motionReference": "cached T2M BVH", "physicalFootstepPrior": True,
                "scene": scene, "gravity": body.model.opt.gravity.tolist(),
                "navigation": {"mode": mode, "sceneSeed": scene_seed,
                    "physicsSeed": physics_seed_base+index, "checkpointSha256": checkpoint_sha256,
                    "motorCheckpointSha256": motor_sha,
                    "difficultyRange": list(difficulty_range), "split": "heldout",
                    "pathController": "learned heading/lookahead commands; independently frozen 20-joint PPO motor",
                    "commandControlContract": control_contract,
                    "observationCount": 112, "actionCount": 2,
                    "motorObservationCount": 110, "motorActionCount": 20,
                    "motorTrainingTimesteps": int(env._motor.num_timesteps),
                    "controlDtSeconds": float(body.dt),
                    "maximumDecisionDtSeconds": float(body.dt*decision_steps),
                    "motorFrozen": True, "rootPositionOverwrittenDuringStep": False,
                    "fixedWaypointsSupplied": False,
                    "evaluationRole": "final" if scene_seed >= 1_800_000 else "development"},
                "frames": [env.snapshot()]}
            trace.update(trace_body_metadata(body))
            terminated = truncated = False
            rewards, decisions, controls = [], 0, 0
            maximum_external_force = 0.
            while not (terminated or truncated):
                action = np.zeros((2,), dtype=np.float32) if policy is None else policy.predict(observation, deterministic=True)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                count = info.get("command_control_steps")
                if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= decision_steps:
                    raise ValueError("evaluation requires the exact motor tick count per decision")
                append_control_frames(trace, env.last_control_frames, expected_count=count, control_dt=body.dt)
                maximum_external_force = max(maximum_external_force,
                    float(np.max(np.abs(body.data.xfrc_applied))),
                    float(np.max(np.abs(body.data.qfrc_applied))))
                rewards.append(float(reward))
                decisions += 1
                controls += count
            if maximum_external_force != 0:
                raise ValueError("command navigation received an external force")
            quality = audit_navigation(trace)
            trace["navigationQuality"] = quality
            final = trace["frames"][-1]["stats"]
            episode = {"sceneSeed": scene_seed, "physicsSeed": physics_seed_base+index,
                "seconds": float(body.data.time), "return": sum(rewards),
                "decisionSteps": decisions, "physicalControlSteps": controls,
                "physicalControlFrameCount": len(trace["frames"])-1,
                "fell": bool(final["fall"]),
                "arrivalStopped": all(quality["gates"][key] for key in
                    ("goalReached", "stopped", "bothFeetSupport", "sustainedStop")),
                "collisionFree": quality["gates"]["collisionFree"],
                "walkingVerified": quality["gates"]["physicalRepeatedGait"],
                "navigationVerified": quality["navigationVerified"], "final": final,
                "quality": quality, "externalForceMaximum": maximum_external_force,
                "sceneSha256": hashlib.sha256(json.dumps(scene, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
            episodes.append(episode)
            kind = "verified" if quality["navigationVerified"] else "failed"
            if output and (index == 0 or not any(item["kind"] == kind for item in recorded)):
                filename = f"stage2-command-{mode}-{scene_seed}-{kind}.json"
                dump_json(output/filename, trace)
                recorded.append({"kind": kind, "sceneSeed": scene_seed, "filename": filename,
                    "traceSha256": hashlib.sha256((output/filename).read_bytes()).hexdigest()})
            print(json.dumps({"sceneSeed": scene_seed, "mode": mode, "seconds": episode["seconds"],
                "goalDistance": final["goal_distance"], "fell": episode["fell"],
                "navigationVerified": episode["navigationVerified"],
                "failedGates": quality["failedGates"]}, ensure_ascii=False), flush=True)
        finally:
            env.close()
    if hashlib.sha256(motor_checkpoint.read_bytes()).hexdigest() != motor_sha:
        raise RuntimeError("frozen motor checkpoint changed during evaluation")
    rate = sum(episode["navigationVerified"] for episode in episodes)/len(episodes)
    result = {"version": 1, "mode": mode,
        "controller": "learned command navigator with frozen learned motor" if policy else "zero command baseline with frozen learned motor",
        "episodeCount": len(episodes), "checkpointSha256": checkpoint_sha256,
        "motorCheckpointSha256": motor_sha,
        "commandControlContract": control_contract, "observationCount": 112, "actionCount": 2,
        "motorObservationCount": 110, "motorActionCount": 20,
        "motorTrainingTimesteps": int(env._motor.num_timesteps),
        "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
        "survivalRate": sum(not episode["fell"] for episode in episodes)/len(episodes),
        "collisionFreeRate": sum(episode["collisionFree"] for episode in episodes)/len(episodes),
        "arrivalStoppedRate": sum(episode["arrivalStopped"] for episode in episodes)/len(episodes),
        "physicalGaitPassRate": sum(episode["walkingVerified"] for episode in episodes)/len(episodes),
        "navigationVerifiedRate": rate,
        "evaluationRole": "final" if final_cohort else "development",
        "proficiencyPassed": policy is not None and final_cohort and len(episodes) >= 32 and rate >= .9,
        "difficultyRange": list(difficulty_range), "seconds": seconds,
        "episodes": episodes, "recordings": recorded,
        "sourceSha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ("command_navigation_evaluation.py", "command_navigation.py", "command_gait.py",
                "navigation_evaluation.py", "navigation.py", "navigation_scenes.py", "steering.py", "gait_audit.py")},
        "scope": "flat floor, one goal, single blocking furniture obstacle or small-angle turning",
        "anyRoomNavigationVerified": False, "stairsVerified": False}
    if output:
        dump_json(output/"evaluation.json", result)
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Trained 112-input/2-action navigator ZIP")
    parser.add_argument("--motor-policy", required=True, help="Separately frozen 110-input/20-joint motor ZIP")
    parser.add_argument("--mode", choices=("turn", "obstacle"), default="obstacle")
    parser.add_argument("--scene-seed-base", type=int, default=1_160_000)
    parser.add_argument("--physics-seed-base", type=int, default=3_160_000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.15)
    parser.add_argument("--decision-steps", type=int, default=5)
    parser.add_argument("--max-heading-radians", type=float, default=.45)
    parser.add_argument("--max-lookahead-m", type=float, default=.5)
    parser.add_argument("--heading-slew-radians-per-second", type=float, default=.175)
    parser.add_argument("--motor-hip-yaw-action-limit", type=float,
                        help="Must match the checkpoint's explicit version-2 motor limit")
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.episodes < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("episodes and seconds must be positive and finite")
    try:
        seeds = _scene_seeds(range(args.scene_seed_base, args.scene_seed_base+args.episodes))
        difficulty_range = _difficulty_range((args.difficulty_min, args.difficulty_max))
    except ValueError as error:
        parser.error(str(error))
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    path = Path(args.policy)
    path = path if path.suffix == ".zip" else path.with_suffix(".zip")
    motor_path = Path(args.motor_policy)
    motor_path = motor_path if motor_path.suffix == ".zip" else motor_path.with_suffix(".zip")
    policy = PPO.load(path, device="cpu")
    result = evaluate(policy, motor_checkpoint=motor_path, mode=args.mode, scene_seeds=seeds,
        physics_seed_base=args.physics_seed_base, difficulty_range=difficulty_range,
        seconds=args.seconds, avatar_profile=args.avatar_profile, output=args.output,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        decision_steps=args.decision_steps, max_heading_radians=args.max_heading_radians,
        max_lookahead_m=args.max_lookahead_m,
        heading_slew_radians_per_second=args.heading_slew_radians_per_second,
        motor_hip_yaw_action_limit=args.motor_hip_yaw_action_limit)
    print(json.dumps({key: result[key] for key in ("episodeCount", "survivalRate",
        "collisionFreeRate", "arrivalStoppedRate", "navigationVerifiedRate", "proficiencyPassed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
