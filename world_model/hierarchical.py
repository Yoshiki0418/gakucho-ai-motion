"""Learned walking followed, when necessary, by a stopped physical approach.

The approach is inverse dynamics, not an additional trained policy. A handoff
never resets the simulation or changes the free body's pose, velocity or
applied forces. The learned 12 s segment is measured before any correction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .gait_audit import audit_gait
from .locomotion import FootstepHumanoidEnv
from .training import dump_json
from .whole_body import WholeBodyController
from .body_contract import assert_policy_body, trace_body_metadata

CONTROLLER = "PPO residual + whole-body goal approach"


def safe_stopped_handoff(info, held_seconds):
    speed = float(np.linalg.norm(info["root_velocity"]))
    eligible = (not info["fall"] and not info["success"] and .25 <= info["goal_distance"] < .6
                and info["upright"] > .95 and speed < .04
                and info["left_foot_force"] > 50 and info["right_foot_force"] > 50)
    return bool(eligible and held_seconds >= .30-1e-9), bool(eligible)


def evaluate_hierarchical(policy, *, scene, seeds=range(1_000_600, 1_000_632),
                          learned_seconds=12., approach_seconds=20., randomize=True, trace_path=None,
                          checkpoint_sha256=None, avatar_profile=None):
    """Evaluate the learned segment and correction separately on fixed seeds."""
    if policy.observation_space.shape != (59,):
        raise ValueError("The physical walking policy must use the 59-input contract")
    if policy.action_space.shape != (20,):
        raise ValueError("The physical walking policy must control 20 joints")
    if learned_seconds < 1 or approach_seconds <= 0:
        raise ValueError("Evaluation windows must be positive")
    episodes, first_trace, approach_trace = [], None, None
    for episode_index, seed in enumerate(seeds):
        env = FootstepHumanoidEnv(scene=scene, reference_start=False, randomize=randomize,
                                  max_episode_seconds=learned_seconds+approach_seconds,
                                  avatar_profile=avatar_profile)
        assert_policy_body(policy, env)
        observation, info = env.reset(seed=int(seed))
        trace = {"version": 1, "task": "walk", "sizeConvention": "mujoco",
                 "geometryScope": "complete-world", "dt": env.dt, "seed": int(seed),
                 "controller": CONTROLLER, "scene": scene, "gravity": env.model.opt.gravity.tolist(),
                 "motionReference": "T2M-derived joint footstep prior; optional model-based final approach",
                 "referenceStart": False, "physicalFootstepPrior": True,
                 "frames": [env.snapshot()]}
        trace.update(trace_body_metadata(env))
        stop_duration = 0.
        while env.data.time < learned_seconds-1e-9 and not info["fall"]:
            action = policy.predict(observation, deterministic=True)[0]
            observation, _, _, _, info = env.step(action)
            _, eligible = safe_stopped_handoff(info, stop_duration)
            stop_duration = stop_duration+env.dt if eligible else 0.
            snapshot = env.snapshot()
            snapshot["stats"]["control_mode"] = "learned_walk"
            trace["frames"].append(snapshot)
        base_info = dict(info)
        base_quality = audit_gait(trace)
        permitted, _ = safe_stopped_handoff(info, stop_duration)
        used_approach = bool(permitted and not info["success"])
        failure = None
        if used_approach:
            before_position, before_velocity = env.data.qpos.copy(), env.data.qvel.copy()
            controller = WholeBodyController(env, walk=True, goal_direction=True)
            np.testing.assert_array_equal(before_position, env.data.qpos)
            np.testing.assert_array_equal(before_velocity, env.data.qvel)
            successful_hold = 0.
            while not env._episode_done and not info["fall"]:
                info = controller.step()
                if np.any(env.data.xfrc_applied) or np.any(env.data.qfrc_applied):
                    raise AssertionError("A goal approach must use joint torques only")
                snapshot = controller.snapshot()
                snapshot["stats"].update(success=info["success"], success_seconds=info["success_seconds"],
                                         control_mode="model_based_approach")
                trace["frames"].append(snapshot)
                successful_hold = successful_hold+env.dt if info["success"] else 0.
                if successful_hold >= 1-1e-9 or controller.failure:
                    break
            failure = controller.failure
        quality = audit_gait(trace)
        episode = {"seed": int(seed), "seconds": float(env.data.time), "fell": bool(info["fall"]),
                   "success": bool(info["success"] and not info["fall"] and failure is None),
                   "gaitQuality": quality, "usedModelBasedApproach": used_approach,
                   "approachFailure": failure, "final": info,
                   "learnedSegment": {"seconds": trace["frames"][round(min(learned_seconds, env.data.time)/env.dt)]["time"],
                                      "success": bool(base_info["success"]), "fell": bool(base_info["fall"]),
                                      "final": base_info, "gaitQuality": base_quality}}
        episodes.append(episode)
        if episode_index == 0:
            first_trace = trace
        if used_approach and approach_trace is None:
            approach_trace = trace
        env.close()
    if not episodes:
        raise ValueError("Evaluation requires at least one seed")
    result = {"version": 1, "controller": CONTROLLER, "task": "walk", "episodeCount": len(episodes),
              "episodes": episodes, "randomized": randomize, "referenceStart": False,
              "physicalFootstepPrior": True, "learnedWindowSeconds": learned_seconds,
              "maximumApproachSeconds": approach_seconds, "evaluationPushForceN": 0,
              "survivalRate": float(np.mean([not e["fell"] for e in episodes])),
              "successRate": float(np.mean([e["success"] for e in episodes])),
              "walkingVerifiedRate": float(np.mean([e["gaitQuality"]["walkingVerified"] for e in episodes])),
              "completionVerifiedRate": float(np.mean([e["success"] and e["gaitQuality"]["walkingVerified"] for e in episodes])),
              "learnedOnlySuccessRate": float(np.mean([e["learnedSegment"]["success"] for e in episodes])),
              "learnedOnlyWalkingVerifiedRate": float(np.mean([e["learnedSegment"]["gaitQuality"]["walkingVerified"] for e in episodes])),
              "learnedOnlyCompletionVerifiedRate": float(np.mean([e["learnedSegment"]["success"] and e["learnedSegment"]["gaitQuality"]["walkingVerified"] for e in episodes])),
              "modelBasedApproachCount": sum(e["usedModelBasedApproach"] for e in episodes),
              "evaluationStrictGaitReward": False,
              "checkpointSha256": checkpoint_sha256,
              "sourceSha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in ("hierarchical.py", "whole_body.py", "gait_audit.py", "humanoid.py",
                                            "environment.py", "imitation.py", "references.py", "locomotion.py",
                                            "contact_metrics.py", "training.py", "avatar_profile.py", "body_contract.py")},
              "scope": "2 m flat-floor task; learned walking with optional stopped inverse-dynamics approach",
              "neuralWorldModelTrained": False, "rootPositionOverwrittenDuringStep": False}
    if avatar_profile is not None:
        result.update(avatarProfileSha256=env.avatar_profile_sha256,
                      bodyModelSha256=env.body_model_sha256)
    if trace_path:
        first_trace["evaluation"] = result
        dump_json(trace_path, first_trace)
        if approach_trace:
            approach_trace["evaluation"] = result
            approach_trace["exampleSelection"] = "first episode requiring physical approach in declared seed order"
            destination = Path(trace_path)
            dump_json(destination.with_name(destination.stem+"-approach-example.json"), approach_trace)
    return result
