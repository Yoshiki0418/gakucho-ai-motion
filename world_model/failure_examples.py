"""Reproduce historical PPO falls and continue real physics for viewing.

The trained checkpoint is unchanged. The historical evaluation stops at the
first fall; a separate, explicitly excluded continuation keeps the same PPO
and bounded PD joint controller active until the body lands. Only Gymnasium's
episode-completion bookkeeping is released, never physical state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .body_contract import assert_policy_body, trace_body_metadata
from .humanoid import TORQUE_LIMITS
from .locomotion import FootstepHumanoidEnv
from .training import dump_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkpoint_sources(checkpoint, provenance):
    """Require the exact historical checkpoint and unchanged simulation code."""
    actual = sha256(checkpoint)
    if actual != provenance["checkpointSha256"]:
        raise ValueError("Historical checkpoint SHA256 differs from its provenance")
    for name, expected in provenance["sourceSha256"].items():
        if Path(name).name != name:
            raise ValueError("Training source names must be local module filenames")
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Historical simulation source changed: {name}")
    return actual


def verify_historical_fall(expected, info, total_return):
    """Do not silently replace a recorded failure with a different trajectory."""
    if not expected["fell"] or not info["fall"]:
        raise ValueError("A failure example must reproduce a recorded fall")
    checks = [(info["time"], expected["seconds"]),
              (info["root_height"], expected["final"]["root_height"]),
              (info["upright"], expected["final"]["upright"]),
              (total_return, expected["return"])]
    if any(not math.isclose(actual, prior, rel_tol=1e-10, abs_tol=1e-10)
           for actual, prior in checks):
        raise ValueError("Fall does not exactly reproduce historical evaluation")
    for key in ("root_position", "root_velocity"):
        if not np.allclose(info[key], expected["final"][key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Historical fall {key} differs")


def continue_physical_step(env, action):
    """Release the stop guard without changing the MuJoCo position/velocity."""
    position, velocity = env.data.qpos.copy(), env.data.qvel.copy()
    env._episode_done = False
    if not (np.array_equal(position, env.data.qpos)
            and np.array_equal(velocity, env.data.qvel)):
        raise AssertionError("Releasing episode bookkeeping changed physical state")
    return env.step(action)


def floor_contacts(env):
    """Record measured static floor contact forces, not inferred avatar poses."""
    result = []
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        ids = (int(contact.geom1), int(contact.geom2))
        names = [env.model.geom(geom).name for geom in ids]
        if "floor" not in names:
            continue
        moving = ids[1] if names[0] == "floor" else ids[0]
        if env.model.geom_bodyid[moving] == 0:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        result.append({"geometry": env.model.geom(moving).name,
                       "body": env.model.body(int(env.model.geom_bodyid[moving])).name,
                       "normalForceNewtons": float(force[0]),
                       "signedDistanceMetres": float(contact.dist)})
    return result


def physics_frame(env, action, phase):
    frame = env.snapshot()
    frame["controlsPhase"] = phase
    frame["physicsTelemetry"] = {
        "jointResidualAction": np.asarray(action).tolist(),
        "jointTorquesNewtonsMetres": env.data.ctrl.tolist(),
        "maximumTorqueLimitFraction": float(np.max(np.abs(env.data.ctrl)/TORQUE_LIMITS)),
        "externalForceMaximumAbs": float(np.max(np.abs(env.data.xfrc_applied))),
        "externallyAppliedGeneralizedForceMaximumAbs": float(np.max(np.abs(env.data.qfrc_applied))),
        "floorContacts": floor_contacts(env),
    }
    if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
        raise ValueError("Non-finite physics state cannot be published as a replay")
    if frame["physicsTelemetry"]["maximumTorqueLimitFraction"] > 1+1e-12:
        raise ValueError("Recorded joint torque exceeds its physical limit")
    if (frame["physicsTelemetry"]["externalForceMaximumAbs"] != 0
            or frame["physicsTelemetry"]["externallyAppliedGeneralizedForceMaximumAbs"] != 0):
        raise ValueError("Failure replay unexpectedly received an external force")
    return frame


def reproduce_failure(policy, *, seed, scene, avatar_profile, expected_episode,
                      checkpoint_sha256, source_sha256, settling_seconds=2.0,
                      evaluation_seconds=12.0):
    if not math.isfinite(settling_seconds) or settling_seconds <= 0:
        raise ValueError("Post-fall continuation must be positive and finite")
    env = FootstepHumanoidEnv(task="walk", scene=scene, randomize=True,
                             max_episode_seconds=evaluation_seconds, reference_start=False,
                             strict_gait_reward=True, avatar_profile=avatar_profile)
    try:
        assert_policy_body(policy, env)
        continuation_steps = round(settling_seconds/env.dt)
        if not math.isclose(continuation_steps*env.dt, settling_seconds, abs_tol=1e-9):
            raise ValueError("Continuation duration must be a whole number of control steps")
        observation, _ = env.reset(seed=int(seed))
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        frames = [physics_frame(env, zero_action, "evaluation")]
        rewards = []
        while True:
            action = policy.predict(observation, deterministic=True)[0]
            env.data.xfrc_applied[:] = 0
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            frames.append(physics_frame(env, action, "evaluation"))
            if terminated:
                verify_historical_fall(expected_episode, info, sum(rewards))
                break
            if truncated:
                raise ValueError("Selected historical failure did not fall before the evaluation limit")
        fall_info = dict(info)
        fall_time = float(env.data.time)
        fall_index = len(frames)-1
        # The same policy, observation, reference timing, body state and bounded
        # torque controller continue. There is no reset or pose/velocity write.
        for _ in range(continuation_steps):
            action = policy.predict(observation, deterministic=True)[0]
            env.data.xfrc_applied[:] = 0
            observation, _, _, _, _ = continue_physical_step(env, action)
            frames.append(physics_frame(env, action, "post-fall-continuation"))
        final = frames[-1]
        floor_landing = [frame for frame in frames[fall_index:]
                         if any(contact["body"] in ("pelvis", "torso", "head")
                                and contact["normalForceNewtons"] > 1
                                for contact in frame["physicsTelemetry"]["floorContacts"])]
        if not floor_landing or final["stats"]["root_height"] >= .3:
            raise ValueError("Continuation does not show the body physically landing on the floor")
        title = "歩行中に横へ転倒（学習途中）" if seed == 10001 else "歩き始めて転倒（学習途中）"
        description = (f"学長の体格へ適応中の以前の PPO。{fall_time:.2f} 秒で転倒判定。"
                       f"その後 {settling_seconds:g} 秒は同じ PPO＋PD 制御で物理を続行（評価外）。")
        failure = {"title": title, "description": description, "seed": int(seed),
                   "checkpointSha256": checkpoint_sha256,
                   "sourcePolicy": "earlier adaptation checkpoint (not selected production policy)",
                   "fallTimeSeconds": fall_time, "fallFrameIndex": fall_index,
                   "settlingSeconds": settling_seconds,
                   "evaluationExcluded": True,
                   "replayStartTimeSeconds": max(0., fall_time-2.),
                   "postFallController": "same deterministic PPO residual + bounded torque PD",
                   "postFallPhysics": "MuJoCo mj_step; only episode stop bookkeeping released",
                   "firstBodyFloorContactTimeSeconds": floor_landing[0]["time"]}
        trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
                 "dt": env.dt, "controller": "PPO residual", "task": "walk", "seed": int(seed),
                 "scene": scene, "motionReference": "cached T2M BVH", "referenceStart": False,
                 "physicalFootstepPrior": True, "gravity": env.model.opt.gravity.tolist(),
                 "frames": frames, "failureExample": failure,
                 "evaluation": {"controller": "PPO residual", "task": "walk", "episodeCount": 1,
                                "survivalRate": 0., "successRate": 0., "walkingVerifiedRate": 0.,
                                "episodes": [expected_episode], "evaluationEndsAtSeconds": fall_time,
                                "postFallContinuationExcluded": True},
                 "provenance": {"checkpointSha256": checkpoint_sha256,
                                "sourceSha256": source_sha256,
                                "failureReplaySourceSha256": sha256(__file__),
                                "deterministicPolicy": True, "randomizedReset": True,
                                "rootPositionOverwrittenDuringStep": False,
                                "rootVelocityOverwrittenDuringStep": False,
                                "externalPushNewtons": 0, "currentSelectedPolicy": False,
                                "historicalFallExactlyReproduced": True,
                                "mujocoVersion": mujoco.__version__,
                                "generatedAt": datetime.now(timezone.utc).isoformat()},
                 "physicsVerification": {"firstFall": fall_info,
                                         "finalRootHeightMetres": final["stats"]["root_height"],
                                         "finalUpright": final["stats"]["upright"],
                                         "firstBodyFloorContactTimeSeconds": floor_landing[0]["time"],
                                         "finalFloorContacts": final["physicsTelemetry"]["floorContacts"],
                                         "bodyMassKilograms": float(env.model.body_mass.sum()),
                                         "postFallPhysicalSteps": continuation_steps,
                                         "noExternalForces": True,
                                         "noResetOrPoseCorrection": True}}
        trace.update(trace_body_metadata(env))
        return trace
    finally:
        env.close()


def export(checkpoint_directory, historical_trace, avatar_profile, output, *, settling_seconds=2.):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    directory, output = Path(checkpoint_directory), Path(output)
    checkpoint = directory / "policy.zip"
    provenance = json.loads((directory / "provenance.json").read_text())
    checkpoint_sha = verify_checkpoint_sources(checkpoint, provenance)
    historical = json.loads(Path(historical_trace).read_text())
    if (historical["avatarProfileSha256"] != provenance["avatarProfileSha256"]
            or historical["bodyModelSha256"] != provenance["bodyModelSha256"]):
        raise ValueError("Historical fall evaluation does not match the checkpoint body")
    policy = PPO.load(checkpoint, device="cpu")
    examples = []
    for episode in historical["evaluation"]["episodes"]:
        if not episode["fell"]:
            continue
        seed = int(episode["seed"])
        trace = reproduce_failure(policy, seed=seed, scene=historical["scene"],
                                  avatar_profile=avatar_profile, expected_episode=episode,
                                  checkpoint_sha256=checkpoint_sha,
                                  source_sha256=provenance["sourceSha256"],
                                  settling_seconds=settling_seconds,
                                  evaluation_seconds=provenance["maxEpisodeSeconds"])
        filename = f"stage1-walking-fall-{seed}.json"
        dump_json(output / filename, trace)
        examples.append({**trace["failureExample"], "filename": filename,
                         "traceSha256": sha256(output / filename),
                         "physicsVerification": trace["physicsVerification"]})
    if not examples:
        raise ValueError("Historical evaluation contains no falling episodes")
    index = {"version": 1, "avatarProfileSha256": provenance["avatarProfileSha256"],
             "bodyModelSha256": provenance["bodyModelSha256"], "examples": examples,
             "checkpointSha256": checkpoint_sha, "historicalTraceSha256": sha256(historical_trace),
             "historicalTrace": str(historical_trace), "evaluationExcluded": True,
             "selectedProductionPolicyChanged": False,
             "sourcePolicy": "earlier adaptation checkpoint (not selected production policy)"}
    dump_json(output / "failure-examples.json", index)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", required=True)
    parser.add_argument("--historical-trace", required=True)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settling-seconds", type=float, default=2.)
    args = parser.parse_args()
    index = export(args.checkpoint_directory, args.historical_trace,
                   args.avatar_profile, args.output, settling_seconds=args.settling_seconds)
    print(json.dumps({"exampleCount": len(index["examples"]),
                      "examples": [{"seed": x["seed"], "fallTimeSeconds": x["fallTimeSeconds"],
                                    "finalRootHeightMetres": x["physicsVerification"]["finalRootHeightMetres"]}
                                   for x in index["examples"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
