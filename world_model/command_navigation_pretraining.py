"""Warm only a command navigator's actor from verified train-room rollouts.

The analytic teacher follows the final goal bearing on an empty turning floor;
it supplies no geometry path, hidden waypoint, root pose or human recording.
Only real episodes passing the independent physical navigation audit enter
the supervised dataset. This does not add to the PPO decision/tick counts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def goal_bearing_teacher(env):
    """Only the real current body orientation and final goal choose commands."""
    body = env.unwrapped
    rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    delta = body.goal[:2] - body.data.qpos[:2]
    bearing = math.atan2(delta[0], -delta[1])
    error = math.atan2(math.sin(bearing-heading), math.cos(bearing-heading))
    return np.asarray([np.clip(error/env.max_heading_radians, -1., 1.), 1.],
                      dtype=np.float32)


def validated_motor_limit(args):
    """Keep old programmatic args compatible with the optional motor adapter."""
    value = getattr(args, "motor_hip_yaw_action_limit", None)
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("motor-hip-yaw-action-limit must be finite and in (0, 1]")
    return float(value)


def collect(args):
    from .body_contract import trace_body_metadata
    from .command_navigation import CommandNavigationEnv
    from .command_navigation_evaluation import append_control_frames
    from .command_navigation_training import _source_hashes
    from .generalization import validate_scene_seed
    from .navigation_evaluation import audit_navigation

    if args.episodes < 1 or args.episodes > 32:
        raise ValueError("teacher collection requires 1..32 train episodes")
    seeds = [validate_scene_seed(args.scene_seed_base+i, "train") for i in range(args.episodes)]
    physics_seeds = [validate_scene_seed(args.physics_seed_base+i, "train") for i in range(args.episodes)]
    output = Path(args.output)
    if (output/"dataset.npz").exists():
        raise ValueError("teacher output already contains data; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    sources = _source_hashes()
    motor_sha = sha256(args.motor_policy)
    env = CommandNavigationEnv(motor_checkpoint=args.motor_policy,
        motor_hip_yaw_action_limit=validated_motor_limit(args), mode="turn", split="train",
        base_seed=seeds[0], difficulty_range=(args.difficulty_min, args.difficulty_max),
        randomize=True, max_episode_seconds=args.seconds, avatar_profile=args.avatar_profile,
        record_trajectory=True)
    episodes, observations, actions, episode_indices = [], [], [], []
    started = time.monotonic()
    try:
        for index, (scene_seed, physics_seed) in enumerate(zip(seeds, physics_seeds)):
            observation, _ = env.reset(seed=physics_seed, options={
                "scene_seed": scene_seed, "physics_seed": physics_seed})
            body = env.unwrapped
            actual_goal, goal_copy = body.goal, body.goal.copy()
            observer = body._physics_contact_observer
            measured_substeps = 0

            def observe(model, data):
                nonlocal measured_substeps
                measured_substeps += 1
                if body.goal is not actual_goal or not np.array_equal(body.goal, goal_copy):
                    raise RuntimeError("teacher command changed the actual final destination")
                if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                    raise RuntimeError("teacher rollout applied an external force")
                observer(model, data)

            body._physics_contact_observer = observe
            trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
                "dt": float(body.dt), "task": "walk", "scene": env.scene,
                "controller": "analytic goal-bearing teacher with frozen learned joint motor",
                "navigation": {"mode": "turn", "split": "train", "sceneSeed": scene_seed,
                    "physicsSeed": physics_seed, "evaluationRole": "teacher_train_collection",
                    "motorCheckpointSha256": motor_sha, "fixedWaypointsSupplied": False,
                    "geometryPathTeacherUsed": False, "humanMotionDataUsed": False,
                    "commandControlContract": copy.deepcopy(env.control_contract),
                    "highLevelPolicyTrained": False}, "frames": [env.snapshot()]}
            trace.update(trace_body_metadata(body))
            current_observations, current_actions = [], []
            terminated = truncated = False
            while not (terminated or truncated):
                rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
                expected = (rotation.T @ (actual_goal-body.data.qpos[:3])).astype(np.float32)
                np.testing.assert_array_equal(observation[50:53], expected)
                command = goal_bearing_teacher(env)
                current_observations.append(observation.copy())
                current_actions.append(command.copy())
                observation, _, terminated, truncated, info = env.step(command)
                append_control_frames(trace, env.last_control_frames,
                    expected_count=info["command_control_steps"], control_dt=body.dt)
            body._physics_contact_observer = observer
            quality = audit_navigation(trace)
            trace["navigationQuality"] = quality
            accepted = bool(quality["navigationVerified"])
            if measured_substeps != (len(trace["frames"])-1)*10:
                raise RuntimeError("teacher rollout did not observe every actual 2ms step")
            trace_name = f"teacher-turn-{scene_seed}-{'verified' if accepted else 'failed'}.json"
            # Keep original native snapshots for independent audit/review.
            dump(output/trace_name, trace)
            row = {"sceneSeed": scene_seed, "physicsSeed": physics_seed, "split": "train",
                "seconds": float(body.data.time), "acceptedForSupervision": accepted,
                "decisions": len(current_actions), "physical20msControls": len(trace["frames"])-1,
                "physical2msSubstepsObserved": measured_substeps,
                "realFinalGoalUnchangedEverySubstep": True, "maximumExternalForce": 0.,
                "navigationVerified": accepted, "failedGates": quality["failedGates"],
                "fell": bool(info["fall"]), "collisionOccurred": bool(info["collision_occurred"]),
                "finalGoalDistanceMeters": float(info["goal_distance"]),
                "sceneSha256": env.navigation_env._scene_sha256,
                "traceFile": trace_name, "traceSha256": sha256(output/trace_name)}
            episodes.append(row)
            if accepted:
                observations.extend(current_observations)
                actions.extend(current_actions)
                episode_indices.extend([index]*len(current_actions))
            dump(output/"collection-progress.json", {"episodes": episodes,
                "qualifiedEpisodeCount": sum(r["acceptedForSupervision"] for r in episodes),
                "qualifiedDecisionCount": len(actions), "heldOutScenesUsed": False})
            print(json.dumps(row), flush=True)
            if _source_hashes() != sources or sha256(args.motor_policy) != motor_sha:
                raise RuntimeError("frozen simulator or motor changed during teacher collection")
        if not observations:
            raise ValueError("no physical teacher episode qualified; supervised fitting is prohibited")
        np.savez_compressed(output/"dataset.npz",
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            episode_indices=np.asarray(episode_indices, dtype=np.int32))
        report = {"role": "verified train-only analytic final-goal bearing teacher dataset",
            "episodeCount": len(episodes), "qualifiedEpisodeCount": sum(r["acceptedForSupervision"] for r in episodes),
            "qualifiedDecisionCount": len(actions), "episodes": episodes,
            "datasetSha256": sha256(output/"dataset.npz"), "motorCheckpointSha256": motor_sha,
            "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
            "sourceSha256": sources, "pretrainingSourceSha256": sha256(__file__),
            "commandControlContract": env.control_contract, "observations": 112, "actions": 2,
            "heldOutScenesUsed": False, "humanMotionDataUsed": False, "geometryPathTeacherUsed": False,
            "rootStateOverwrittenDuringStep": False, "failedEpisodesExcluded": True,
            "wallSeconds": time.monotonic()-started}
        dump(output/"collection.json", report)
        return report
    finally:
        env.close()


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_actor(policy, observations, actions, *, epochs=150, batch_size=256,
              learning_rate=3e-4, seed=88401):
    """Fit action means only, preserving value weights and exploration variance."""
    import torch
    if observations.ndim != 2 or observations.shape[1] != 112 or actions.shape != (len(observations), 2):
        raise ValueError("pretraining data must contain paired 112 observations / two actions")
    if len(observations) < 2 or not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("pretraining data must be finite and contain at least two decisions")
    if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("supervised fitting parameters must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    actor_prefixes = ("mlp_extractor.policy_net.", "action_net.")
    before = {key: value.clone() for key, value in policy.policy.state_dict().items()}
    protected = {key: value for key, value in before.items() if not key.startswith(actor_prefixes)}
    parameters = list(policy.policy.mlp_extractor.policy_net.parameters()) + list(policy.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    inputs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    targets = torch.as_tensor(actions, dtype=torch.float32, device=policy.device)

    def mean_action(batch):
        features = policy.policy.extract_features(batch)
        if isinstance(features, tuple):
            features = features[0]
        return policy.policy.action_net(policy.policy.mlp_extractor.forward_actor(features))

    def errors():
        with torch.no_grad():
            error = mean_action(inputs)-targets
            return {"mse": float(torch.mean(error.square())),
                "headingMeanAbsoluteError": float(error[:, 0].abs().mean()),
                "lookaheadMeanAbsoluteError": float(error[:, 1].abs().mean())}

    original_errors = errors()
    policy.policy.set_training_mode(True)
    minibatches = 0
    for _ in range(epochs):
        ordering = torch.randperm(len(inputs), device=policy.device)
        for start in range(0, len(inputs), batch_size):
            indices = ordering[start:start+batch_size]
            loss = (mean_action(inputs[indices])-targets[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            minibatches += 1
    policy.policy.set_training_mode(False)
    after = policy.policy.state_dict()
    for key, value in protected.items():
        if not torch.equal(after[key], value):
            raise RuntimeError(f"supervision changed protected critic/log-std parameter {key}")
    policy.policy.optimizer = policy.policy.optimizer_class(
        policy.policy.parameters(), lr=policy.lr_schedule(1.), **policy.policy.optimizer_kwargs)
    return {"algorithm": "actor action-mean MSE / Adam", "epochs": epochs,
        "supervisedMinibatches": minibatches, "supervisedExamples": len(inputs),
        "learningRate": learning_rate, "errorsBefore": original_errors, "errorsAfter": errors(),
        "criticAndLogStdPreservedExactly": True, "protectedTensorSha256": tensor_digest(protected),
        "ppoOptimizerReset": True, "rlDecisionTimestepsAdded": 0, "physicsControlTimestepsAdded": 0}


def inherited_source_provenance(source, *, source_sha, policy):
    """Copy audited RL/motor fields without losing their original lineage."""
    if source.get("checkpointSha256") != source_sha:
        raise ValueError("source provenance does not identify the loaded checkpoint")
    required = ("motorCheckpoint", "motorCheckpointSha256", "motorFrozen",
        "observationCount", "actionCount", "motorObservationCount", "motorActionCount",
        "motorTrainingTimesteps", "actualDecisionTimesteps", "physicalControlTimesteps",
        "avatarProfileSha256", "bodyModelSha256", "commandControlContract")
    if any(source.get(key) is None for key in required):
        raise ValueError("source provenance lacks the audited command/motor contract")
    if (source["motorFrozen"] is not True or source["observationCount"] != 112
            or source["actionCount"] != 2 or source["motorObservationCount"] != 110
            or source["motorActionCount"] != 20
            or source["actualDecisionTimesteps"] != int(policy.num_timesteps)
            or source["physicalControlTimesteps"] != int(policy.command_physical_control_timesteps)
            or source["avatarProfileSha256"] != policy.avatar_profile_sha256
            or source["bodyModelSha256"] != policy.body_model_sha256
            or source["commandControlContract"] != policy.command_control_contract):
        raise ValueError("source provenance differs from actual loaded policy attributes")
    copied = copy.deepcopy(source)
    copied["sourceRlProvenance"] = copy.deepcopy(source)
    return copied


def pretrain(args):
    from stable_baselines3 import PPO
    from .command_navigation import CommandNavigationEnv, assert_command_policy
    from .command_navigation_training import _source_hashes

    output, dataset = Path(args.output), Path(args.dataset)
    if (output/"policy.zip").exists():
        raise ValueError("pretraining must use a new output directory")
    source_sha, source_hashes = sha256(args.policy), _source_hashes()
    collection = json.loads((dataset/"collection.json").read_text())
    if collection.get("heldOutScenesUsed") is not False or collection.get("failedEpisodesExcluded") is not True:
        raise ValueError("pretraining requires verified train-only teacher episodes")
    if sha256(dataset/"dataset.npz") != collection["datasetSha256"]:
        raise ValueError("teacher dataset differs from its audited collection")
    if collection.get("sourceSha256") != source_hashes:
        raise ValueError("teacher data was collected with a different simulator contract")
    raw = np.load(dataset/"dataset.npz", allow_pickle=False)
    allowed = {index for index, episode in enumerate(collection["episodes"])
               if episode["acceptedForSupervision"] and episode["navigationVerified"]
               and episode["split"] == "train" and episode["sceneSeed"] < 1_000_000
               and episode["physicsSeed"] < 1_000_000}
    if set(raw["episode_indices"].tolist()) - allowed:
        raise ValueError("unqualified teacher episode leaked into supervised data")
    policy = PPO.load(args.policy, device="cpu")
    source_provenance = json.loads((Path(args.policy).parent/"provenance.json").read_text())
    common = inherited_source_provenance(source_provenance, source_sha=source_sha, policy=policy)
    env = CommandNavigationEnv(motor_checkpoint=args.motor_policy,
        motor_hip_yaw_action_limit=validated_motor_limit(args), mode="turn", split="train",
        base_seed=88401, randomize=False, max_episode_seconds=30., avatar_profile=args.avatar_profile)
    try:
        assert_command_policy(policy, env)
        if (collection["avatarProfileSha256"] != policy.avatar_profile_sha256
                or collection["bodyModelSha256"] != policy.body_model_sha256
                or collection["commandControlContract"] != env.control_contract):
            raise ValueError("teacher, checkpoint and motor/body contracts differ")
        initial_rl = int(policy.num_timesteps)
        initial_physics = int(policy.command_physical_control_timesteps)
        fitted = fit_actor(policy, raw["observations"], raw["actions"], epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
        if int(policy.num_timesteps) != initial_rl or int(policy.command_physical_control_timesteps) != initial_physics:
            raise RuntimeError("supervised fitting changed actual RL/physics training counts")
        if source_hashes != _source_hashes() or sha256(args.policy) != source_sha:
            raise RuntimeError("source checkpoint or physics changed during actor fitting")
        policy.command_supervised_pretraining = {**fitted, "sourceCheckpointSha256": source_sha,
            "teacherDatasetSha256": collection["datasetSha256"], "teacherQualifiedEpisodes": len(allowed),
            "teacher": "analytic actual final-goal bearing; no paths or waypoints",
            "heldOutScenesUsed": False, "rlDecisionTimesteps": initial_rl,
            "physicalControlTimesteps": initial_physics}
        output.mkdir(parents=True, exist_ok=True)
        policy.save(output/"policy.zip")
        provenance = {**common, "role": "actor-only supervised warm-start of learned command navigator",
            "algorithm": "PPO lineage + verified train-room actor MSE warm-start",
            "mode": "turn", "sourceCheckpoint": str(args.policy), "sourceCheckpointSha256": source_sha,
            "checkpointSha256": sha256(output/"policy.zip"), "teacherDataset": str(dataset),
            "teacherDatasetSha256": collection["datasetSha256"], "teacherCollection": collection,
            "supervisedPretraining": fitted, "timesteps": initial_rl,
            "actualDecisionTimesteps": initial_rl, "physicalControlTimesteps": initial_physics,
            "phaseTimesteps": 0, "phaseDecisionTimesteps": 0, "phasePhysicalControlTimesteps": 0,
            "phasePhysicsSubsteps": 0, "supervisedRlDecisionTimestepsAdded": 0,
            "supervisedPhysicalControlTimestepsAdded": 0,
            "avatarProfileSha256": policy.avatar_profile_sha256, "bodyModelSha256": policy.body_model_sha256,
            "motorCheckpointSha256": env.motor_checkpoint_sha256, "commandControlContract": env.control_contract,
            "sourceSha256": source_hashes, "pretrainingSourceSha256": sha256(__file__),
            "highLevelInferenceUsesTeacher": False, "heldOutScenesUsedForTraining": False,
            "criticAndLogStdPreservedExactly": True, "ppoOptimizerReset": True,
            "fixedWaypointsSupplied": False, "rootStateOverwrittenDuringStep": False,
            "proficiencyPassed": False, "anyRoomNavigationVerified": False,
            "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
        dump(output/"provenance.json", provenance)
        print(json.dumps({"output": str(output), "fit": fitted,
            "actualRlDecisionTimesteps": initial_rl, "actualPhysicsControlTimesteps": initial_physics}), flush=True)
        return provenance
    finally:
        env.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "fit"):
        sub = commands.add_parser(name)
        sub.add_argument("--motor-policy", required=True)
        sub.add_argument("--motor-hip-yaw-action-limit", type=float,
                         help="Optional explicit hip-yaw residual limit; requires matching transferred policy/data contracts")
        sub.add_argument("--avatar-profile", required=True)
        sub.add_argument("--output", required=True)
        if name == "collect":
            sub.add_argument("--episodes", type=int, default=16)
            sub.add_argument("--scene-seed-base", type=int, default=88401)
            sub.add_argument("--physics-seed-base", type=int, default=89401)
            sub.add_argument("--difficulty-min", type=float, default=0.)
            sub.add_argument("--difficulty-max", type=float, default=.5)
            sub.add_argument("--seconds", type=float, default=30.)
        else:
            sub.add_argument("--policy", required=True)
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--epochs", type=int, default=150)
            sub.add_argument("--batch-size", type=int, default=256)
            sub.add_argument("--learning-rate", type=float, default=3e-4)
            sub.add_argument("--seed", type=int, default=88401)
    return parser


def main():
    args = argument_parser().parse_args()
    collect(args) if args.command == "collect" else pretrain(args)


if __name__ == "__main__":
    main()
