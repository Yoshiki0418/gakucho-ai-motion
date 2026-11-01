"""Train-only physical success qualification and conservative CUDA imitation.

Every attempted teacher rollout is audited at native 50 Hz with sticky 2 ms
contacts. Only trajectories passing all independent navigation gates become
labels. Rejected episodes remain in the manifest and physics accounting.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .command_motor_training import _dump, _sha256, _state_sha256
from .generalization import validate_scene_seed


def make_qualified_env(motor, profile, seed, seconds=30., difficulty=(0., .5), recording=False):
    from .visual_navigation import FROZEN_MOTOR_CHECKPOINT_SHA256
    if _sha256(motor) == FROZEN_MOTOR_CHECKPOINT_SHA256:
        from .visual_training import make_env
        return make_env(motor, profile, seed, seconds, difficulty, recording)
    from .tracked_visual_navigation import TrackedVisualNavigationEnv
    return TrackedVisualNavigationEnv(motor_checkpoint=motor, avatar_profile=profile,
        decision_steps=5, depth_stack=3, record_trajectory=recording, mode="obstacle", split="train",
        base_seed=seed, difficulty_range=difficulty, randomize=True, max_episode_seconds=seconds)


def implementation_sources(motor):
    from .visual_training import source_hashes
    from .visual_navigation import FROZEN_MOTOR_CHECKPOINT_SHA256
    hashes = source_hashes()
    for name in (Path(__file__).name, "visual_refinement.py"):
        hashes[name] = _sha256(Path(__file__).parent/name)
    if _sha256(motor) != FROZEN_MOTOR_CHECKPOINT_SHA256:
        for name in ("tracked_visual_navigation.py", "motor_tracking_training.py"):
            hashes[name] = _sha256(Path(__file__).parent/name)
    return hashes


def compact_audit_frame(frame):
    """Retain actual geometry/rig and all independent audit inputs, not contracts."""
    keep = ("root_position", "root_height", "root_velocity", "upright", "fall",
        "goal_distance", "left_foot_force", "right_foot_force", "navigation_floor_support",
        "navigation_success_seconds", "collision_occurred", "collision_force_newtons")
    stats = {key: copy.deepcopy(frame["stats"][key]) for key in keep}
    stats["navigation"] = {"metrics": copy.deepcopy(frame["stats"]["navigation"]["metrics"])}
    stats["command_motor"] = {"observedPhysicsSubstepsThisEpisode":
        frame["stats"]["command_motor"]["observedPhysicsSubstepsThisEpisode"]}
    return {"time": frame["time"], "task": "walk", "geoms": copy.deepcopy(frame["geoms"]),
            "bodies": copy.deepcopy(frame.get("bodies", [])), "stats": stats}


def make_qualification_trace(env, scene_seed, physics_seed):
    body = env.unwrapped
    return {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
        "task": "walk", "controller": "privileged geometry teacher for physical training qualification",
        "dt": float(body.dt), "scene": copy.deepcopy(env.scene),
        "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
        "motorCheckpointSha256": env.motor_checkpoint_sha256,
        "navigation": {"mode": "obstacle", "split": "train", "sceneSeed": scene_seed,
            "physicsSeed": physics_seed, "privilegedTeacherUsed": True,
            "role": "offline physically qualified supervision; never visual actor inference"},
        "frames": [compact_audit_frame(env.snapshot())]}


def require_qualified_manifest(manifest):
    """Fail closed on mixed splits, unaudited labels or inconsistent counters."""
    if manifest.get("split") != "train" or manifest.get("heldoutUsedForLabels") is not False:
        raise ValueError("qualified supervision must contain train-only physical labels")
    episodes = manifest.get("episodes", [])
    if not episodes or not any(item.get("accepted") for item in episodes):
        raise ValueError("qualified dataset has no verified successful physical episodes")
    for item in episodes:
        validate_scene_seed(item["sceneSeed"], "train")
        if item["physicsSubsteps"] != 10*item["physicalControls"]:
            raise ValueError("qualified physical accounting must preserve exact native substeps")
        quality = item.get("quality", {})
        if item.get("accepted") != bool(quality.get("navigationVerified")):
            raise ValueError("accepted teacher episode must pass independent navigation gates")
        gates = quality.get("gates", {})
        expected = {"survived", "collisionFree", "goalReached", "stopped", "bothFeetSupport",
                    "sustainedStop", "physicalRepeatedGait", "obstacleActuallyBlocksDirectRoute"}
        if item["accepted"] and (set(gates) != expected or not all(value is True for value in gates.values())):
            raise ValueError("all independent gates must pass before accepting a teacher")
    return episodes


def _collect_one(config):
    import torch
    from .visual_teacher import TrainingGeometryTeacher
    from .navigation_evaluation import audit_navigation
    from .command_navigation_evaluation import append_control_frames
    torch.set_num_threads(1)
    seed, physics = config["sceneSeed"], config["physicsSeed"]
    validate_scene_seed(seed, "train")
    env = make_qualified_env(config["motor"], config["profile"], seed, config["seconds"],
                   (config["difficultyMin"], config["difficultyMax"]), recording=True)
    try:
        obs, _ = env.reset(seed=physics, options={"scene_seed": seed, "physics_seed": physics})
        body = env.unwrapped
        initial = {"qposSha256": hashlib.sha256(body.data.qpos.tobytes()).hexdigest(),
                   "qvelSha256": hashlib.sha256(body.data.qvel.tobytes()).hexdigest()}
        teacher = TrainingGeometryTeacher(env, margin=config["margin"], carrot=config["carrot"],
                                           walking_action=config["pace"])
        trace = make_qualification_trace(env, seed, physics)
        inputs, states, labels = [], [], []
        done = False
        controls = substeps = 0
        while not done:
            action = teacher.action()
            inputs.append(obs["depth"].copy()); states.append(obs["state"].copy()); labels.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            count, native = info["command_control_steps"], info["command_physics_substeps"]
            if type(count) is not int or not 1 <= count <= 5 or native != 10*count:
                raise RuntimeError("qualified teacher native physical accounting violated")
            append_control_frames(trace, [compact_audit_frame(f) for f in env.last_control_frames],
                                  expected_count=count, control_dt=.02)
            controls += count; substeps += native
            done = terminated or truncated
        env.assert_frozen_motor(check_checkpoint=True)
        if substeps != trace["frames"][-1]["stats"]["command_motor"]["observedPhysicsSubstepsThisEpisode"]:
            raise RuntimeError("independent teacher trace does not match native solver counts")
        quality = audit_navigation(trace)
        accepted = bool(quality["navigationVerified"])
        record = {"sceneSeed": seed, "physicsSeed": physics, "split": "train",
            "accepted": accepted, "quality": quality, "initialState": initial,
            "sceneSha256": env.navigation_env._scene_sha256,
            "physicalControls": controls, "physicsSubsteps": substeps, "decisions": len(labels),
            "finalGoalDistance": float(info["goal_distance"]),
            "terminationReason": info["navigation"].get("terminal_reason"),
            "collisionGeoms": info["navigation"]["metrics"].get("obstacle_contact_geom_ids", [])}
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        if accepted:
            prefix = output/f"qualified-{seed}-{physics}"
            dataset = prefix.with_suffix(".npz")
            np.savez_compressed(dataset, depth=np.asarray(inputs, np.float32),
                state=np.asarray(states, np.float32), actions=np.asarray(labels, np.float32))
            trace["navigationQuality"] = quality
            filename = prefix.with_suffix(".json.gz")
            with gzip.open(filename, "wt", encoding="utf8", compresslevel=3) as stream:
                json.dump(trace, stream, allow_nan=False, separators=(",", ":"))
            record.update(dataset=dataset.name, datasetSha256=_sha256(dataset),
                          trace=filename.name, traceSha256=_sha256(filename))
        return record
    finally:
        env.close()


def collect(args):
    from .avatar_profile import load_avatar_profile
    from .navigation_training import _asset_hashes
    if (not 1 <= args.envs <= 16 or args.episodes < 1 or not 0 <= args.seed
            or args.seed+args.episodes >= 1_000_000 or args.seconds <= 0
            or not 0 <= args.difficulty_min <= args.difficulty_max <= 1):
        raise ValueError("invalid train-only teacher collection configuration")
    output = Path(args.output)
    if output.exists():
        raise ValueError("preserve teacher attempts; use a fresh output directory")
    profile = load_avatar_profile(args.avatar_profile)
    sources = implementation_sources(args.motor_policy)
    assets, motor = _asset_hashes(profile), _sha256(args.motor_policy)
    output.mkdir(parents=True)
    manifest = {"version": 1, "split": "train", "heldoutUsedForLabels": False,
        "arguments": vars(args), "sourceSha256": sources, "assetSha256": assets,
        "motorCheckpointSha256": motor, "episodes": [], "ppoDecisionsAdded": 0,
        "qualifiedOnly": True, "privilegedGeometryForOfflineTeacherOnly": True}
    _dump(output/"collection-started.json", manifest)
    jobs = [{"sceneSeed": args.seed+i, "physicsSeed": args.seed+10000+i,
        "motor": args.motor_policy, "profile": profile, "seconds": args.seconds,
        "difficultyMin": args.difficulty_min, "difficultyMax": args.difficulty_max,
        "margin": args.margin, "carrot": args.carrot, "pace": args.pace,
        "output": str(output)} for i in range(args.episodes)]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.envs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_collect_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); manifest["episodes"].append(result)
            manifest["episodes"].sort(key=lambda item: item["sceneSeed"])
            manifest.update(actualPhysicalControls=sum(x["physicalControls"] for x in manifest["episodes"]),
                actualPhysicsSubsteps=sum(x["physicsSubsteps"] for x in manifest["episodes"]),
                acceptedEpisodes=sum(x["accepted"] for x in manifest["episodes"]),
                acceptedExamples=sum(x["decisions"] for x in manifest["episodes"] if x["accepted"]),
                elapsedSeconds=time.monotonic()-start)
            _dump(output/"collection-progress.json", manifest)
            print(json.dumps({"phase": "physical-teacher-qualification", "attempts": len(manifest["episodes"]),
                "accepted": manifest["acceptedEpisodes"], "examples": manifest["acceptedExamples"],
                "sceneSeed": result["sceneSeed"], "success": result["accepted"]}), flush=True)
    if (_sha256(args.motor_policy) != motor or _asset_hashes(profile) != assets
            or any(_sha256(Path(__file__).parent/name) != sha for name, sha in sources.items())):
        raise RuntimeError("teacher collection modified frozen sources, motor or assets")
    manifest["complete"] = True
    _dump(output/"dataset.json", manifest)
    require_qualified_manifest(manifest)
    return manifest


def fit(args):
    import torch
    from stable_baselines3 import PPO
    from .visual_training import depth_dependence
    from .visual_navigation import assert_visual_policy
    from .navigation_evaluation import audit_navigation
    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA required for physically qualified visual learning")
    if args.epochs < 1 or args.batch_size < 2 or not 0 < args.learning_rate <= 1e-4:
        raise ValueError("invalid conservative qualified imitation configuration")
    dataset, output = Path(args.dataset), Path(args.output)
    if output.exists():
        raise ValueError("preserve fitted candidate; use a fresh output directory")
    manifest = json.loads((dataset/"dataset.json").read_text())
    episodes = require_qualified_manifest(manifest)
    source = Path(args.resume); source_sha = _sha256(source)
    torch.set_num_threads(1); torch.cuda.set_device(0)
    model = PPO.load(source, device="cuda")
    probe = make_qualified_env(args.motor_policy, args.avatar_profile, 93601)
    try:
        assert_visual_policy(model, probe)
        if manifest["motorCheckpointSha256"] != probe.motor_checkpoint_sha256:
            raise ValueError("qualified data motor must match the exact physical inference motor")
        inputs = {key: [] for key in ("depth", "state", "actions")}
        for episode in episodes:
            if not episode["accepted"]:
                continue
            path = dataset/episode["dataset"]
            if _sha256(path) != episode["datasetSha256"] or _sha256(dataset/episode["trace"]) != episode["traceSha256"]:
                raise ValueError("qualified dataset or independent physical trace changed")
            with gzip.open(dataset/episode["trace"], "rt", encoding="utf8") as stream:
                trace = json.load(stream)
            if (not audit_navigation(trace)["navigationVerified"]
                    or trace["navigation"]["split"] != "train"
                    or trace["navigation"]["sceneSeed"] != episode["sceneSeed"]
                    or trace["motorCheckpointSha256"] != probe.motor_checkpoint_sha256
                    or trace["bodyModelSha256"] != probe.unwrapped.body_model_sha256
                    or trace["avatarProfileSha256"] != probe.unwrapped.avatar_profile_sha256
                    or len(trace["frames"])-1 != episode["physicalControls"]):
                raise ValueError("qualified physical trace does not match declared native data/body/motor")
            with np.load(path) as data:
                count = episode["decisions"]
                for key, shape in (("depth", (count, 3, 48, 64)),
                                   ("state", (count, 11)), ("actions", (count, 2))):
                    array = data[key]
                    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError("qualified dataset dimensions, type or finite values differ")
                    if key == "depth" and (np.any(array < 0) or np.any(array > 1)):
                        raise ValueError("qualified camera ranges are outside their normalized bounds")
                    if key == "actions" and np.any(np.abs(array) > 1):
                        raise ValueError("qualified actions exceed their physical command contract")
                    if key == "state" and (np.any(array < probe.observation_space["state"].low)
                                           or np.any(array > probe.observation_space["state"].high)):
                        raise ValueError("qualified local state violates the inference observation bounds")
                    inputs[key].append(array)
        inputs = {key: np.concatenate(value) for key, value in inputs.items()}
        obs = {key: torch.as_tensor(inputs[key], device="cuda") for key in ("depth", "state")}
        targets = torch.as_tensor(inputs["actions"], device="cuda")
        rng = np.random.default_rng(args.seed)
        before = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
        inherited = (model.num_timesteps, model._n_updates, model.visual_physical_control_timesteps,
                     model.visual_physics_substeps)
        for name, parameter in model.policy.named_parameters():
            parameter.requires_grad_(name.startswith(("mlp_extractor.policy_net.", "action_net.")))
            parameter.grad = None
        actor = [p for p in model.policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(actor, lr=args.learning_rate)
        model.policy.set_training_mode(False)
        with torch.no_grad():
            features = model.policy.extract_features(obs).detach()
        # Frozen encoder output is computed once on CUDA; every actor update
        # below uses actual accepted camera/body inputs, without teacher maps.
        # Every intermediate checkpoint has the same explicit supervised
        # transition semantics, including a fresh future-PPO optimizer.
        model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
            lr=float(model.lr_schedule(1.)), **model.policy.optimizer_kwargs)
        model.qualified_supervision_contract = {"version": 1, "allEpisodesPhysicallyQualified": True,
            "privilegedLabelsTrainOnly": True, "frozenFeaturesAndCritic": True,
            "ppoAdamResetAfterSupervision": True, "ppoCountersUnchanged": True,
            "sourceCheckpointSha256": source_sha}
        output.mkdir(parents=True)
        sources = implementation_sources(args.motor_policy)
        _dump(output/"started.json", {"arguments": vars(args), "sourceSha256": sources,
            "sourceCheckpointSha256": source_sha, "datasetManifestSha256": _sha256(dataset/"dataset.json"),
            "acceptedPhysicalEpisodes": manifest["acceptedEpisodes"], "examples": len(targets),
            "gpu": torch.cuda.get_device_name(), "device": str(model.device),
            "inheritedPpoCounters": list(inherited), "frozenFeaturesAndCritic": True})
        updates, losses = 0, []
        for epoch in range(args.epochs):
            errors = []
            for indices in np.array_split(rng.permutation(len(targets)), max(1, math.ceil(len(targets)/args.batch_size))):
                latent = model.policy.mlp_extractor.forward_actor(features[indices])
                predicted = model.policy.action_net(latent)
                loss = torch.nn.functional.mse_loss(predicted, targets[indices])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor, .5); optimizer.step()
                updates += 1; errors.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(errors)))
            if epoch % 5 == 0:
                print(json.dumps({"phase": "qualified-cuda-imitation", "epoch": epoch+1, "mse": losses[-1]}), flush=True)
            if (epoch+1) in (5, 15, args.epochs):
                model.save(output/f"epoch-{epoch+1}.zip")
        # A supervised actor update changes the optimization problem. Preserve
        # critic/counters but explicitly reset PPO Adam before future PPO use.
        model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
            lr=float(model.lr_schedule(1.)), **model.policy.optimizer_kwargs)
        model.qualified_supervision_contract = {"version": 1, "allEpisodesPhysicallyQualified": True,
            "privilegedLabelsTrainOnly": True, "frozenFeaturesAndCritic": True,
            "supervisedOptimizerSteps": updates, "ppoAdamResetAfterSupervision": True,
            "ppoCountersUnchanged": True, "sourceCheckpointSha256": source_sha}
        model.save(output/"policy.zip")
        changed = {key: float((value.detach().cpu()-before[key]).abs().max())
                   for key,value in model.policy.state_dict().items() if not torch.equal(value.detach().cpu(),before[key])}
        if not changed or any(not key.startswith(("mlp_extractor.policy_net.", "action_net.")) for key in changed):
            raise RuntimeError("qualified imitation changed frozen features, noise or critic")
        if inherited != (model.num_timesteps, model._n_updates, model.visual_physical_control_timesteps, model.visual_physics_substeps):
            raise RuntimeError("supervised fitting changed inherited PPO/physics counters")
        if _sha256(source) != source_sha or any(_sha256(Path(__file__).parent/name) != sha for name, sha in sources.items()):
            raise RuntimeError("qualified fitting changed frozen input sources")
        report = {"completedUtc": datetime.now(timezone.utc).isoformat(),
            "device": str(model.device), "qualifiedTeacherExamples": len(targets),
            "qualifiedTeacherEpisodes": manifest["acceptedEpisodes"], "optimizerSteps": updates,
            "lossMse": losses, "changedActorTensors": changed, "featuresCriticAndStdByteIdentical": True,
            "ppoDecisionsAdded": 0, "physicsControlsAddedByFitting": 0,
            "inheritedPpoCounters": list(inherited), "ppoAdamResetAfterSupervision": True,
            "checkpointSha256": _sha256(output/"policy.zip"), "sourceCheckpointSha256": source_sha,
            "wholeControllerVisionOnly": False, "publishedToLiveAvatar": False, "phase2Complete": False}
        _dump(output/"depth-dependence.json", depth_dependence(model, {k:inputs[k][:128] for k in ("depth", "state")}))
        _dump(output/"training.json", report)
        return report
    finally:
        probe.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "fit"):
        sub = commands.add_parser(name)
        sub.add_argument("--motor-policy", required=True)
        sub.add_argument("--avatar-profile", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--seed", type=int, default=93100 if name == "collect" else 93701)
        if name == "collect":
            sub.add_argument("--episodes", type=int, default=96)
            sub.add_argument("--envs", type=int, default=8)
            sub.add_argument("--seconds", type=float, default=30.)
            sub.add_argument("--difficulty-min", type=float, default=0.)
            sub.add_argument("--difficulty-max", type=float, default=.5)
            sub.add_argument("--margin", type=float, default=.5)
            sub.add_argument("--carrot", type=float, default=.55)
            sub.add_argument("--pace", type=float, default=-.6)
        else:
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--resume", required=True)
            sub.add_argument("--epochs", type=int, default=30)
            sub.add_argument("--batch-size", type=int, default=256)
            sub.add_argument("--learning-rate", type=float, default=2e-5)
    return parser


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    (collect if arguments.command == "collect" else fit)(arguments)
