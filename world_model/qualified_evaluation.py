"""Spawned, CPU-only matched cohort evaluation with complete physical traces.

All trials retain their actual initial qpos/qvel and native 50 Hz trajectory.
The original visual evaluator supplies its fixed-goal, frozen-motor and 2 ms
solver guards. A second audit reloads every written trace independently.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from .command_motor_training import _dump, _sha256
from .command_navigation_evaluation import _scene_seeds
from .generalization import _difficulty_range


def array_sha256(array):
    """Include dtype and shape so physically distinct arrays never alias."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{value.dtype}:{value.shape}:".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def scene_sha256(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sources():
    from .visual_training import source_hashes
    result = source_hashes()
    for name in ("visual_refinement.py", "qualified_evaluation.py",
                 "tracked_visual_navigation.py", "motor_tracking_training.py",
                 "motor_cycle_tracking_training.py"):
        result[name] = _sha256(Path(__file__).parent/name)
    return result


def independently_audit(record, output):
    """Reopen native output, independently recalculate every qualification gate."""
    from .navigation_evaluation import audit_navigation
    output = Path(output)
    episode = record["episodes"][0]
    recording = record["recordings"][0]
    path = output/recording["filename"]
    if _sha256(path) != recording["traceSha256"]:
        raise RuntimeError("stored native trace checksum differs")
    trace = json.loads(path.read_text())
    quality = audit_navigation(trace)
    if quality != episode["quality"] or quality != trace["navigationQuality"]:
        raise RuntimeError("independent native audit disagrees with evaluation gates")
    times = np.asarray([frame["time"] for frame in trace["frames"]])
    if len(times)-1 != episode["physicalControlSteps"] or not np.allclose(
            np.diff(times), .02, atol=1e-8, rtol=0.):
        raise RuntimeError("native trace must contain every actual 20 ms frame")
    if episode["physical2msSubsteps"] != (len(times)-1)*10:
        raise RuntimeError("native physical integration counters disagree")
    initial = json.loads((output/"initial-state.json").read_text())
    with np.load(output/"initial-state.npz", allow_pickle=False) as arrays:
        for key in ("qpos", "qvel", "depth", "state", "goal"):
            if array_sha256(arrays[key]) != initial[key+"Sha256"]:
                raise RuntimeError(f"initial {key} checksum differs")
        np.testing.assert_array_equal(arrays["qpos"][:3], trace["frames"][0]["stats"]["root_position"])
        np.testing.assert_array_equal(arrays["qvel"][:3], trace["frames"][0]["stats"]["root_velocity"])
        depth_file = output/recording["depthFilename"]
        with np.load(depth_file, allow_pickle=False) as depths:
            np.testing.assert_array_equal(arrays["depth"], depths["actualDepthStack"][0])
            np.testing.assert_array_equal(arrays["state"], depths["state11"][0])
    if (initial["sceneSha256"] != episode["sceneSha256"]
            or initial["sceneSha256"] != scene_sha256(trace["scene"])):
        raise RuntimeError("initial scene does not match actual native trace")
    for key in ("avatarProfileSha256", "bodyModelSha256"):
        if initial[key] != trace[key] or trace[key] != record[key]:
            raise RuntimeError("same-avatar body checksum differs")
    actor_file = output/"first-actor-input-action.npz"
    if actor_file.exists():
        actor = json.loads((output/"first-actor-input-action.json").read_text())
        with np.load(actor_file,allow_pickle=False) as actual:
            for key in ("depth","state","action"):
                if array_sha256(actual[key])!=actor[key+"Sha256"]:
                    raise RuntimeError("actual first actor input/action checksum differs")
            with np.load(output/recording["depthFilename"],allow_pickle=False) as depths:
                np.testing.assert_array_equal(actual["depth"],depths["policyDepthStack"][0])
                np.testing.assert_array_equal(actual["state"],depths["state11"][0])
            action=actual["action"]
            if action.shape!=(2,) or not np.isfinite(action).all() or np.max(np.abs(action))>1.:
                raise RuntimeError("actual first actor action is invalid")
            first=trace["frames"][1]["stats"]
            requested=first["command_motor"]["nextCommand"]["requestedHeadingOffsetRadians"]
            bound=trace["navigation"]["visualControlContract"]["maxHeadingRadians"]
            if (requested!=float(action[0])*bound
                    or first["visual_navigation"]["previousLookaheadAction"]!=float(action[1])):
                raise RuntimeError("recorded first action differs from the actual consumed physical command")
    result = {"nativeTraceSha256": _sha256(path), "allNativeFramesReaudited": len(times),
        "quality": quality, "initialState": initial,
        "actualInitialObservationMatchesTraceAndDepthSamples": True,
        "actualInitialRootStateMatchesNativeTrace": True}
    _dump(output/"independent-audit.json", result)
    return result


def _evaluate_one(arguments):
    """One isolated worker loads CPU policies and records the actual reset."""
    import torch
    from stable_baselines3 import PPO
    from . import visual_navigation
    from .visual_evaluation import evaluate
    torch.set_num_threads(1)
    output = Path(arguments["output"])
    output.mkdir(parents=True, exist_ok=False)
    original = visual_navigation.VisualNavigationEnv
    if arguments["motor_sha"] != visual_navigation.FROZEN_MOTOR_CHECKPOINT_SHA256:
        from .tracked_visual_navigation import TrackedVisualNavigationEnv
        original = TrackedVisualNavigationEnv
    captures = []

    class RecordingInitialState(original):
        def reset(self, **kwargs):
            observation, info = super().reset(**kwargs)
            body = self.unwrapped
            # Copies only: neither world state nor actor input is changed.
            arrays = {"qpos": body.data.qpos.copy(), "qvel": body.data.qvel.copy(),
                "depth": observation["depth"].copy(), "state": observation["state"].copy(),
                "goal": body.goal.copy()}
            initial = {"sceneSeed": arguments["scene_seed"], "physicsSeed": arguments["physics_seed"],
                "sceneSha256": scene_sha256(self.scene), "physicalTimeSeconds": float(body.data.time),
                "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
                **{key+"Sha256": array_sha256(value) for key,value in arrays.items()}}
            np.savez_compressed(output/"initial-state.npz", **arrays)
            _dump(output/"initial-state.json", initial)
            captures.append(initial)
            return observation, info

    policy = PPO.load(arguments["policy"], device="cpu")
    predict = policy.predict
    first_actions = []

    def recording_predict(observation, **kwargs):
        result = predict(observation, **kwargs)
        if not first_actions:
            action = np.asarray(result[0]).copy()
            arrays = {"action":action, "depth":observation["depth"].copy(),
                      "state":observation["state"].copy()}
            np.savez_compressed(output/"first-actor-input-action.npz", **arrays)
            metadata = {"scope":"actual first predict result consumed by the evaluator",
                "inferenceDevice":"cpu", "torchThreads":1, "deterministic":True,
                "checkpointSha256":arguments["policy_sha"],
                **{key+"Sha256":array_sha256(value) for key,value in arrays.items()},
                "action":action.tolist(),"blankDepthAblation":arguments["blank_depth"]}
            _dump(output/"first-actor-input-action.json",metadata)
            first_actions.append(metadata)
        return result

    policy.predict = recording_predict
    with patch.object(visual_navigation, "VisualNavigationEnv", RecordingInitialState):
        record = evaluate(policy, motor_checkpoint=arguments["motor"],
            scene_seeds=[arguments["scene_seed"]], physics_seed_base=arguments["physics_seed"],
            difficulty_range=arguments["difficulty"], seconds=arguments["seconds"],
            avatar_profile=arguments["avatar_profile"], output=output,
            checkpoint_sha256=arguments["policy_sha"], blank_depth=arguments["blank_depth"])
    if len(captures) != 1 or len(first_actions)!=1:
        raise RuntimeError("one physical reset is required per matched episode")
    audit = independently_audit(record, output)
    return {"episode": record["episodes"][0], "initialState": captures[0],
        "audit": {key:value for key,value in audit.items() if key not in ("quality", "initialState")},
        "output": str(output), "firstActorInputAction":first_actions[0],
        "inferenceDevice":"cpu", "avatarProfileSha256": record["avatarProfileSha256"],
        "bodyModelSha256": record["bodyModelSha256"], "sourceSha256": record["sourceSha256"]}


def _validate(scene_seeds, physics_seed_base, difficulty_range, seconds, envs):
    seeds = _scene_seeds(scene_seeds)
    difficulty = _difficulty_range(difficulty_range)
    if any(seed >= 1_800_000 for seed in seeds):
        raise ValueError("qualified comparison is development only; preserve reserved final cohort")
    if type(physics_seed_base) is not int or not 0 <= physics_seed_base < 2**32-len(seeds):
        raise ValueError("distinct integer physics seeds are required")
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if type(envs) is not int or not 1 <= envs <= 16:
        raise ValueError("CPU worker count must be an integer from 1 to 16")
    return seeds, difficulty


def evaluate_cohort(policy_checkpoint, output, *, motor_checkpoint,
        avatar_profile="frontend/public/world-model/gakucho-avatar-profile.json",
        scene_seeds=range(1_182_000, 1_182_032), physics_seed_base=3_182_000,
        difficulty_range=(0., .5), seconds=30., envs=4, blank_depth=False):
    seeds, difficulty = _validate(scene_seeds, physics_seed_base, difficulty_range, seconds, envs)
    if not isinstance(blank_depth, bool):
        raise ValueError("blank_depth must be boolean")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("preserve every prior comparison; use a new output directory")
    policy, motor, profile = (Path(path).resolve() for path in
                             (policy_checkpoint, motor_checkpoint, avatar_profile))
    policy_sha, motor_sha, profile_sha = map(_sha256, (policy,motor,profile))
    sources, start = _sources(), time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    archive=output/"source"
    archive.mkdir()
    for name,digest in sources.items():
        original=Path(__file__).parent/name
        if _sha256(original)!=digest:
            raise RuntimeError("source changed before evaluation archive")
        shutil.copy2(original,archive/name)
    _dump(output/"started.json", {"startedUtc":datetime.now(timezone.utc).isoformat(),
        "policyCheckpoint":str(policy),"checkpointSha256":policy_sha,
        "motorCheckpointSha256":motor_sha,"avatarProfileFileSha256":profile_sha,
        "sceneSeeds":seeds,"physicsSeeds":[physics_seed_base+i for i in range(len(seeds))],
        "sourceSha256":sources,"processStartMethod":"spawn","device":"cpu",
        "workers":envs,"seconds":seconds,"difficultyRange":list(difficulty),
        "blankDepthAblation":blank_depth,"trainingUpdates":0,"evaluationRole":"development",
        "candidateSelectionMustNotUseThisCohort":True})
    arguments = [{"policy":str(policy),"policy_sha":policy_sha,"motor":str(motor),"motor_sha":motor_sha,
        "avatar_profile":str(profile),"scene_seed":seed,"physics_seed":physics_seed_base+i,
        "difficulty":difficulty,"seconds":seconds,"blank_depth":blank_depth,
        "output":str(output/f"scene-{seed}")} for i,seed in enumerate(seeds)]
    results = []
    with ProcessPoolExecutor(max_workers=envs,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_evaluate_one, argument):argument["scene_seed"] for argument in arguments}
        for future in as_completed(futures):
            results.append(future.result())
            episodes = [item["episode"] for item in results]
            _dump(output/"progress.json",{"completedEpisodes":len(episodes),"totalEpisodes":len(seeds),
                "verified":sum(item["navigationVerified"] for item in episodes),
                "elapsedSeconds":time.perf_counter()-start,
                "completedSceneSeeds":sorted(item["sceneSeed"] for item in episodes)})
    results.sort(key=lambda result:result["episode"]["sceneSeed"])
    if _sources()!=sources or _sha256(policy)!=policy_sha or _sha256(motor)!=motor_sha or _sha256(profile)!=profile_sha:
        raise RuntimeError("source, policy, motor or avatar changed during locked cohort evaluation")
    episodes = [item["episode"] for item in results]
    for key in ("avatarProfileSha256","bodyModelSha256"):
        if len({item[key] for item in results}) != 1:
            raise RuntimeError("cohort contains different physical bodies")
    result = {"version":1,"completedUtc":datetime.now(timezone.utc).isoformat(),
        "checkpointSha256":policy_sha,"motorCheckpointSha256":motor_sha,
        "sourceSha256":sources,"avatarProfileFileSha256":profile_sha,
        "avatarProfileSha256":results[0]["avatarProfileSha256"],"bodyModelSha256":results[0]["bodyModelSha256"],
        "evaluationRole":"development","reservedFinalCohortUsed":False,"proficiencyPassed":False,
        "candidateSelectionMustNotUseThisCohort":True,"trainingUpdates":0,"device":"cpu",
        "episodeCount":len(episodes),"verified":sum(e["navigationVerified"] for e in episodes),
        "collisionFree":sum(e["collisionFree"] for e in episodes),"survivors":sum(not e["fell"] for e in episodes),
        "physicalGaitPass":sum(e["walkingVerified"] for e in episodes),
        "totalDecisionSteps":sum(e["decisionSteps"] for e in episodes),
        "totalPhysicalControls":sum(e["physicalControlSteps"] for e in episodes),
        "totalPhysicsSubsteps":sum(e["physical2msSubsteps"] for e in episodes),
        "elapsedSeconds":time.perf_counter()-start,"difficultyRange":list(difficulty),"seconds":seconds,
        "blankDepthAblation":blank_depth,"episodes":episodes,"records":results,
        "everyEpisodeHasCompleteNativeTrace":True,"independentNativeAudits":len(results),
        "initialStateScope":"actual reset qpos/qvel/depth/state/goal copied before first action, all episodes",
        "scope":"one blocking furniture item, one goal, flat floor; not arbitrary homes or stairs",
        "phase2Complete":False,"anyRoomNavigationVerified":False,"publishedToLiveAvatar":False}
    _dump(output/"evaluation.json", result)
    return result


def compare_cohorts(before, after, *, allow_motor_transfer=False):
    """Reject unmatched physical cohorts before reporting candidate differences."""
    if not isinstance(allow_motor_transfer,bool):
        raise ValueError("explicit motor transfer comparison flag must be boolean")
    for key in ("avatarProfileSha256","bodyModelSha256","difficultyRange","seconds","device"):
        if before[key] != after[key]:
            raise ValueError(f"matched cohort {key} differs")
    motor_changed=before["motorCheckpointSha256"]!=after["motorCheckpointSha256"]
    if motor_changed and not allow_motor_transfer:
        raise ValueError("matched cohort motorCheckpointSha256 differs; explicit whole-controller transfer comparison required")
    left = {record["episode"]["sceneSeed"]:record for record in before["records"]}
    right = {record["episode"]["sceneSeed"]:record for record in after["records"]}
    if set(left)!=set(right):
        raise ValueError("matched cohort scene seeds differ")
    for seed in left:
        for key in ("physicsSeed","sceneSha256","qposSha256","qvelSha256","depthSha256","stateSha256","goalSha256"):
            if left[seed]["initialState"][key]!=right[seed]["initialState"][key]:
                raise ValueError(f"scene {seed} initial {key} differs")
    return {"exactInitialQposQvelGoalDepthStateAndSceneMatched":True,"episodeCount":len(left),
        "beforeMotorCheckpointSha256":before["motorCheckpointSha256"],
        "afterMotorCheckpointSha256":after["motorCheckpointSha256"],
        "motorDynamicsChanged":motor_changed,"explicitMotorTransferComparison":allow_motor_transfer,
        "inferenceDevice":before["device"],
        "attributionScope":"whole-controller comparison; cannot isolate high versus low learning effects" if motor_changed else "same low-controller checkpoint",
        "beforeVerified":before["verified"],"afterVerified":after["verified"],
        "gainedSceneSeeds":[seed for seed in left if not left[seed]["episode"]["navigationVerified"] and right[seed]["episode"]["navigationVerified"]],
        "regressedSceneSeeds":[seed for seed in left if left[seed]["episode"]["navigationVerified"] and not right[seed]["episode"]["navigationVerified"]],
        "evaluationCountsExcludedFromTraining":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",required=True)
    parser.add_argument("--motor-policy",required=True)
    parser.add_argument("--avatar-profile",default="frontend/public/world-model/gakucho-avatar-profile.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--scene-seed-base",type=int,default=1_182_000)
    parser.add_argument("--physics-seed-base",type=int,default=3_182_000)
    parser.add_argument("--episodes",type=int,default=32)
    parser.add_argument("--envs",type=int,default=4)
    parser.add_argument("--seconds",type=float,default=30.)
    parser.add_argument("--difficulty-min",type=float,default=0.)
    parser.add_argument("--difficulty-max",type=float,default=.5)
    parser.add_argument("--blank-depth",action="store_true")
    args=parser.parse_args()
    result=evaluate_cohort(args.policy,args.output,motor_checkpoint=args.motor_policy,
        avatar_profile=args.avatar_profile,scene_seeds=range(args.scene_seed_base,args.scene_seed_base+args.episodes),
        physics_seed_base=args.physics_seed_base,difficulty_range=(args.difficulty_min,args.difficulty_max),
        seconds=args.seconds,envs=args.envs,blank_depth=args.blank_depth)
    print(json.dumps({key:result[key] for key in ("episodeCount","verified","survivors","collisionFree","elapsedSeconds")}))


if __name__=="__main__":
    main()
