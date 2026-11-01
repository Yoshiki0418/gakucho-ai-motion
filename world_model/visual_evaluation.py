"""Held-out visual-policy evaluation using every native 20 ms control frame.

The learned actor receives depth/state only. Scene geometry is retained solely
for replay and the independent contact/gait/arrival/stop audit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .body_contract import trace_body_metadata
from .command_navigation_evaluation import _scene_seeds, append_control_frames
from .generalization import _difficulty_range
from .navigation_evaluation import audit_navigation
from .training import dump_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(policy, *, motor_checkpoint, mode="obstacle",
             scene_seeds=range(1_170_200, 1_170_208), physics_seed_base=3_170_200,
             difficulty_range=(0., .15), seconds=30., avatar_profile=None,
             output=None, checkpoint_sha256=None, blank_depth=False,
             decision_steps=5, depth_config=None, depth_stack=3,
             max_heading_radians=.45, max_lookahead_m=.5,
             heading_slew_radians_per_second=.175):
    from .visual_navigation import VisualNavigationEnv, assert_visual_policy

    # Validate the complete cohort before opening a checkpoint or a simulator.
    seeds = _scene_seeds(scene_seeds)
    difficulty_range = _difficulty_range(difficulty_range)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("visual evaluation seconds must be positive and finite")
    if (isinstance(physics_seed_base, bool) or not isinstance(physics_seed_base, int)
            or not 0 <= physics_seed_base < 2**32-len(seeds)):
        raise ValueError("visual evaluation requires valid distinct integer physics seeds")
    if not isinstance(blank_depth, bool):
        raise ValueError("blank_depth must be boolean")
    motor = Path(motor_checkpoint)
    motor = motor if motor.suffix == ".zip" else motor.with_suffix(".zip")
    motor_sha = _sha(motor)
    final_cohort = all(1_800_000 <= seed < 2_000_000 for seed in seeds)
    output = Path(output) if output else None
    if output and (output/"evaluation.json").exists():
        raise ValueError("preserve completed visual evaluation; use a new output directory")
    episodes, recordings = [], []
    for index, seed in enumerate(seeds):
        env = VisualNavigationEnv(motor_checkpoint=motor, record_trajectory=True,
            decision_steps=decision_steps, depth_config=depth_config, depth_stack=depth_stack,
            max_heading_radians=max_heading_radians, max_lookahead_m=max_lookahead_m,
            heading_slew_radians_per_second=heading_slew_radians_per_second,
            mode=mode, split="heldout", base_seed=seed, difficulty_range=difficulty_range,
            randomize=True, max_episode_seconds=seconds, avatar_profile=avatar_profile)
        try:
            observation, _ = env.reset(seed=physics_seed_base+index, options={
                "scene_seed":seed, "physics_seed":physics_seed_base+index})
            assert_visual_policy(policy, env)
            env.assert_frozen_motor(check_checkpoint=True)
            body, scene = env.motor_env.unwrapped, copy.deepcopy(env.scene)
            observation_contract, control_contract = copy.deepcopy(env.observation_contract),copy.deepcopy(env.control_contract)
            goal_object, goal_value = body.goal,body.goal.copy()
            trace = {"version":1, "sizeConvention":"mujoco", "geometryScope":"complete-world",
                "dt":float(body.dt), "task":"walk", "scene":scene,
                "gravity":body.model.opt.gravity.tolist(), "motionReference":"cached T2M BVH",
                "physicalFootstepPrior":True,
                "controller":"learned depth/state visual navigator with frozen PPO joint motor" if policy else "zero visual-command baseline with frozen PPO joint motor",
                "navigation":{"mode":mode, "sceneSeed":seed, "physicsSeed":physics_seed_base+index,
                    "checkpointSha256":checkpoint_sha256, "motorCheckpointSha256":motor_sha,
                    "split":"heldout", "evaluationRole":"final" if seed >= 1_800_000 else "development",
                    "difficultyRange":list(difficulty_range), "visualObservationContract":observation_contract,
                    "visualControlContract":control_contract, "navigatorActionCount":2,
                    "motorObservationCount":110,"motorActionCount":20,
                    "motorTrainingTimesteps":int(env._motor.num_timesteps),
                    "maximumDecisionDtSeconds":decision_steps*float(body.dt),
                    "motorFrozen":True,"rootPositionOverwrittenDuringStep":False,
                    "fixedWaypointsSupplied":False,"privilegedTeacherUsed":False,
                    "sceneGraphOrIdealRaysExposedToVisualPolicy":False,
                    "blankDepthAblation":blank_depth,
                    "blankDepthValue":"all far/no-hit (1.0); state and actual camera captures unchanged" if blank_depth else None},
                "frames":[env.snapshot()]}
            trace.update(trace_body_metadata(body))
            done = truncated = False
            controls = substeps = decisions = 0
            total_reward = maximum_external_force = 0.
            depth_samples, policy_samples, state_samples, sample_times, depth_metadata = [],[],[],[],[]
            while not (done or truncated):
                # Ablation never mutates the environment observation or stack.
                actor_observation = {key:value.copy() for key,value in observation.items()}
                if blank_depth:
                    actor_observation["depth"].fill(1.)
                if len(depth_samples) < 8:
                    depth_samples.append(observation["depth"].copy())
                    policy_samples.append(actor_observation["depth"].copy())
                    state_samples.append(observation["state"].copy())
                    sample_times.append(float(body.data.time))
                    depth_metadata.append(copy.deepcopy(env.depth_metadata))
                action = np.zeros(2,dtype=np.float32) if policy is None else policy.predict(actor_observation,deterministic=True)[0]
                observation,reward,done,truncated,info = env.step(action)
                count, native = info.get("command_control_steps"),info.get("command_physics_substeps")
                if (isinstance(count,bool) or not isinstance(count,int) or not 1 <= count <= decision_steps
                        or isinstance(native,bool) or not isinstance(native,int) or native != count*10):
                    raise ValueError("visual evaluation requires exact actual20ms and actual2ms counts")
                if body.goal is not goal_object or not np.array_equal(body.goal,goal_value):
                    raise RuntimeError("visual evaluation changed its actual final goal")
                append_control_frames(trace,env.last_control_frames,expected_count=count,control_dt=body.dt)
                maximum_external_force = max(maximum_external_force,float(np.max(np.abs(body.data.qfrc_applied))),
                                             float(np.max(np.abs(body.data.xfrc_applied))))
                controls += count; substeps += native; decisions += 1; total_reward += float(reward)
            if maximum_external_force != 0.:
                raise RuntimeError("visual navigation evaluation received an external force")
            env.assert_frozen_motor(check_checkpoint=True)
            if controls != len(trace["frames"])-1 or substeps != 10*controls:
                raise ValueError("visual evaluation native trace and physical counts disagree")
            final = trace["frames"][-1]["stats"]
            if final["command_motor"]["observedPhysicsSubstepsThisEpisode"] != substeps:
                raise ValueError("visual evaluation counters differ from actual native solver observer")
            quality = audit_navigation(trace)
            trace["navigationQuality"] = quality
            episode = {"sceneSeed":seed,"physicsSeed":physics_seed_base+index,
                "sceneSha256":hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
                "seconds":float(body.data.time),"return":total_reward,"decisionSteps":decisions,
                "physicalControlSteps":controls,"physicalControlFrameCount":len(trace["frames"])-1,
                "physical2msSubsteps":substeps,"fell":bool(final["fall"]),
                "arrivalStopped":all(quality["gates"][key] for key in ("goalReached","stopped","bothFeetSupport","sustainedStop")),
                "collisionFree":quality["gates"]["collisionFree"],"walkingVerified":quality["gates"]["physicalRepeatedGait"],
                "navigationVerified":quality["navigationVerified"],"quality":quality,"final":final,
                "externalForceMaximum":maximum_external_force,"actualFinalGoalFixed":True,
                "motorFrozenVerified":True,"depthSampleCount":len(depth_samples),"blankDepthAblation":blank_depth}
            episodes.append(episode)
            kind = "verified" if quality["navigationVerified"] else "failed"
            if output and (index == 0 or not any(item["kind"] == kind for item in recordings)):
                prefix = f"stage2-visual-{mode}-{seed}-{kind}"+("-blank-depth" if blank_depth else "")
                depth_name = prefix+"-first-depths.npz"
                output.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output/depth_name,actualDepthStack=np.stack(depth_samples),policyDepthStack=np.stack(policy_samples),
                                    state11=np.stack(state_samples),physicalTimeSeconds=np.asarray(sample_times,dtype=np.float64))
                trace["visualDepthSamples"] = {"filename":depth_name,"sha256":_sha(output/depth_name),
                    "sampleCount":len(depth_samples),"scope":"first up to eight actual high-level actor inputs; no resizing or extra camera captures",
                    "cameraMetadata":depth_metadata,"blankDepthAblation":blank_depth}
                filename = prefix+".json"
                dump_json(output/filename,trace)
                recordings.append({"kind":kind,"sceneSeed":seed,"filename":filename,"traceSha256":_sha(output/filename),
                                   "depthFilename":depth_name,"depthSha256":_sha(output/depth_name)})
            print(json.dumps({"sceneSeed":seed,"seconds":episode["seconds"],"goalDistance":final["goal_distance"],
                "navigationVerified":episode["navigationVerified"],"failedGates":quality["failedGates"],"blankDepth":blank_depth}),flush=True)
        finally:
            env.close()
    if _sha(motor) != motor_sha:
        raise RuntimeError("frozen motor checkpoint changed during visual evaluation")
    rate = sum(e["navigationVerified"] for e in episodes)/len(episodes)
    result = {"version":1,"mode":mode,"episodeCount":len(episodes),"checkpointSha256":checkpoint_sha256,
        "motorCheckpointSha256":motor_sha,"visualObservationContract":observation_contract,"visualControlContract":control_contract,
        "motorTrainingTimesteps":int(env._motor.num_timesteps),"avatarProfileSha256":body.avatar_profile_sha256,
        "bodyModelSha256":body.body_model_sha256,"survivalRate":sum(not e["fell"] for e in episodes)/len(episodes),
        "collisionFreeRate":sum(e["collisionFree"] for e in episodes)/len(episodes),"arrivalStoppedRate":sum(e["arrivalStopped"] for e in episodes)/len(episodes),
        "physicalGaitPassRate":sum(e["walkingVerified"] for e in episodes)/len(episodes),"navigationVerifiedRate":rate,
        "totalDecisionSteps":sum(e["decisionSteps"] for e in episodes),"totalPhysicalControls":sum(e["physicalControlSteps"] for e in episodes),
        "totalPhysicsSubsteps":sum(e["physical2msSubsteps"] for e in episodes),
        "evaluationRole":"final" if final_cohort else "development",
        "proficiencyPassed":policy is not None and not blank_depth and final_cohort and len(episodes)>=32 and rate>=.9,
        "blankDepthAblation":blank_depth,"blankDepthValue":1. if blank_depth else None,
        "difficultyRange":list(difficulty_range),"seconds":seconds,"episodes":episodes,"recordings":recordings,
        "privilegedTeacherUsed":False,"fixedWaypointsSupplied":False,"anyRoomNavigationVerified":False,"stairsVerified":False,
        "scope":"depth/state actor on one-goal flat-floor held-out rooms with one blocking obstacle; native gait/contact/sustained-stop audit",
        "sourceSha256":{name:_sha(Path(__file__).parent/name) for name in ("visual_evaluation.py","visual_navigation.py","visual_policy.py",
            "depth_perception.py","command_motor.py","command_gait.py","navigation_evaluation.py","navigation_scenes.py","gait_audit.py")}}
    if output:
        dump_json(output/"evaluation.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",required=True)
    parser.add_argument("--motor-policy",required=True)
    parser.add_argument("--avatar-profile",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=("turn","obstacle"),default="obstacle")
    parser.add_argument("--scene-seed-base",type=int,default=1_170_200)
    parser.add_argument("--physics-seed-base",type=int,default=3_170_200)
    parser.add_argument("--episodes",type=int,default=8)
    parser.add_argument("--seconds",type=float,default=30.)
    parser.add_argument("--difficulty-min",type=float,default=0.)
    parser.add_argument("--difficulty-max",type=float,default=.15)
    parser.add_argument("--blank-depth",action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    path = Path(args.policy)
    path = path if path.suffix == ".zip" else path.with_suffix(".zip")
    policy = PPO.load(path,device="cpu")
    result = evaluate(policy,motor_checkpoint=args.motor_policy,mode=args.mode,
        scene_seeds=range(args.scene_seed_base,args.scene_seed_base+args.episodes),physics_seed_base=args.physics_seed_base,
        difficulty_range=(args.difficulty_min,args.difficulty_max),seconds=args.seconds,
        avatar_profile=args.avatar_profile,output=args.output,checkpoint_sha256=_sha(path),blank_depth=args.blank_depth)
    print(json.dumps({key:result[key] for key in ("episodeCount","navigationVerifiedRate","survivalRate","proficiencyPassed")}))


if __name__ == "__main__":
    main()
