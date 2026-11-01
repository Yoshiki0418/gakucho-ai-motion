"""Train-only physical turn and brake probes of the immutable visual motor.

This diagnostic applies ordinary declared high actions. It never changes the
root, physical state, goal, model or command bounds and performs no learning.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np


def summarize_samples(samples, stop_seconds=None):
    """Summarize actual measurements; absent brake completion remains absent."""
    times = np.asarray([s["time"] for s in samples])
    headings = np.unwrap([s["heading"] for s in samples])
    speeds = np.asarray([s["speed"] for s in samples])
    active = (times >= 3.) & (times <= (stop_seconds or times[-1]))
    indices = np.flatnonzero(active)
    yaw_rate = (float((headings[indices[-1]]-headings[indices[0]]) /
                      (times[indices[-1]]-times[indices[0]])) if len(indices)>1 else None)
    speed = float(np.mean(speeds[active])) if np.any(active) else None
    result = {"seconds":float(times[-1]), "headingChangeRadians":float(headings[-1]-headings[0]),
              "steadyYawRateRadiansPerSecond":yaw_rate,"steadySpeedMetersPerSecond":speed,
              "steadyRadiusMeters":abs(speed/yaw_rate) if yaw_rate and speed is not None else None,
              "minimumUpright":min(s["upright"] for s in samples),
              "maximumExternalForce":max(s["externalForce"] for s in samples)}
    if stop_seconds is not None:
        start = int(np.searchsorted(times,stop_seconds))
        stopped = next((i for i in range(start,len(samples))
                        if times[i]>=stop_seconds+.5 and np.all(speeds[(times>=times[i]-.5)&(times<=times[i])]<.1)),None)
        start_xy = np.asarray(samples[start]["positionXY"])
        positions = np.asarray([sample["positionXY"] for sample in samples[start:]])
        result.update(stopRequestedAtSeconds=stop_seconds,
            sustainedBelowPointOneMetersPerSecondAt=None if stopped is None else float(times[stopped]),
            displacementAfterStopMeters=float(np.linalg.norm(np.asarray(samples[-1]["positionXY"])-start_xy)),
            maximumExcursionAfterStopMeters=float(np.max(np.linalg.norm(positions-start_xy,axis=1))),
            sampledPathLengthAfterStopMeters=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()),
            finalSpeedMetersPerSecond=float(speeds[-1]))
    return result


def run_probe(spec):
    from .tracked_visual_navigation import make_visual_env
    name, heading, stop, motor = spec[:4]
    mixture=bool(spec[4]) if len(spec)>4 else False
    if mixture:
        from .motor_expert_mixture import ExpertMixtureVisualNavigationEnv
        make_visual_env=ExpertMixtureVisualNavigationEnv
    env = make_visual_env(motor_checkpoint=motor, mode="turn",split="train",base_seed=90301,
        difficulty_range=(0.,0.),randomize=False,max_episode_seconds=8.)
    samples=[]
    try:
        env.reset(seed=91301,options={"scene_seed":90301,"physics_seed":91301})
        body=env.unwrapped
        initial_goal=body.goal.copy(); identity=body.goal
        terminated=truncated=False; controls=substeps=0
        while not (terminated or truncated):
            pace=-1. if stop is not None and body.data.time>=stop-1e-8 else -.6
            _,_,terminated,truncated,info=env.step(np.asarray([heading,pace],dtype=np.float32))
            controls+=info["command_control_steps"];substeps+=info["command_physics_substeps"]
            rotation=body.data.xmat[body.pelvis_body_id].reshape(3,3)
            samples.append({"time":float(body.data.time),"heading":math.atan2(rotation[1,0],rotation[0,0]),
                "positionXY":body.data.qpos[:2].tolist(),"speed":float(np.linalg.norm(body.data.qvel[:2])),
                "upright":float(info["upright"]),"effectiveHeading":float(env.motor_env._effective_heading_offset),
                "consumedYawTarget":float(info["command_motor"]["trackingReward"]["desiredYawRateRadiansPerSecond"]),
                "paceAction":pace,"externalForce":max(float(np.max(np.abs(body.data.qfrc_applied))),float(np.max(np.abs(body.data.xfrc_applied))))})
            if body.goal is not identity or not np.array_equal(body.goal,initial_goal):
                raise RuntimeError("probe modified actual goal")
        env.assert_frozen_motor(check_checkpoint=True)
        return {"name":name,"sceneSeed":90301,"physicsSeed":91301,"mode":"turn","split":"train",
                "physicalControls":controls,"physicsSubsteps":substeps,"learningSteps":0,
                "summary":summarize_samples(samples,stop),"samples":samples,
                "terminalReason":info["navigation"]["terminal_reason"],"motorSha256":env.motor_checkpoint_sha256,
                "motorFrozen":True,"actualFinalGoalFixed":True,"expertMixture":mixture,
                "controlContract":env.control_contract}
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--motor-policy",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--expert-mixture",action="store_true")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise ValueError("preserve probe output; choose new path")
    specs=[("left-saturated",-1.,None,args.motor_policy,args.expert_mixture),("right-saturated",1.,None,args.motor_policy,args.expert_mixture),
           ("straight-stop",0.,4.,args.motor_policy,args.expert_mixture),("right-stop",1.,4.,args.motor_policy,args.expert_mixture)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        episodes=list(pool.map(run_probe,specs))
    result={"version":1,"scope":"train-only turn/stop feasibility; no parameter or physics changes",
            "samplesAtHz":10,"learningSteps":0,"episodes":episodes}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for episode in episodes:
        print(json.dumps({"name":episode["name"],"terminalReason":episode["terminalReason"],**episode["summary"]}),flush=True)


if __name__=="__main__":
    main()
