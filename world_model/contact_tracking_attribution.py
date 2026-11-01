"""Matched geometry-oracle contact evaluation of an explicit tracking-v4 motor.

Only the checkpoint changes. Held-out rooms remain diagnostic only, the goal
and all physical state are fixed through actual joint-torque simulation, and
every native control frame passes the original independent navigation audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .contact_attribution import rollout_controller
from .training import dump_json


def evaluate_tracking_motor(*, motor_checkpoint, avatar_profile, output,
                            baseline, seeds=(1180100, 1180102, 1180104, 1180106)):
    from .tracked_visual_navigation import make_visual_env
    output, baseline = Path(output), Path(baseline)
    if output.exists():
        raise ValueError("preserve completed tracking diagnostic; choose a new output")
    # Validate the complete old cohort before starting costly physical episodes.
    old = {seed:json.loads((baseline/f"geometry-{seed}"/"episode.json").read_text()) for seed in seeds}
    sources = {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("contact_attribution.py", "contact_tracking_attribution.py", "tracked_visual_navigation.py", "motor_tracking_training.py")}
    episodes = []
    for seed in seeds:
        result = rollout_controller("geometry",scene_seed=seed,physics_seed=seed+2_000_000,
            motor_checkpoint=motor_checkpoint,avatar_profile=avatar_profile,output=output/f"geometry-{seed}",
            env_factory=make_visual_env)
        if result["initialPhysicalState"] != old[seed]["initialPhysicalState"]:
            raise RuntimeError("tracking motor comparison received different scene or actual qpos/qvel")
        episodes.append({"sceneSeed":seed,"initialPhysicalStateExactlyPreserved":True,
            "before":old[seed],"after":result})
        report = {"version":1,"complete":len(episodes)==len(seeds),"episodes":episodes,
            "beforeVerified":sum(e["before"]["quality"]["navigationVerified"] for e in episodes),
            "afterVerified":sum(e["after"]["quality"]["navigationVerified"] for e in episodes),
            "controller":"full actual scene geometry diagnostic oracle, not a visual actor",
            "marginMeters":.5,"carrotMeters":.55,"walkingAction":-.6,
            "heldoutDataUsedForTraining":False,"reservedFinalCohortUsed":False,
            "motorCheckpointSha256":hashlib.sha256(Path(motor_checkpoint).read_bytes()).hexdigest(),
            "sourceSha256":sources,"explicitFactory":"world_model.tracked_visual_navigation.make_visual_env"}
        dump_json(output/"comparison.json",report)
    if any(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha for name,sha in sources.items()):
        raise RuntimeError("tracking diagnostic implementation changed during physical evaluation")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-policy",required=True)
    parser.add_argument("--avatar-profile",required=True)
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seeds",default="1180100,1180102,1180104,1180106")
    args=parser.parse_args()
    result=evaluate_tracking_motor(motor_checkpoint=args.motor_policy,avatar_profile=args.avatar_profile,
        baseline=args.baseline,output=args.output,seeds=tuple(map(int,args.seeds.split(","))))
    print(json.dumps({key:result[key] for key in ("complete","beforeVerified","afterVerified")}))


if __name__=="__main__":
    main()
