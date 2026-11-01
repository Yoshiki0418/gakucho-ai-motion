"""Four-room diagnostic of two frozen motor experts, with every actual gate.

The full-map geometry controller is explicitly a development counterfactual,
not learned visual inference or held-out training data.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from .command_motor import observed_command_angle
from .motor_expert_mixture import ExpertMixtureVisualNavigationEnv


class RecordedExpertMixtureEnv(ExpertMixtureVisualNavigationEnv):
    def __init__(self,**kwargs):
        self._native_mixture_controls=[]
        super().__init__(**kwargs)

    def _blend_predict(self,observation,*args,**kwargs):
        action,state=super()._blend_predict(observation,*args,**kwargs)
        angle,distance=observed_command_angle(observation)
        self._native_mixture_controls.append({
            "rightExpertWeightActuallyUsed":self._last_expert_weight,
            "actualConsumedCommandAngleRadians":angle,
            "actualConsumedCommandDistanceMeters":distance,
            "returnedJointResidualAction20":action.tolist(),
            "sourceExpertSha256":self.control_contract["expertMixture"]["leftStopExpertSha256"],
            "rightExpertSha256":self.motor_checkpoint_sha256,
            "extraTrainingSteps":0,"extraOptimizerUpdates":0,
        })
        return action,state

    def step(self,action):
        self._native_mixture_controls=[]
        observation,reward,done,truncated,info=super().step(action)
        count=info["command_control_steps"]
        if len(self._native_mixture_controls)!=count:
            raise RuntimeError("mixture gate record missed actual native20ms actor calls")
        if self.record_trajectory:
            if len(self.last_control_frames)!=count:
                raise RuntimeError("mixture gate record missed physical native20ms snapshots")
            for frame,control in zip(self.last_control_frames,self._native_mixture_controls):
                measured=frame["stats"]["command_motor"]["trackingReward"]["consumedCommandRelativeHeadingRadians"]
                if abs(measured-control["actualConsumedCommandAngleRadians"])>1e-5:
                    raise RuntimeError("mixture gate did not consume the physically measured short command")
                frame["stats"]["expert_mixture"]=copy.deepcopy(control)
        info["expert_mixture"]=copy.deepcopy(self._native_mixture_controls[-1])
        return observation,reward,done,truncated,info


def run_scene(spec):
    from .contact_attribution import rollout_controller
    from .training import dump_json
    seed,motor,avatar,output,baseline=spec
    output,baseline=Path(output),Path(baseline)
    names=("motor_expert_mixture.py","motor_mixture_evaluation.py","tracked_visual_navigation.py",
           "motor_tracking_training.py","motor_cycle_tracking_training.py","contact_attribution.py")
    source={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}
    result=rollout_controller("geometry",scene_seed=seed,physics_seed=seed+2_000_000,
        motor_checkpoint=motor,avatar_profile=avatar,output=output,
        env_factory=RecordedExpertMixtureEnv)
    old=json.loads((baseline/f"geometry-{seed}"/"episode.json").read_text())
    if result["initialPhysicalState"]!=old["initialPhysicalState"]:
        raise RuntimeError("mixture comparison changed scene or full initialqpos/qvel")
    trace=json.loads((output/"trace.json").read_text())
    trace["controller"]="full-map development geometry oracle with two frozen source5aa/V5intermediate joint experts"
    trace["navigation"].update(expertMixture=True,expertMixtureIsNewlyTrainedSinglePolicy=False,
        heldoutDataUsedForTraining=False,source5aaSha256="5aa9536b35defac5d715f98c75f558f6f5cdaf9a26344672d02e7e4d4358252b",
        learnedRightExpertSha256=hashlib.sha256(Path(motor).read_bytes()).hexdigest())
    # The independent quality record includes the controller label. Re-audit
    # after replacing the legacy helper's single-motor label so every saved
    # field agrees with the final explicit two-expert trace.
    from .navigation_evaluation import audit_navigation
    trace["navigationQuality"]=audit_navigation(trace)
    result["quality"]=copy.deepcopy(trace["navigationQuality"])
    controls=[frame["stats"]["expert_mixture"] for frame in trace["frames"][1:]]
    if len(controls)!=result["physicalControls"]:
        raise RuntimeError("saved expert gate count differs from actual physicalcontrols")
    weights=np.asarray([control["rightExpertWeightActuallyUsed"] for control in controls])
    gate={"actualNative20msGateCount":len(controls),"sourceOnlyControls":int(np.sum(weights==0.)),
        "rightOnlyControls":int(np.sum(weights==1.)),"blendedControls":int(np.sum((weights>0.)&(weights<1.))),
        "meanRightWeight":float(np.mean(weights)),"minimumWeight":float(weights.min()),"maximumWeight":float(weights.max()),
        "eachGateCheckedAgainstActualConsumedPhysicalCommand":True,
        "bothActorCriticAdamAndTrainingCountsFrozen":True,"extraTrainingSteps":0,"extraOptimizerUpdates":0}
    dump_json(output/"trace.json",trace)
    result.update(initialPhysicalStateExactlyPreserved=True,expertMixture=gate,
        source5aaSha256=trace["navigation"]["source5aaSha256"],
        rightExpertSha256=trace["navigation"]["learnedRightExpertSha256"],
        traceSha256=hashlib.sha256((output/"trace.json").read_bytes()).hexdigest(),sourceFilesSha256=source)
    dump_json(output/"episode.json",result)
    archive=output/"source";archive.mkdir(exist_ok=True)
    for name,expected in source.items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f"mixture diagnostic source changed: {name}")
        shutil.copy2(Path(__file__).parent/name,archive/name)
    return {"sceneSeed":seed,"before":old,"after":result,"gate":gate}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--motor-policy",required=True);parser.add_argument("--avatar-profile",required=True)
    parser.add_argument("--output",required=True);parser.add_argument("--baseline",required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():
        raise ValueError("preserve completed mixture diagnostic")
    specs=[(seed,args.motor_policy,args.avatar_profile,str(output/f"geometry-{seed}"),args.baseline)
        for seed in (1180100,1180102,1180104,1180106)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        episodes=list(pool.map(run_scene,specs))
    report={"version":1,"complete":True,"episodes":episodes,
        "beforeVerified":sum(x["before"]["quality"]["navigationVerified"] for x in episodes),
        "afterVerified":sum(x["after"]["quality"]["navigationVerified"] for x in episodes),
        "fullMapGeometryControllerIsDevelopmentDiagnosticOnly":True,"heldoutDataUsedForTraining":False,
        "reservedFinalCohortUsed":False,"extraTrainingSteps":0,"extraOptimizerUpdates":0,
        "adoptedIntoLiveAvatar":False,"phase2Complete":False}
    from .training import dump_json
    dump_json(output/"comparison.json",report)
    print(json.dumps({key:report[key] for key in ("complete","beforeVerified","afterVerified")}),flush=True)


if __name__=="__main__":
    main()
