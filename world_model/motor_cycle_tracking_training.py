"""Prepared reward-v5 experiment: measured whole-gait yaw tracking.

This module does not start training. It preserves the original v3 instantaneous
yaw term and adds one bounded cycle-mean penalty. History is reward-only and
never modifies physical state or supplies extra policy observations.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .command_motor import COMMAND_MOTOR_REWARD_CONTRACT,command_tracking_reward
from .motor_tracking_training import MotorTrackingEnv


MOTOR_CYCLE_TRACKING_REWARD_CONTRACT=copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT)
MOTOR_CYCLE_TRACKING_REWARD_CONTRACT.update({
    "version":5,"explicitTransferFromVersion":3,
    "instantaneousYawTrackingUnchangedFromV3":True,
    "cycleWindowSeconds":"actual body's immutable cached walking reference.duration",
    "cycleMeanYawTracking":"time-weighted sum of actual wrapped pre/post heading deltas divided by covered physical20ms duration",
    "cycleMeanTarget":"time-weighted consumed short-command desired yaw rate from v3 measurements; no unslewed request",
    "cycleYawPenalty":"4 * min(coveredSeconds/windowSeconds,1) * (exp(-((mean actual yaw - mean consumed target)/.15)^2)-1)",
    "cycleYawPenaltyBoundsRaw":[-4.,0.],
    "cycleHistory":"reset-empty reward-only deque; fractionally trim oldest physical interval to exact source gait duration",
    "actorObservationAndActionUnchanged":True,
    "denseFloorAfterAdditionalPenalty":-8.,
})


class CycleYawTracker:
    """Integrate measured yaw increments over an exact physical-time window."""

    def __init__(self,window_seconds):
        if not math.isfinite(window_seconds) or window_seconds<=0.:
            raise ValueError("cycle window must be finite and positive")
        self.window=float(window_seconds)
        self.intervals=deque()

    def update(self,dt,actual_delta,target_rate):
        if (not all(math.isfinite(x) for x in (dt,actual_delta,target_rate)) or dt<=0.
                or abs(actual_delta)>math.pi+1e-9):
            raise ValueError("cycle tracker requires finite physical duration, wrapped actual yaw and target")
        self.intervals.append((float(dt),float(actual_delta),float(target_rate)*dt))
        covered=sum(entry[0] for entry in self.intervals)
        while covered>self.window+1e-12:
            first=self.intervals.popleft()
            excess=covered-self.window
            if first[0]<=excess+1e-12:
                covered-=first[0]
            else:
                keep=(first[0]-excess)/first[0]
                self.intervals.appendleft(tuple(value*keep for value in first))
                covered=self.window
        actual=sum(entry[1] for entry in self.intervals)/covered
        target=sum(entry[2] for entry in self.intervals)/covered
        coverage=min(covered/self.window,1.)
        penalty=4.*coverage*(math.exp(-((actual-target)/.15)**2)-1.)
        return {"windowSeconds":self.window,"coveredSeconds":covered,"coverage":coverage,
                "actualMeanYawRateRadiansPerSecond":actual,"consumedTargetMeanYawRateRadiansPerSecond":target,
                "boundedPenaltyRaw":penalty,"physicalIntervalsRetained":len(self.intervals)}


def cycle_tracking_reward(measurements,cycle):
    reward=command_tracking_reward(measurements)
    reward["components"]["commandCycleMeanYawRate"]=float(cycle["boundedPenaltyRaw"])
    raw=float(sum(reward["components"].values()))
    reward.update(denseRawRewardBeforeClamp=raw,denseRawReward=max(-8.,raw),
                  denseClampAdjustment=max(-8.,raw)-raw,cycleYawTracking=copy.deepcopy(cycle))
    return reward


class MotorCycleTrackingEnv(MotorTrackingEnv):
    """Same curriculum and physical stepping, v3 plus measured cycle penalty."""

    def __init__(self,**kwargs):
        self._cycle_tracker=None
        super().__init__(**kwargs)
        self.reward_control_contract["rewardContract"]=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
        self.training_command_generator_contract["rewardVersion"]=5

    def _add_info(self,info):
        info=super()._add_info(info)
        info["command_motor"]["rewardContract"]=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
        return info

    def reset(self,**kwargs):
        self._cycle_tracker=None
        observation,info=super().reset(**kwargs)
        self._cycle_tracker=CycleYawTracker(float(self.unwrapped.reference.duration))
        return observation,info

    def step(self,action):
        observation,_,terminated,truncated,info=super().step(action)
        if self._cycle_tracker is None:
            raise RuntimeError("reset must initialize reward-only gait history")
        measurements=info["command_motor"]["trackingReward"]["measurements"]
        base=command_tracking_reward(measurements)
        delta=math.atan2(math.sin(measurements["rootHeadingRadians"]-measurements["rootPreHeadingRadians"]),
                         math.cos(measurements["rootHeadingRadians"]-measurements["rootPreHeadingRadians"]))
        cycle=self._cycle_tracker.update(measurements["controlDtSeconds"],delta,base["desiredYawRateRadiansPerSecond"])
        reward=cycle_tracking_reward(measurements,cycle)
        metadata=self._last_reward_info
        raw=reward["denseRawReward"]+metadata["successBonusRaw"]-metadata["terminalCostRawOutsideDenseClamp"]
        metadata.pop("sourceRewardVersion3DenseRaw",None)
        metadata.update(trackingReward=reward,physicalReward=raw,scaledMotorReward=raw*self.reward_scale,rewardVersion=5)
        info=self._add_info(info)
        if self.record_trajectory:
            self.last_control_frames=[self.snapshot()]
        return observation,raw*self.reward_scale,terminated,truncated,info


def assert_cycle_tracking_motor_policy(policy,env):
    from .body_contract import assert_policy_body
    from .command_motor import COMMAND_MOTOR_OBSERVATION_CONTRACT
    control=copy.deepcopy(env.control_contract)
    control["rewardContract"]=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
    if (getattr(policy,"command_motor_reward_contract",None)!=MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
            or getattr(policy,"motor_cycle_tracking_reward_contract",None)!=MOTOR_CYCLE_TRACKING_REWARD_CONTRACT
            or getattr(policy,"command_motor_control_contract",None)!=control
            or getattr(policy,"command_motor_observation_contract",None)!=COMMAND_MOTOR_OBSERVATION_CONTRACT
            or getattr(policy,"gamma",None)!=.99):
        raise ValueError("explicit reward-v5 same-body motor binding required")
    for name in ("observation_space","action_space"):
        actual,expected=getattr(policy,name),getattr(env,name)
        if (actual.shape!=expected.shape or actual.dtype!=expected.dtype
                or not np.array_equal(actual.low,expected.low) or not np.array_equal(actual.high,expected.high)):
            raise ValueError("reward-v5 motor spaces differ")
    assert_policy_body(policy,env.unwrapped)


def _make_cycle_env(index,seed):
    root=Path(__file__).resolve().parents[1]
    env=MotorCycleTrackingEnv(base_seed=seed+index,randomize=True,difficulty_range=(0.,.15),
        avatar_profile=root/"frontend/public/world-model/gakucho-avatar-profile.json")
    env.reset(seed=seed+1000+index)
    return env


def transfer_cycle_policy(source_checkpoint,vector,*,device="cuda:0",seed=95301):
    """Reuse the immutable audited tensor transfer, explicitly bind v5 before learning."""
    from .motor_tracking_training import transfer
    from .command_motor_training import _state_sha256
    policy,audit=transfer(source_checkpoint,vector,device=device,seed=seed)
    tensors=_state_sha256(policy.policy.state_dict())
    adam=_state_sha256(policy.policy.optimizer.state_dict())
    target=_make_cycle_env(0,seed)
    try:
        policy.command_motor_reward_contract=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
        policy.motor_tracking_reward_contract=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
        policy.motor_cycle_tracking_reward_contract=copy.deepcopy(MOTOR_CYCLE_TRACKING_REWARD_CONTRACT)
        policy.command_motor_control_contract=copy.deepcopy(target.reward_control_contract)
        policy.motor_tracking_training_command_contract=copy.deepcopy(target.training_command_generator_contract)
        audit.update(targetRewardVersion=5,
            onlyRewardTermChanged="retain all v3 terms and add measured one-gait-cycle mean yaw penalty",
            cycleWindowSeconds=float(target.unwrapped.reference.duration),
            auditedTensorTransferHelperVersion4ReusedWithoutLearning=True,
            intermediateVersion4BindingPhysicalControls=0,intermediateVersion4BindingOptimizerUpdates=0)
        policy.motor_tracking_transfer_audit=copy.deepcopy(audit)
        if (_state_sha256(policy.policy.state_dict())!=tensors
                or _state_sha256(policy.policy.optimizer.state_dict())!=adam):
            raise RuntimeError("explicit v5 binding changed source policy tensors or Adam")
        assert_cycle_tracking_motor_policy(policy,target)
        return policy,audit
    finally:
        target.close()


def main():
    parser=argparse.ArgumentParser(description="Explicit whole-cycle tracking experiment; never auto-adopts a checkpoint")
    parser.add_argument("--source",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=204800);parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=95301)
    args=parser.parse_args()
    if args.workers<1 or args.steps<1 or args.steps%(args.workers*256) or not 0<=args.seed<900000:
        raise ValueError("require complete256×workers train-only rollouts")
    output=Path(args.output)
    if (output/"progress.json").exists() or (output/"policy.zip").exists():
        raise ValueError("preserve existing training run")
    output.mkdir(parents=True,exist_ok=True)
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from functools import partial
    from .command_motor_training import _dump,_sha256,_source_hashes,_state_sha256
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("whole-cycle PPO requires actual CUDA")
    torch.cuda.reset_peak_memory_stats()
    source_sha=_sha256(args.source)
    hashes=_source_hashes()
    for name in ("motor_tracking_training.py","motor_cycle_tracking_training.py"):
        hashes[name]=_sha256(Path(__file__).parent/name)
    archive=output/"source";archive.mkdir(exist_ok=True)
    for name in hashes:
        shutil.copy2(Path(__file__).parent/name,archive/name)
    _dump(output/"sources.json",hashes)
    vector=SubprocVecEnv([partial(_make_cycle_env,index,args.seed) for index in range(args.workers)],start_method="spawn")
    started=time.monotonic()
    try:
        policy,audit=transfer_cycle_policy(args.source,vector,seed=args.seed)
        _dump(output/"transfer-audit.json",audit)
        policy.save(output/"before-policy.zip")
        initial=int(policy.num_timesteps)
        before=_state_sha256(policy.policy.state_dict())
        class Counter(BaseCallback):
            def __init__(self):
                super().__init__();self.controls=0;self.episodes=0;self.last_save=0
                self.signs={"negative":0,"positive":0,"zero":0};self.stops=0
            def _on_step(self):
                for info,done in zip(self.locals["infos"],self.locals["dones"]):
                    cm=info["command_motor"];nav=info["navigation"]
                    if cm["physicalControlsExecuted"]!=1 or cm["observedPhysicsSubstepsThisControl"]!=10:
                        raise RuntimeError("cycle PPO lost native physical20ms/2ms counts")
                    if nav["split"]!="train" or not 0<=nav["sceneSeed"]<1000000:
                        raise RuntimeError("held-out room leaked into cycle motor training")
                    if cm["rewardVersion"]!=5 or cm["rewardContract"]!=MOTOR_CYCLE_TRACKING_REWARD_CONTRACT:
                        raise RuntimeError("cycle PPO reward metadata differs from actual v5 reward")
                    tracking=cm["trackingReward"]
                    angle=tracking["consumedCommandRelativeHeadingRadians"]
                    distance=tracking["measurements"]["commandDistanceMeters"]
                    if abs(angle)>.450001 or not 0<=distance<=.500001:
                        raise RuntimeError("cycle motor escaped unchanged short-command bounds")
                    cycle=tracking["cycleYawTracking"]
                    if not -4.000001<=cycle["boundedPenaltyRaw"]<=0. or cycle["coveredSeconds"]>cycle["windowSeconds"]+1e-8:
                        raise RuntimeError("cycle reward integration or bounds invalid")
                    self.signs["positive" if angle>1e-6 else "negative" if angle< -1e-6 else "zero"]+=1
                    self.stops+=int(distance<1e-6);self.controls+=1;self.episodes+=int(done)
                    self.model.command_motor_physical_control_timesteps+=1
                if self.model.num_timesteps!=initial+self.controls:
                    raise RuntimeError("cycle motor actual native control counter mismatch")
                if self.controls-self.last_save>=4096:
                    self.last_save=self.controls
                    progress={"status":"training","rewardVersion":5,"actualPhysicalControlsAdded":self.controls,
                        "actualPhysicsSubstepsAdded":10*self.controls,"totalActualPhysicalControls":int(self.model.num_timesteps),
                        "ppoEpochs":int(self.model._n_updates),"elapsedSeconds":time.monotonic()-started,
                        "episodes":self.episodes,"consumedAngleControlCounts":self.signs,"zeroDistanceStopControls":self.stops,
                        "cudaDevice":str(self.model.device),"cudaPeakAllocatedBytes":torch.cuda.max_memory_allocated()}
                    _dump(output/"progress.json",progress);print(json.dumps(progress),flush=True)
                if self.controls%40960==0:
                    self.model.save(output/f"checkpoint-{self.controls}.zip")
                return True
        counter=Counter()
        policy.learn(total_timesteps=args.steps,reset_num_timesteps=False,callback=counter)
        if counter.controls!=args.steps or policy.num_timesteps!=initial+args.steps:
            raise RuntimeError("cycle PPO did not finish actual requested physical controls")
        if _sha256(args.source)!=source_sha:
            raise RuntimeError("immutable source checkpoint changed")
        for name,expected in hashes.items():
            if _sha256(Path(__file__).parent/name)!=expected:
                raise RuntimeError(f"frozen cycle training source changed: {name}")
        policy.save(output/"policy.zip")
        after=_state_sha256(policy.policy.state_dict())
        if before==after:
            raise RuntimeError("CUDA cycle PPO did not actually update tensors")
        result={"status":"complete","rewardVersion":5,"completedUtc":datetime.now(timezone.utc).isoformat(),
            "sourceCheckpointSha256":source_sha,"checkpointSha256":_sha256(output/"policy.zip"),
            "initialActualPhysicalControls":initial,"actualPhysicalControlsAdded":counter.controls,
            "actualPhysicsSubstepsAdded":10*counter.controls,"totalActualPhysicalControls":int(policy.num_timesteps),
            "ppoEpochsAdded":int(policy._n_updates)-audit["initialPpoEpochs"],
            "tensorSha256Before":before,"tensorSha256After":after,"cudaDevice":str(policy.device),
            "cudaDeviceName":torch.cuda.get_device_name(),"cudaPeakAllocatedBytes":torch.cuda.max_memory_allocated(),
            "adamStateCount":len(policy.policy.optimizer.state),"finalAdamStateSha256":_state_sha256(policy.policy.optimizer.state_dict()),
            "elapsedSeconds":time.monotonic()-started,"episodes":counter.episodes,
            "consumedAngleControlCounts":counter.signs,"zeroDistanceStopControls":counter.stops,
            "rewardContract":MOTOR_CYCLE_TRACKING_REWARD_CONTRACT,"phase2Complete":False,"adoptedIntoLiveAvatar":False}
        _dump(output/"training.json",result);_dump(output/"progress.json",result)
        print(json.dumps(result),flush=True)
    finally:
        vector.close()


if __name__=="__main__":
    main()
