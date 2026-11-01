"""Explicit reward-v3 to reward-v4 low-motor transfer and GPU PPO training.

The only reward change narrows the measured yaw-rate tracking Gaussian. The
balanced train-only curriculum adds phase-random ordinary zero-distance stop
commands; no physics state, goal, actuator limit or source contract is edited.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .command_motor import COMMAND_MOTOR_REWARD_CONTRACT, command_tracking_reward
from .command_motor_curriculum import CommandMotorCurriculumEnv
from .command_motor_training import _dump, _sha256, _state_sha256


MOTOR_TRACKING_REWARD_CONTRACT = copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT)
MOTOR_TRACKING_REWARD_CONTRACT.update({
    "version":4,
    "yawRateReward":"4 * (exp(-((actual yaw rate - desired yaw rate)/.15)^2)-1); bounded [-4,0]",
    "explicitTransferFromVersion":3,
    "allOtherPhysicalRewardTermsUnchanged":True,
})


def tracking_reward_v4(measurements):
    result=command_tracking_reward(measurements)
    error=result["actualYawRateRadiansPerSecond"]-result["desiredYawRateRadiansPerSecond"]
    result["components"]["commandYawRate"]=4.*(math.exp(-(error/.15)**2)-1.)
    raw=float(sum(result["components"].values()))
    result.update(denseRawRewardBeforeClamp=raw,denseRawReward=max(-8.,raw),denseClampAdjustment=max(-8.,raw)-raw)
    return result


class MotorTrackingEnv(CommandMotorCurriculumEnv):
    """Unchanged physical motor, narrow measured-yaw reward, balanced commands."""

    def __init__(self, *, stop_probability=.5, **kwargs):
        if not 0.<=stop_probability<=1.:
            raise ValueError("stop probability must be within [0,1]")
        self.stop_probability=float(stop_probability)
        self._stop_start=None
        self._stop_end=None
        self._walking_lookahead=None
        self._stop_active=False
        kwargs.setdefault("command_lookahead_range",(.35,.5))
        kwargs.setdefault("mode","turn")
        kwargs.setdefault("split","train")
        kwargs.setdefault("max_episode_seconds",8.)
        super().__init__(**kwargs)
        # Keep the inherited physical motor's reset-time v3 contract intact.
        # The new training wrapper binds its distinct reward contract below;
        # replacing the parent's dictionary would defeat its reset audit.
        self.reward_control_contract=copy.deepcopy(self.control_contract)
        self.reward_control_contract["rewardContract"]=copy.deepcopy(MOTOR_TRACKING_REWARD_CONTRACT)
        self.training_command_generator_contract.update({
            "version":2,"rewardContractUnchanged":False,"rewardVersion":4,
            "stopEpisodeProbability":self.stop_probability,
            "stopStartTimeRangeSeconds":[3.,4.5],"stopDurationRangeSeconds":[1.,2.],
            "stopCommandDistanceMeters":0.,"stopCommandReplacedThroughOrdinaryCommandGenerator":True,
            "stopScheduleNotExposedToActor":True,
            "stopInputObservability":"actual consumed short-command features50:53 become zero horizontal displacement",
        })

    def _add_info(self,info):
        info=super()._add_info(info)
        if hasattr(self,"reward_control_contract"):
            info["command_motor"]["physicalMotorResetContract"]=copy.deepcopy(self.control_contract)
            info["command_motor"]["controlContract"]=copy.deepcopy(self.reward_control_contract)
        return info

    def _update_next_command(self, *, advance):
        if not advance:
            self._stop_start=self._stop_end=None
            self._stop_active=False
            super()._update_next_command(advance=False)
            self._walking_lookahead=self._lookahead
            if self.unwrapped.np_random.random()<self.stop_probability:
                self._stop_start=float(self.unwrapped.np_random.uniform(3.,4.5))
                self._stop_end=self._stop_start+float(self.unwrapped.np_random.uniform(1.,2.))
            return
        time_now=float(self.unwrapped.data.time)
        self._stop_active=self._stop_start is not None and self._stop_start<=time_now<self._stop_end
        self._lookahead=0. if self._stop_active else self._walking_lookahead
        super()._update_next_command(advance=True)
        self._command.update(trainingStopActive=self._stop_active,
            stopStartSeconds=self._stop_start,stopEndSeconds=self._stop_end)

    def step(self, action):
        observation,_,terminated,truncated,info=super().step(action)
        old=info["command_motor"]["trackingReward"]
        reward=tracking_reward_v4(old["measurements"])
        metadata=self._last_reward_info
        raw=reward["denseRawReward"]+metadata["successBonusRaw"]-metadata["terminalCostRawOutsideDenseClamp"]
        metadata.update(trackingReward=reward,physicalReward=raw,scaledMotorReward=raw*self.reward_scale,
                        rewardVersion=4,sourceRewardVersion3DenseRaw=old["denseRawReward"])
        info=self._add_info(info)
        if self.record_trajectory:
            self.last_control_frames=[self.snapshot()]
        return observation,raw*self.reward_scale,terminated,truncated,info


def assert_motor_tracking_policy(policy,env):
    from .body_contract import assert_policy_body
    from .command_motor import COMMAND_MOTOR_OBSERVATION_CONTRACT
    expected_control=copy.deepcopy(env.control_contract)
    expected_control["rewardContract"]=copy.deepcopy(MOTOR_TRACKING_REWARD_CONTRACT)
    if (getattr(policy,"command_motor_observation_contract",None)!=COMMAND_MOTOR_OBSERVATION_CONTRACT
            or getattr(policy,"command_motor_reward_contract",None)!=MOTOR_TRACKING_REWARD_CONTRACT
            or getattr(policy,"command_motor_control_contract",None)!=expected_control
            or getattr(policy,"motor_tracking_reward_contract",None)!=MOTOR_TRACKING_REWARD_CONTRACT
            or getattr(policy,"gamma",None)!=.99):
        raise ValueError("explicit reward-v4 motor/control/observation binding required")
    for key in ("observation_space","action_space"):
        actual,expected=getattr(policy,key),getattr(env,key)
        if (actual.shape!=expected.shape or actual.dtype!=expected.dtype
                or not np.array_equal(actual.low,expected.low) or not np.array_equal(actual.high,expected.high)):
            raise ValueError("reward-v4 motor spaces differ")
    assert_policy_body(policy,env.unwrapped)


assert_tracking_motor_policy=assert_motor_tracking_policy


def _make_env(index,seed):
    profile=Path(__file__).resolve().parents[1]/"frontend/public/world-model/gakucho-avatar-profile.json"
    env=MotorTrackingEnv(base_seed=seed+index,avatar_profile=profile,randomize=True,difficulty_range=(0.,.15))
    env.reset(seed=seed+1000+index)
    return env


def transfer(source_checkpoint,vector,*,device="cuda:0",seed=93301):
    """Validate exact v3 first, then preserve all tensors/Adam and rebind v4."""
    import torch
    from stable_baselines3 import PPO
    from .command_motor import CommandMotorEnv,assert_command_motor_policy
    profile=Path(__file__).resolve().parents[1]/"frontend/public/world-model/gakucho-avatar-profile.json"
    source=PPO.load(source_checkpoint,device="cpu")
    validator=CommandMotorEnv(avatar_profile=profile,command_lookahead_range=(.35,.5),mode="turn",split="train")
    target=_make_env(0,seed)
    try:
        assert_command_motor_policy(source,validator)
        initial_count=int(source.command_motor_physical_control_timesteps)
        if initial_count!=source.num_timesteps:
            raise ValueError("source actual control lineage mismatch")
        tensor_sha=_state_sha256(source.policy.state_dict())
        adam_sha=_state_sha256(source.policy.optimizer.state_dict())
        policy=PPO.load(source_checkpoint,env=vector,device=device,
            custom_objects={"n_steps":256,"batch_size":256,"n_epochs":5,
                            "learning_rate":.00005,"ent_coef":.0002})
        if _state_sha256(policy.policy.state_dict())!=tensor_sha or _state_sha256(policy.policy.optimizer.state_dict())!=adam_sha:
            raise RuntimeError("reward transfer changed actor/critic/std/Adam before learning")
        observations=[]
        for index in range(8):
            observation,_=validator.reset(seed=seed+100+index,options={"scene_seed":seed+index,"physics_seed":seed+100+index})
            observations.append(observation)
        observations=np.asarray(observations,dtype=np.float32)
        action_error=float(np.max(np.abs(source.predict(observations,deterministic=True)[0]-policy.predict(observations,deterministic=True)[0])))
        with torch.no_grad():
            values_before=source.policy.predict_values(torch.as_tensor(observations)).cpu().numpy()
            values_after=policy.policy.predict_values(torch.as_tensor(observations,device=policy.device)).cpu().numpy()
        value_error=float(np.max(np.abs(values_before-values_after)))
        if max(action_error,value_error)>1e-4:
            raise RuntimeError("CUDA transfer changed source outputs beyond floating-point tolerance")
        policy.command_motor_reward_contract=copy.deepcopy(MOTOR_TRACKING_REWARD_CONTRACT)
        policy.motor_tracking_reward_contract=copy.deepcopy(MOTOR_TRACKING_REWARD_CONTRACT)
        policy.command_motor_control_contract=copy.deepcopy(target.reward_control_contract)
        policy.motor_tracking_initial_physical_controls=initial_count
        policy.motor_tracking_training_command_contract=copy.deepcopy(target.training_command_generator_contract)
        audit={"sourceCheckpointSha256":_sha256(source_checkpoint),"sourceRewardVersion":3,"targetRewardVersion":4,
            "initialActualPhysicalControls":initial_count,"initialPpoEpochs":int(source._n_updates),
            "sourceAllPolicyTensorSha256":tensor_sha,"sourceAdamStateSha256":adam_sha,
            "allActorCriticLogStdTensorsPreservedExactly":True,"allAdamStatePreservedExactly":True,
            "criticReset":False,"freshAdam":False,"actualTrainInputsTested":8,
            "cudaActionMaxAbsError":action_error,"cudaCriticMaxAbsError":value_error,
            "sourceV3BodyObservationActionRewardAndControlContractsValidated":True,
            "trainingControlsAddedByTransfer":0,"onlyRewardTermChanged":"measured yaw Gaussian width .5 to .15",
            "trainingStopCommandsDeclaredSeparately":True}
        policy.motor_tracking_transfer_audit=copy.deepcopy(audit)
        assert_motor_tracking_policy(policy,target)
        return policy,audit
    finally:
        validator.close();target.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--steps",type=int,default=204800);parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=93301)
    args=parser.parse_args()
    if args.workers<1 or args.steps<1 or args.steps%(args.workers*256):
        raise ValueError("steps must be positive complete256×workers rollouts")
    if not 0<=args.seed<900000:
        raise ValueError("training seed must be train only")
    output=Path(args.output)
    if (output/"policy.zip").exists() or (output/"progress.json").exists():
        raise ValueError("preserve existing training run")
    output.mkdir(parents=True,exist_ok=True)
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from functools import partial
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("requested training requires actual CUDA")
    torch.cuda.reset_peak_memory_stats()
    source_hash=_sha256(args.source)
    from .command_motor_training import _source_hashes
    sources=_source_hashes()
    sources["motor_tracking_training.py"]=_sha256(Path(__file__))
    archive=output/"source";archive.mkdir(exist_ok=True)
    for name in sources:
        shutil.copy2(Path(__file__).parent/name,archive/name)
    _dump(output/"sources.json",sources)
    vector=SubprocVecEnv([partial(_make_env,index,args.seed) for index in range(args.workers)],start_method="spawn")
    started=time.monotonic()
    try:
        policy,audit=transfer(args.source,vector,seed=args.seed)
        _dump(output/"transfer-audit.json",audit)
        policy.save(output/"before-policy.zip")
        tensor_before=_state_sha256(policy.policy.state_dict())
        initial=int(policy.num_timesteps)
        class Counter(BaseCallback):
            def __init__(self):
                super().__init__();self.controls=0;self.episodes=0;self.signs={"negative":0,"positive":0,"zero":0};self.stops=0;self.last_save=0
            def _on_step(self):
                for info,done in zip(self.locals["infos"],self.locals["dones"]):
                    cm=info["command_motor"];nav=info["navigation"]
                    if cm["physicalControlsExecuted"]!=1 or cm["observedPhysicsSubstepsThisControl"]!=10:
                        raise RuntimeError("PPO lost actual20ms/2ms counts")
                    if nav["split"]!="train" or not 0<=nav["sceneSeed"]<1000000:
                        raise RuntimeError("held-out scene in low motor training")
                    if cm["rewardVersion"]!=4:
                        raise RuntimeError("PPO consumed old reward")
                    angle=cm["trackingReward"]["consumedCommandRelativeHeadingRadians"]
                    distance=cm["trackingReward"]["measurements"]["commandDistanceMeters"]
                    if abs(angle)>.450001 or distance>.500001:
                        raise RuntimeError("ordinary command bounds escaped")
                    self.signs["positive" if angle>1e-6 else "negative" if angle< -1e-6 else "zero"]+=1
                    self.stops+=int(distance<1e-6);self.controls+=1;self.episodes+=int(done)
                    self.model.command_motor_physical_control_timesteps+=1
                if self.model.num_timesteps!=initial+self.controls:
                    raise RuntimeError("actual low motor timestep counter mismatch")
                if self.controls-self.last_save>=4096:
                    self.last_save=self.controls
                    progress={"status":"training","actualPhysicalControlsAdded":self.controls,
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
            raise RuntimeError("PPO training did not complete requested actual controls")
        if _sha256(args.source)!=source_hash:
            raise RuntimeError("source checkpoint changed")
        for name,expected in sources.items():
            if _sha256(Path(__file__).parent/name)!=expected:
                raise RuntimeError(f"frozen source changed during training: {name}")
        policy.save(output/"policy.zip")
        tensor_after=_state_sha256(policy.policy.state_dict())
        if tensor_before==tensor_after:
            raise RuntimeError("actual CUDA PPO did not update policy tensors")
        result={"status":"complete","completedUtc":datetime.now(timezone.utc).isoformat(),
            "sourceCheckpointSha256":source_hash,"checkpointSha256":_sha256(output/"policy.zip"),
            "initialActualPhysicalControls":initial,"actualPhysicalControlsAdded":counter.controls,
            "actualPhysicsSubstepsAdded":10*counter.controls,"totalActualPhysicalControls":int(policy.num_timesteps),
            "ppoEpochsAdded":int(policy._n_updates)-audit["initialPpoEpochs"],
            "tensorSha256Before":tensor_before,"tensorSha256After":tensor_after,"cudaDevice":str(policy.device),
            "cudaDeviceName":torch.cuda.get_device_name(),"cudaPeakAllocatedBytes":torch.cuda.max_memory_allocated(),
            "adamStateCount":len(policy.policy.optimizer.state),"finalAdamStateSha256":_state_sha256(policy.policy.optimizer.state_dict()),
            "elapsedSeconds":time.monotonic()-started,"episodes":counter.episodes,
            "consumedAngleControlCounts":counter.signs,"zeroDistanceStopControls":counter.stops,
            "rewardContract":MOTOR_TRACKING_REWARD_CONTRACT,"phase2Complete":False,"adoptedIntoLiveAvatar":False}
        _dump(output/"training.json",result);_dump(output/"progress.json",result)
        print(json.dumps(result),flush=True)
    finally:
        vector.close()


if __name__=="__main__":
    main()
