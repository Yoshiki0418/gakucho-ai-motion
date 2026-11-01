"""Disclosed diagnostic blend of two frozen same-body joint policies.

The gate reads only the ordinary consumed local short command. It has no room
map, route or geometry teacher. Every returned joint residual remains bounded
and the inherited physical motor applies actual 20ms torques/2ms contacts.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from .command_motor import assert_command_motor_policy,observed_command_angle
from .command_motor_training import _sha256,_state_sha256
from .tracked_visual_navigation import TrackedVisualNavigationEnv,_training_metadata
from .visual_navigation import FROZEN_MOTOR_CHECKPOINT_SHA256


def right_expert_weight(observation,full_weight_heading_radians=.2):
    if not math.isfinite(full_weight_heading_radians) or not 0.<full_weight_heading_radians<=.45:
        raise ValueError("right expert blending heading must be in (0,.45]")
    angle,distance=observed_command_angle(observation)
    if distance<=1e-6:
        return 0.
    value=float(np.clip(angle/full_weight_heading_radians,0.,1.))
    return value*value*(3.-2.*value)


class ExpertMixtureVisualNavigationEnv(TrackedVisualNavigationEnv):
    """Keep source left/stop actions; smoothly use a learned right expert."""

    def __init__(self,*,motor_checkpoint,full_weight_heading_radians=.2,**kwargs):
        self._mixture_ready=False
        # Validate gate parameters without reading or changing physical state.
        if not math.isfinite(full_weight_heading_radians) or not 0.<full_weight_heading_radians<=.45:
            raise ValueError("right expert blending heading must be in (0,.45]")
        self.full_weight_heading_radians=float(full_weight_heading_radians)
        super().__init__(motor_checkpoint=motor_checkpoint,**kwargs)
        try:
            from stable_baselines3 import PPO
            self._source_motor=PPO.load(self.original_motor_checkpoint,device="cpu")
            assert_command_motor_policy(self._source_motor,self.motor_env)
            if _sha256(self.original_motor_checkpoint)!=FROZEN_MOTOR_CHECKPOINT_SHA256:
                raise ValueError("left/stop expert must be immutable validated source5aa")
            self._source_motor.policy.set_training_mode(False)
            for parameter in self._source_motor.policy.parameters():
                parameter.requires_grad_(False)
            self._source_tensor_sha=_state_sha256(self._source_motor.policy.state_dict())
            self._source_adam_sha=_state_sha256(self._source_motor.policy.optimizer.state_dict())
            self._source_metadata_sha=_state_sha256(_training_metadata(self._source_motor))
            counts=(self._source_motor.num_timesteps,self._source_motor._n_updates)
            if any(type(value) is not int or value<0 for value in counts):
                raise ValueError("frozen source expert requires exact integer native training counters")
            self._source_counts=counts
            self._right_predict=self._motor.predict
            # A disclosed inference observer/gate; neither actor tensor or
            # underlying physics method is replaced or altered.
            self._motor.predict=self._blend_predict
            self._last_expert_weight=None
            self._mixture_ready=True
            self.control_contract=copy.deepcopy(self.control_contract)
            self.control_contract.update({
                "version":4,"role":"diagnostic smooth two-frozen-expert joint controller",
                "expertMixture":{"leftStopExpertSha256":FROZEN_MOTOR_CHECKPOINT_SHA256,
                    "rightExpertSha256":self.motor_checkpoint_sha256,
                    "gateInput":"reconstructed consumed horizontal short-command angle from existing motor110; no world goal/map/path",
                    "gate":"smoothstep(clip(angle/fullRightHeading,0,1)); explicit zero-distance stop uses source expert",
                    "fullRightHeadingRadians":self.full_weight_heading_radians,
                    "jointOutput":"convex blend of independently clipped deterministic20jointresidual actions",
                    "bothExpertsAndAdamAndTrainingCountersFrozen":True,
                    "extraTrainingControls":0,"extraOptimizerUpdates":0,
                    "scope":"diagnostic skill reuse, not a newly trained single actor or adopted live controller"},
            })
            self._mixture_contract_sha=_state_sha256(self.control_contract)
            self.assert_frozen_motor(check_checkpoint=True)
        except Exception:
            self.close()
            raise

    def _blend_predict(self,observation,*args,**kwargs):
        if self._mixture_ready:
            self._assert_gate_contract()
        if kwargs.get("deterministic",False) is not True:
            raise ValueError("diagnostic expert mixture requires deterministic inference")
        weight=right_expert_weight(observation,self.full_weight_heading_radians)
        right,state=self._right_predict(observation,*args,**kwargs)
        left,_=self._source_motor.predict(observation,*args,**kwargs)
        self._last_expert_weight=weight
        if weight==0.:
            return left,state
        if weight==1.:
            return right,state
        return ((1.-weight)*left+weight*right).astype(np.float32),state

    def _assert_gate_contract(self):
        if (self.full_weight_heading_radians!=self.control_contract["expertMixture"]["fullRightHeadingRadians"]
                or _state_sha256(self.control_contract)!=self._mixture_contract_sha):
            raise RuntimeError("diagnostic mixture changed its frozen consumed-command gate contract")

    def assert_frozen_motor(self,*,check_checkpoint=False):
        super().assert_frozen_motor(check_checkpoint=check_checkpoint)
        if not self._mixture_ready:
            return
        self._assert_gate_contract()
        counts=(self._source_motor.num_timesteps,self._source_motor._n_updates)
        if (any(parameter.requires_grad for parameter in self._source_motor.policy.parameters())
                or _state_sha256(self._source_motor.policy.state_dict())!=self._source_tensor_sha
                or _state_sha256(self._source_motor.policy.optimizer.state_dict())!=self._source_adam_sha
                or _state_sha256(_training_metadata(self._source_motor))!=self._source_metadata_sha
                or any(type(value) is not int for value in counts)
                or counts!=self._source_counts):
            raise RuntimeError("diagnostic mixture changed frozen source actor/critic/Adam/counters")
        assert_command_motor_policy(self._source_motor,self.motor_env)
        if check_checkpoint and _sha256(self.original_motor_checkpoint)!=FROZEN_MOTOR_CHECKPOINT_SHA256:
            raise RuntimeError("diagnostic mixture changed original source checkpoint")
