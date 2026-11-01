"""Explicit visual-policy transfer to a separately trained tracking motor.

The original frozen-5aa visual environment and its strict contracts stay
unchanged. This adapter validates that baseline first, then validates the new
motor through its own training contract. Only the low inference weights change;
joint limits, physical stepping, sensing and high-level task rewards do not.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .command_motor_training import _sha256, _state_sha256
from .visual_navigation import (FROZEN_MOTOR_CHECKPOINT_SHA256,
                                VisualNavigationEnv, assert_visual_policy)


_ROOT = Path(__file__).resolve().parents[1]
_ORIGINAL_MOTOR = _ROOT/".runtime/world-model/stage2-gakucho-command-motor-local-curriculum-adapted/policy.zip"


def _tracking_validation(policy, motor_env):
    # Each training module owns its exact contract and authoritative validator.
    # An integer version alone never grants inference compatibility.
    from .motor_tracking_training import (MOTOR_TRACKING_REWARD_CONTRACT,
                                          assert_tracking_motor_policy)
    from .motor_cycle_tracking_training import (MOTOR_CYCLE_TRACKING_REWARD_CONTRACT,
                                                assert_cycle_tracking_motor_policy)
    reward=getattr(policy,"command_motor_reward_contract",None)
    if reward==MOTOR_TRACKING_REWARD_CONTRACT:
        version=4
        assert_tracking_motor_policy(policy,motor_env)
    elif reward==MOTOR_CYCLE_TRACKING_REWARD_CONTRACT:
        version=5
        assert_cycle_tracking_motor_policy(policy,motor_env)
    else:
        raise ValueError("tracking motor requires an exact registered v4 or v5 training contract")
    if getattr(policy,"motor_tracking_reward_contract",None)!=reward:
        raise ValueError("tracking motor reward provenance disagrees with its exact contract")
    audit = getattr(policy, "motor_tracking_transfer_audit", None)
    initial = getattr(policy, "motor_tracking_initial_physical_controls", None)
    if (not isinstance(audit,dict)
            or audit.get("sourceCheckpointSha256")!=FROZEN_MOTOR_CHECKPOINT_SHA256
            or audit.get("sourceRewardVersion")!=3 or audit.get("targetRewardVersion")!=version
            or audit.get("sourceV3BodyObservationActionRewardAndControlContractsValidated") is not True
            or audit.get("allActorCriticLogStdTensorsPreservedExactly") is not True
            or audit.get("allAdamStatePreservedExactly") is not True
            or type(initial) is not int or initial < 0
            or audit.get("initialActualPhysicalControls")!=initial
            or type(policy.num_timesteps) is not int or policy.num_timesteps < initial
            or getattr(policy,"command_motor_physical_control_timesteps",None)!=policy.num_timesteps):
        raise ValueError("tracking motor requires verified explicit source-5aa/v3 transfer provenance and native counts")
    return version


def _training_metadata(policy):
    """Freeze all new training provenance alongside the legacy native counts."""
    return {name:copy.deepcopy(value) for name,value in vars(policy).items()
        if name.startswith(("command_motor_", "motor_tracking_", "tracking_motor_", "motor_cycle_tracking_"))}


class TrackedVisualNavigationEnv(VisualNavigationEnv):
    """Same visual interface, with an explicitly validated/frozen tracking motor."""

    def __init__(self, *, motor_checkpoint, baseline_motor_checkpoint=_ORIGINAL_MOTOR, **kwargs):
        tracked = Path(motor_checkpoint)
        if not tracked.is_file():
            raise ValueError("the explicit tracking motor checkpoint must exist")
        tracked_sha = _sha256(tracked)
        if tracked_sha == FROZEN_MOTOR_CHECKPOINT_SHA256:
            raise ValueError("use the original visual environment for the unchanged 5aa motor")
        self._tracking_motor_installed = False
        super().__init__(motor_checkpoint=baseline_motor_checkpoint, **kwargs)
        self.original_visual_control_contract = copy.deepcopy(self.control_contract)
        self.original_motor_checkpoint = Path(baseline_motor_checkpoint)
        try:
            from stable_baselines3 import PPO
            candidate = PPO.load(tracked, device="cpu")
            version=_tracking_validation(candidate, self.motor_env)
            candidate.policy.set_training_mode(False)
            for parameter in candidate.policy.parameters():
                parameter.requires_grad_(False)
            self._motor = candidate
            self.motor_checkpoint = tracked
            self.motor_checkpoint_sha256 = tracked_sha
            self._motor_tensor_sha256 = _state_sha256(candidate.policy.state_dict())
            self._motor_optimizer_sha256 = _state_sha256(candidate.policy.optimizer.state_dict())
            self._motor_counts = self._motor_training_counts()
            self._tracking_metadata_sha256 = _state_sha256(_training_metadata(candidate))
            self._tracking_motor_installed = True
            self.control_contract = copy.deepcopy(self.original_visual_control_contract)
            self.control_contract.update({
                "version":3,
                "role":"depth navigation over explicitly transferred frozen tracking joint motor",
                "sourceMotorTrainingRewardVersion":version,
                "motorCheckpointSha256":tracked_sha,
                "originalMotorCheckpointSha256":FROZEN_MOTOR_CHECKPOINT_SHA256,
                "motorControlContract":copy.deepcopy(candidate.command_motor_control_contract),
                "sourceMotorTrainingRewardContract":copy.deepcopy(candidate.command_motor_reward_contract),
                "inferenceLowIntrinsicRewardContract":copy.deepcopy(self.motor_env.control_contract["rewardContract"]),
                "inferenceLowIntrinsicRewardUsedByHighController":False,
                "trackingMotorTransfer":"explicit same-body/input/control validation; training reward differs only",
                "physicalSteppingAndHighRewardUnchanged":True,
            })
            self.assert_frozen_motor(check_checkpoint=True)
        except Exception:
            self.close()
            raise

    def assert_frozen_motor(self, *, check_checkpoint=False):
        if not self._tracking_motor_installed:
            return super().assert_frozen_motor(check_checkpoint=check_checkpoint)
        if (self._motor_training_counts()!=self._motor_counts
                or any(parameter.requires_grad for parameter in self._motor.policy.parameters())
                or _state_sha256(self._motor.policy.state_dict())!=self._motor_tensor_sha256
                or _state_sha256(self._motor.policy.optimizer.state_dict())!=self._motor_optimizer_sha256
                or _state_sha256(_training_metadata(self._motor))!=self._tracking_metadata_sha256):
            raise RuntimeError("tracked visual navigation changed frozen motor weights, Adam, provenance or counts")
        _tracking_validation(self._motor, self.motor_env)
        if check_checkpoint and _sha256(self.motor_checkpoint)!=self.motor_checkpoint_sha256:
            raise RuntimeError("the frozen tracking motor checkpoint changed")


def make_visual_env(*, motor_checkpoint, **kwargs):
    """Dispatch through strict original or explicit tracking-v4/v5 validation."""
    if _sha256(motor_checkpoint)==FROZEN_MOTOR_CHECKPOINT_SHA256:
        return VisualNavigationEnv(motor_checkpoint=motor_checkpoint, **kwargs)
    return TrackedVisualNavigationEnv(motor_checkpoint=motor_checkpoint, **kwargs)


def transfer_visual_policy(source_checkpoint, target_env, *, output=None, actual_observation=None):
    """Preserve every learned tensor/Adam/native counter; update explicit motor metadata."""
    if not isinstance(target_env, TrackedVisualNavigationEnv):
        raise ValueError("explicit visual transfer requires a validated tracking environment")
    import torch
    from stable_baselines3 import PPO
    model = PPO.load(source_checkpoint, device="cpu")
    reference = SimpleNamespace(observation_space=target_env.observation_space,
        action_space=target_env.action_space, observation_contract=target_env.observation_contract,
        control_contract=target_env.original_visual_control_contract, unwrapped=target_env.unwrapped)
    assert_visual_policy(model, reference)
    target_env.assert_frozen_motor(check_checkpoint=True)
    counts = {key:copy.deepcopy(getattr(model,key,None)) for key in (
        "num_timesteps","_n_updates","visual_physical_control_timesteps","visual_physics_substeps")}
    if (any(type(value) is not int or value < 0 for value in counts.values())
            or counts["visual_physics_substeps"]!=counts["visual_physical_control_timesteps"]*10):
        raise ValueError("explicit visual transfer requires exact PPO and native physical counters")
    weights = _state_sha256(model.policy.state_dict())
    adam = _state_sha256(model.policy.optimizer.state_dict())
    observation = actual_observation
    if observation is None:
        if not target_env._has_reset:
            raise ValueError("supply actual observation or reset the target environment before visual transfer")
        observation = target_env._observation()
    observation = {key:value.copy() for key,value in observation.items()}
    before = model.predict(observation,deterministic=True)[0].copy()
    model.visual_control_contract = copy.deepcopy(target_env.control_contract)
    assert_visual_policy(model,target_env)
    after = model.predict(observation,deterministic=True)[0]
    np.testing.assert_array_equal(before,after)
    if (_state_sha256(model.policy.state_dict())!=weights
            or _state_sha256(model.policy.optimizer.state_dict())!=adam
            or any(getattr(model,key,None)!=value for key,value in counts.items())):
        raise RuntimeError("explicit motor metadata transfer changed visual policy, Adam or counts")
    audit = {"sourceCheckpointSha256":_sha256(source_checkpoint),
        "sourceMotorCheckpointSha256":FROZEN_MOTOR_CHECKPOINT_SHA256,
        "targetMotorCheckpointSha256":target_env.motor_checkpoint_sha256,
        "visualTensorSha256":weights,"visualAdamSha256":adam,
        "actorCriticLogStdAdamAndAllNativeCountsPreservedExactly":True,
        "actualObservationActionPreservedExactly":True,
        "actualObservationSha256":_state_sha256({key:torch.as_tensor(value) for key,value in observation.items()}),
        "actualAction":after.tolist(),
        "inferenceDevice":"cpu","trainingCounters":counts,"ppoDecisionsAdded":0,
        "lowMotorIntrinsicTrainingRewardUsedForHighReward":False}
    model.visual_motor_transfer_audit = copy.deepcopy(audit)
    if output is not None:
        output=Path(output)
        if output.exists() or output.with_suffix(".zip").exists():
            raise ValueError("preserve prior policy; choose a new transfer checkpoint")
        output.parent.mkdir(parents=True,exist_ok=True)
        model.save(output)
    return model,audit
