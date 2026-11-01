"""Explicit hip-yaw action bounds and a no-learning navigator contract transfer.

Only the two motor residual actions named hip_yaw are bounded. This module
never writes body states, joint targets, torques or policy tensors in-place.
The separate transfer command preserves the complete 112-input navigator and
its counters while changing exactly its control-contract version and yaw
bound. A fresh Adam optimizer intentionally replaces the old optimizer.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import shutil

import numpy as np

from .humanoid import JOINT_NAMES

HIP_YAW_INDICES = (JOINT_NAMES.index("left_hip_yaw"), JOINT_NAMES.index("right_hip_yaw"))


def _validated_limit(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not 0 < float(value) <= 1):
        raise ValueError("hip-yaw action limit must be finite and within (0,1]")
    return float(value)


def bounded_motor_actions(action, max_hip_yaw_action):
    """Return a copy with only the two hip-yaw residual actions clipped.

Floating NumPy inputs retain their dtype and all other eighteen values retain
their exact bits. CPU/GPU tensor inputs are read through a detached copy and
the returned value is a NumPy array; no tensor or autograd state is modified.
"""
    limit = _validated_limit(max_hip_yaw_action)
    if type(action).__module__.startswith("torch") and hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    source = np.asarray(action)
    if source.shape != (20,) or source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError("motor action must contain twenty finite real values")
    # Integer/list callers need a floating array to represent a fractional
    # bound. Native floating policy outputs keep their original precision.
    result = source.copy() if source.dtype.kind == "f" else source.astype(np.float64)
    for index in HIP_YAW_INDICES:
        result[index] = np.clip(result[index], -limit, limit)
    return result


def _assert_transition(source, source_env, target_env):
    # Lazy imports let CommandNavigationEnv import the pure clipping function
    # without a module-level cycle back through that same environment.
    from .command_navigation import assert_command_policy
    from .body_contract import assert_policy_body

    assert_command_policy(source, source_env)
    old = source_env.control_contract
    new = target_env.control_contract
    if old.get("version") != 1 or "motorHipYawResidualActionLimit" in old:
        raise ValueError("source must use the explicit unlimited version-1 command contract")
    if new.get("version") != 2 or "motorHipYawResidualActionLimit" not in new:
        raise ValueError("target must use version 2 with an explicit motor hip-yaw action limit")
    _validated_limit(new["motorHipYawResidualActionLimit"])
    projected = copy.deepcopy(new)
    projected.pop("motorHipYawResidualActionLimit")
    projected["version"] = 1
    if projected != old:
        raise ValueError("contract transfer may change only version and motorHipYawResidualActionLimit")
    if (getattr(source_env, "motor_checkpoint_sha256", None) != old.get("motorCheckpointSha256")
            or getattr(target_env, "motor_checkpoint_sha256", None) != old.get("motorCheckpointSha256")):
        raise ValueError("both environments must expose the same actually loaded motor checkpoint")
    for kind in ("observation_space", "action_space"):
        first, second = getattr(source_env, kind), getattr(target_env, kind)
        if (first.__class__ is not second.__class__ or first.shape != second.shape
                or first.dtype != second.dtype or not np.array_equal(first.low, second.low)
                or not np.array_equal(first.high, second.high)):
            raise ValueError("contract transfer must preserve observation and action spaces exactly")
    assert_policy_body(source, target_env.unwrapped)
    count = getattr(source, "command_physical_control_timesteps", None)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("source must record its exact physical-control step count")
    if isinstance(source.num_timesteps, bool) or int(source.num_timesteps) != source.num_timesteps or source.num_timesteps < 0:
        raise ValueError("source must record a nonnegative decision-step count")


def audit_command_policy_copy(source, target, *, seed=74031, sample_count=32):
    """Compare real network tensors, deterministic actions and critic outputs."""
    import torch

    old, new = source.policy.state_dict(), target.policy.state_dict()
    if set(old) != set(new) or any(not torch.equal(value, new[name]) for name, value in old.items()):
        raise ValueError("contract transfer changed an actor, critic or standard-deviation tensor")
    random = np.random.default_rng(seed)
    observations = random.uniform(-1., 1., (sample_count, 112)).astype(np.float32)
    first = source.predict(observations, deterministic=True)[0]
    second = target.predict(observations, deterministic=True)[0]
    with torch.no_grad():
        source_values = source.policy.predict_values(torch.as_tensor(observations, device=source.device)).cpu().numpy()
        target_values = target.policy.predict_values(torch.as_tensor(observations, device=target.device)).cpu().numpy()
    errors = {"deterministicActionMaxAbsError": float(np.max(np.abs(first-second))),
              "criticValueMaxAbsError": float(np.max(np.abs(source_values-target_values)))}
    if any(value > 1e-6 for value in errors.values()):
        raise ValueError(f"contract transfer changed real network outputs: {errors}")
    return {"verified": True, "sampleCount": sample_count, "seed": seed,
        "allActorCriticAndLogStdTensorsExactlyEqual": True,
        "absoluteOutputTolerance": 1e-6, **errors}


def initialize_bounded_command_policy(source, env, *, source_env, body_env=None,
                                      seed=0, learning_rate=None, entropy=None,
                                      n_steps=None, batch_size=None, n_epochs=None):
    """Preserve a version-1 navigator in an explicitly bounded version-2 env.

    ``env`` can be a vector environment. ``body_env`` then supplies the real
    target simulator and its control contract. No PPO learning takes place.
    """
    import torch
    from stable_baselines3 import PPO
    from .command_navigation import assert_command_policy
    from .body_contract import bind_policy_body

    target_env = env if body_env is None else body_env
    _assert_transition(source, source_env, target_env)
    if (source.policy.__class__.__name__ != "ActorCriticPolicy" or source.use_sde
            or source.policy.features_extractor.__class__.__name__ != "FlattenExtractor"
            or source.policy_kwargs.get("net_arch") != {"pi": [128, 128], "vf": [128, 128]}
            or getattr(source.policy, "squash_output", False)):
        raise ValueError("contract transfer requires the standard unsquashed 128x128 MlpPolicy")
    target = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
        learning_rate=source.lr_schedule(1.) if learning_rate is None else learning_rate,
        n_steps=source.n_steps if n_steps is None else n_steps,
        batch_size=source.batch_size if batch_size is None else batch_size,
        n_epochs=source.n_epochs if n_epochs is None else n_epochs,
        gamma=source.gamma, gae_lambda=source.gae_lambda,
        ent_coef=source.ent_coef if entropy is None else entropy,
        clip_range=source.clip_range(1.), vf_coef=source.vf_coef,
        max_grad_norm=source.max_grad_norm, normalize_advantage=source.normalize_advantage,
        policy_kwargs=copy.deepcopy(source.policy_kwargs))
    target.policy.load_state_dict({name: value.detach().clone()
                                  for name, value in source.policy.state_dict().items()})
    if target.policy.optimizer.state:
        raise ValueError("transferred optimizer must start empty")
    source_ids = {id(parameter) for parameter in source.policy.parameters()}
    if source_ids & {id(parameter) for parameter in target.policy.parameters()}:
        raise ValueError("transferred navigator must not share policy tensors with its source")
    target.num_timesteps = int(source.num_timesteps)
    target.command_physical_control_timesteps = int(source.command_physical_control_timesteps)
    target.command_observation_contract = copy.deepcopy(source.command_observation_contract)
    target.command_control_contract = copy.deepcopy(target_env.control_contract)
    if hasattr(source, "command_training_mode"):
        target.command_training_mode = source.command_training_mode
    bind_policy_body(target, target_env.unwrapped)
    assert_command_policy(target, target_env)
    report = audit_command_policy_copy(source, target)
    report.update({"sourceControlContract": copy.deepcopy(source_env.control_contract),
        "targetControlContract": copy.deepcopy(target_env.control_contract),
        "changedFields": ["version", "motorHipYawResidualActionLimit"],
        "optimizerStateReused": False, "targetOptimizerStateInitiallyEmpty": True,
        "decisionTimestepsPreserved": target.num_timesteps,
        "physicalControlTimestepsPreserved": target.command_physical_control_timesteps,
        "additionalTrainingTimesteps": 0, "bodyTransferAllowed": False})
    target.command_control_transfer_audit = copy.deepcopy(report)
    return target, report


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--motor-policy", required=True)
    parser.add_argument("--motor-hip-yaw-action-limit", type=float, default=.15)
    parser.add_argument("--mode", choices=("turn", "obstacle"), default="turn")
    parser.add_argument("--seed", type=int, default=8741)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    return parser


def transfer_checkpoint(args):
    _validated_limit(args.motor_hip_yaw_action_limit)
    if not 0 <= args.seed < 1_000_000 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("transfer uses a training seed and a positive finite duration")
    import torch
    from stable_baselines3 import PPO
    from .avatar_profile import load_avatar_profile
    from .command_navigation import CommandNavigationEnv
    from .command_navigation_training import _checkpoint_path, _source_hashes
    from .command_navigation_pretraining import inherited_source_provenance
    from .navigation_training import _asset_hashes, _dump_json, _sha256

    torch.set_num_threads(1)
    source_path, motor_path = _checkpoint_path(args.source_policy), _checkpoint_path(args.motor_policy)
    source_sha, motor_sha = _sha256(source_path), _sha256(motor_path)
    profile = load_avatar_profile(args.avatar_profile)
    if profile is None:
        raise ValueError("contract transfer requires an explicitly measured avatar profile")
    output = Path(args.output)
    if (output/"policy.zip").exists():
        raise ValueError("output already contains a checkpoint; use a new output directory")
    if (output/"policy.zip").resolve() in (source_path.resolve(), motor_path.resolve()):
        raise ValueError("output must not overwrite the source navigator or fixed motor")
    hashes = _source_hashes()
    hashes[Path(__file__).name] = _sha256(__file__)
    assets = _asset_hashes(profile)
    kwargs = dict(motor_checkpoint=motor_path, mode=args.mode, split="train", base_seed=args.seed,
        difficulty_range=(0., .5), randomize=False, max_episode_seconds=args.seconds, avatar_profile=profile)
    source_env = CommandNavigationEnv(motor_hip_yaw_action_limit=None, **kwargs)
    target_env = None
    try:
        target_env = CommandNavigationEnv(motor_hip_yaw_action_limit=args.motor_hip_yaw_action_limit, **kwargs)
        source_env.reset(seed=args.seed)
        target_env.reset(seed=args.seed)
        source = PPO.load(source_path, device="cpu")
        source_provenance_path = source_path.parent/"provenance.json"
        source_provenance_sha = _sha256(source_provenance_path)
        source_provenance = json.loads(source_provenance_path.read_text())
        inherited_source_provenance(source_provenance, source_sha=source_sha, policy=source)
        target, audit = initialize_bounded_command_policy(source, target_env,
            source_env=source_env, seed=args.seed)
        if (_sha256(source_path) != source_sha or _sha256(motor_path) != motor_sha
                or _sha256(source_provenance_path) != source_provenance_sha
                or any(_sha256(Path(__file__).parent/name) != sha for name, sha in hashes.items())
                or _asset_hashes(profile) != assets):
            raise RuntimeError("source, motor, simulator or avatar assets changed during transfer")
        output.mkdir(parents=True, exist_ok=True)
        pending = output/"policy.pending.zip"
        target.save(pending)
        pending.replace(output/"policy.zip")
        provenance = {"version": 1, "algorithm": "PPO lineage; explicit control contract transfer",
            "task": "explicit_command_control_contract_transfer",
            "role": "unchanged learned navigator; only fixed motor hip-yaw residual limits change",
            "initialization": "explicitVersion1ToVersion2BoundedMotor", "mode": args.mode,
            "sourceCheckpoint": str(source_path), "sourceCheckpointSha256": source_sha,
            "sourceCheckpointProvenanceSha256": source_provenance_sha,
            "sourceCheckpointProvenance": source_provenance,
            "inheritedSupervisedPretraining": source_provenance.get("supervisedPretraining"),
            "additionalSupervisedUpdates": 0,
            "motorCheckpoint": str(motor_path), "motorCheckpointSha256": motor_sha,
            "motorFrozen": True, "motorObservationCount": 110, "motorActionCount": 20,
            "motorTrainingTimesteps": int(target_env._motor.num_timesteps),
            "checkpointSha256": _sha256(output/"policy.zip"),
            "timesteps": target.num_timesteps, "actualDecisionTimesteps": target.num_timesteps,
            "phaseTimesteps": 0, "phaseDecisionTimesteps": 0,
            "physicalControlTimesteps": target.command_physical_control_timesteps,
            "phasePhysicalControlTimesteps": 0, "phasePhysicsSubsteps": 0,
            "additionalTrainingTimesteps": 0,
            "supervisedRlDecisionTimestepsAdded": 0,
            "supervisedPhysicalControlTimestepsAdded": 0,
            "observationCount": 112, "actionCount": 2,
            "commandObservationContract": target.command_observation_contract,
            "commandControlContract": target.command_control_contract,
            "controlTransferAudit": audit, "sourceSha256": hashes, "assetSha256": assets,
            "avatarProfile": profile, "avatarProfileSha256": target.avatar_profile_sha256,
            "physicalAssumptions": target_env.unwrapped.physical_assumptions,
            "bodyModelSha256": target.body_model_sha256,
            "bodyTransferAllowed": False, "rootPositionOverwrittenDuringStep": False,
            "heldOutScenesUsedForTraining": False, "physicsStepsDuringTransfer": 0,
            "highLevelInferenceUsesTeacher": False, "fixedWaypointsSupplied": False,
            "proficiencyPassed": False, "anyRoomNavigationVerified": False,
            "complete": True, "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
        _dump_json(output/"provenance.json", provenance)
        _dump_json(output/"initialization.json", provenance)
        archive = output/"simulation-sources"
        archive.mkdir(exist_ok=True)
        for name, sha in hashes.items():
            destination = archive/name
            shutil.copyfile(Path(__file__).parent/name, destination)
            if _sha256(destination) != sha:
                raise RuntimeError("source archive differs from the audited transfer source")
        print(json.dumps({key: provenance[key] for key in ("checkpointSha256", "timesteps",
            "physicalControlTimesteps", "additionalTrainingTimesteps")}), flush=True)
        return provenance
    finally:
        if target_env is not None:
            target_env.close()
        source_env.close()


def main():
    parser = argument_parser()
    args = parser.parse_args()
    try:
        _validated_limit(args.motor_hip_yaw_action_limit)
    except ValueError as error:
        parser.error(str(error))
    transfer_checkpoint(args)


if __name__ == "__main__":
    main()
