"""Explicitly transfer a same-body walking PPO to 110-input room navigation.

The existing 59 physical observations remain a prefix. The first actor and
critic layers gain 51 zero-initialized sensor columns; all other parameters
remain unchanged. Transfer initializes a new optimizer and is audited against
the original actor, critic and deterministic action before any learning.
Training scenes always use the train split. This command deliberately does
not evaluate held-out scenes or select a checkpoint using their results.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from .avatar_profile import load_avatar_profile
from .body_contract import assert_policy_body, bind_policy_body

BASE_OBSERVATION_COUNT = 59
NAVIGATION_OBSERVATION_COUNT = 110
ACTION_COUNT = 20
NAVIGATION_OBSERVATION_CONTRACT = {
    "version": 1,
    "baseObservationCount": BASE_OBSERVATION_COUNT,
    "sensorObservationCount": 51,
    "observationCount": NAVIGATION_OBSERVATION_COUNT,
    "actionCount": ACTION_COUNT,
    "baseObservation": "FootstepHumanoidEnv 59-feature prefix including T2M gait phase",
    "sensorObservation": "SceneSensorWrapper: knee/body rays32, local floor heights9, validity9, headroom1",
    "actions": "bounded residual targets for 20 torque-limited physical joints",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _assert_spaces(policy, env, observation_count):
    if not isinstance(policy.observation_space, gym.spaces.Box) or policy.observation_space.shape != (observation_count,):
        raise ValueError(f"source policy must have {observation_count} observations")
    if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape != (NAVIGATION_OBSERVATION_COUNT,):
        raise ValueError("navigation environment must have 110 observations")
    if (not isinstance(policy.action_space, gym.spaces.Box)
            or policy.action_space.shape != (ACTION_COUNT,)
            or env.action_space.shape != (ACTION_COUNT,)
            or not np.array_equal(policy.action_space.low, env.action_space.low)
            or not np.array_equal(policy.action_space.high, env.action_space.high)):
        raise ValueError("navigation policy must preserve all 20 action bounds")
    if (not np.array_equal(policy.observation_space.low, env.observation_space.low[:observation_count])
            or not np.array_equal(policy.observation_space.high, env.observation_space.high[:observation_count])):
        raise ValueError("navigation observation bounds differ from the checkpoint")


def assert_navigation_policy(policy, env):
    """Reject mismatched input semantics or body before predicting or resuming."""
    if policy is None:
        return
    _assert_spaces(policy, env, NAVIGATION_OBSERVATION_COUNT)
    if getattr(policy, "navigation_observation_contract", None) != NAVIGATION_OBSERVATION_CONTRACT:
        raise ValueError("navigation observation contract is missing or differs")
    assert_policy_body(policy, env.unwrapped)


def audit_transfer(source, target, *, seed=73019, sample_count=32):
    """Compare actual SB3 network outputs with arbitrary appended sensor values."""
    import torch

    random = np.random.default_rng(seed)
    base = random.uniform(-1., 1., size=(sample_count, BASE_OBSERVATION_COUNT)).astype(np.float32)
    sensors = random.uniform(-1., 1., size=(sample_count, 51)).astype(np.float32)
    extended = np.concatenate([base, sensors], axis=1)
    source.policy.set_training_mode(False)
    target.policy.set_training_mode(False)
    with torch.no_grad():
        old_tensor = torch.as_tensor(base, device=source.device)
        new_tensor = torch.as_tensor(extended, device=target.device)
        old_actor, old_critic = source.policy.mlp_extractor(source.policy.extract_features(old_tensor))
        new_actor, new_critic = target.policy.mlp_extractor(target.policy.extract_features(new_tensor))
        old_value = source.policy.predict_values(old_tensor)
        new_value = target.policy.predict_values(new_tensor)
    old_action = source.predict(base, deterministic=True)[0]
    new_action = target.predict(extended, deterministic=True)[0]
    errors = {
        "actorLatentMaxAbsError": float(torch.max(torch.abs(old_actor-new_actor)).item()),
        "criticLatentMaxAbsError": float(torch.max(torch.abs(old_critic-new_critic)).item()),
        "criticValueMaxAbsError": float(torch.max(torch.abs(old_value-new_value)).item()),
        "deterministicActionMaxAbsError": float(np.max(np.abs(old_action-new_action))),
    }
    tolerance = 1e-6
    if any(error > tolerance for error in errors.values()):
        raise ValueError(f"59-to-110 transfer changed the source outputs: {errors}")
    return {"verified": True, "sampleCount": sample_count, "seed": seed,
            "arbitrarySensorValues": True, "absoluteTolerance": tolerance,
            "toleranceReason": "CPU matrix multiplication can round differently with 110 versus 59 input columns",
            **errors}


def initialize_from59(source, env, *, seed=0, learning_rate=1e-4, entropy=.001,
                      n_steps=512, batch_size=256, n_epochs=5, body_env=None):
    """Create a 110-input same-body PPO without adapting its observation silently."""
    import torch
    from stable_baselines3 import PPO

    _assert_spaces(source, env, BASE_OBSERVATION_COUNT)
    physical_env = (env if body_env is None else body_env).unwrapped
    assert_policy_body(source, physical_env)
    if (source.policy.__class__.__name__ != "ActorCriticPolicy" or source.use_sde
            or source.policy.features_extractor.__class__.__name__ != "FlattenExtractor"):
        raise ValueError("transfer requires the standard MlpPolicy without generalized state-dependent noise")
    if source.policy_kwargs.get("net_arch") != {"pi": [128, 128], "vf": [128, 128]}:
        raise ValueError("source actor and critic must each use the 128x128 architecture")
    if getattr(source.policy, "squash_output", False):
        raise ValueError("source action squashing is incompatible with this residual contract")
    target = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=1,
                 learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                 n_epochs=n_epochs, gamma=source.gamma, gae_lambda=source.gae_lambda,
                 ent_coef=entropy, clip_range=source.clip_range(1.),
                 vf_coef=source.vf_coef, max_grad_norm=source.max_grad_norm,
                 normalize_advantage=source.normalize_advantage,
                 policy_kwargs=copy.deepcopy(source.policy_kwargs))
    old = source.policy.state_dict()
    new = target.policy.state_dict()
    if set(old) != set(new):
        raise ValueError("source and target policy parameter names differ")
    expanded = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}
    with torch.no_grad():
        for name, value in old.items():
            if name in expanded:
                if value.shape != (128, BASE_OBSERVATION_COUNT) or new[name].shape != (128, NAVIGATION_OBSERVATION_COUNT):
                    raise ValueError(f"unexpected first layer shape: {name}")
                new[name].zero_()
                new[name][:, :BASE_OBSERVATION_COUNT].copy_(value)
            else:
                if new[name].shape != value.shape:
                    raise ValueError(f"source and target parameter shapes differ: {name}")
                new[name].copy_(value)
    target.policy.load_state_dict(new)
    # PPO constructed its own optimizer with 110-input parameters. Do not load
    # Adam moments or parameter references from the differently shaped source.
    if target.policy.optimizer.state:
        raise ValueError("the navigation optimizer must start with empty state")
    bind_policy_body(target, physical_env)
    target.navigation_observation_contract = copy.deepcopy(NAVIGATION_OBSERVATION_CONTRACT)
    target.num_timesteps = int(source.num_timesteps)
    report = audit_transfer(source, target)
    report.update({"copiedInputColumns": BASE_OBSERVATION_COUNT,
                   "zeroInitializedSensorColumns": 51,
                   "allRemainingParametersCopied": True, "logStdCopied": True,
                   "optimizerStateReused": False,
                   "sourceAvatarProfileSha256": getattr(source, "avatar_profile_sha256", None),
                   "sourceBodyModelSha256": getattr(source, "body_model_sha256", None)})
    target.navigation_transfer_audit = copy.deepcopy(report)
    return target, report


def resume_from110(checkpoint, vector, *, body_env, seed, learning_rate=1e-4, entropy=.001):
    """Restore weights/Adam while explicitly reseeding the new worker cohort."""
    from stable_baselines3 import PPO

    # SB3 otherwise restores the source seed in _setup_model and overrides the
    # new vector's pending reset seeds. Passing an explicit load override applies
    # the requested seed before that setup seeds each worker with seed + index.
    policy = PPO.load(checkpoint, env=vector, device="cpu", seed=seed)
    assert_navigation_policy(policy, body_env)
    policy.learning_rate = learning_rate
    policy.lr_schedule = lambda progress_remaining: learning_rate
    policy.ent_coef = entropy
    return policy


def _validated_args(args):
    if isinstance(args.steps, bool) or args.steps < 1:
        raise ValueError("steps must be positive")
    if not 1 <= args.envs <= 32 or args.checkpoint_every < 1:
        raise ValueError("envs must be 1..32 and checkpoint-every must be positive")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not (0 <= args.difficulty_min <= args.difficulty_max <= 1):
        raise ValueError("difficulty limits must be ordered within 0..1")
    if not 0 <= args.seed < 1_000_000 or args.seed + args.envs > 1_000_000:
        raise ValueError("training worker seeds must lie below the held-out split boundary 1000000")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.entropy) or args.entropy < 0:
        raise ValueError("entropy must be finite and nonnegative")
    if args.reset_log_std is not None and not math.isfinite(args.reset_log_std):
        raise ValueError("reset-log-std must be finite")
    if bool(args.initialize_from59) == bool(args.resume110):
        raise ValueError("supply exactly one of --initialize-from59 or --resume110")
    if args.mode not in ("turn", "obstacle"):
        raise ValueError("mode must be turn or obstacle")
    return args


def _source_hashes():
    names = ["navigation_training.py", "navigation.py", "navigation_scenes.py", "scenes.py", "sensors.py", "generalization.py",
             "environment.py", "humanoid.py", "avatar_profile.py", "body_contract.py",
             "locomotion.py", "steering.py", "imitation.py", "references.py", "contact_metrics.py"]
    directory = Path(__file__).parent
    return {name: _sha256(directory/name) for name in names if (directory/name).is_file()}


def _asset_hashes(avatar_profile):
    root = Path(__file__).resolve().parents[1]
    provenance = avatar_profile.get("provenance", {})
    model_path = root/provenance.get("modelPath", "")
    if not model_path.is_file() or not model_path.resolve().is_relative_to(root):
        raise ValueError("avatar profile must identify a local measured model asset")
    model_sha = _sha256(model_path)
    if model_sha != provenance.get("modelSha256"):
        raise ValueError("avatar model asset SHA differs from the measured profile")
    motions = root/"frontend/public/motions/room-prototype"
    paths = [model_path, motions/"manifest.json", *sorted(motions.glob("*.bvh"))]
    return {str(path.relative_to(root)): _sha256(path) for path in paths}


def train(args):
    _validated_args(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    import mujoco
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .navigation import EpisodeNavigationEnv

    torch.set_num_threads(1)
    avatar_profile = load_avatar_profile(args.avatar_profile)
    if avatar_profile is None:
        raise ValueError("phase2 requires an explicit measured --avatar-profile")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.initialize_from59 or args.resume110)
    if source_path.suffix != ".zip":
        source_path = source_path.with_suffix(".zip")
    if source_path.resolve() == (output/"policy.zip").resolve():
        raise ValueError("training output must not overwrite its source checkpoint")
    source_sha = _sha256(source_path)
    hashes = _source_hashes()
    asset_hashes = _asset_hashes(avatar_profile)

    def factory(index):
        def make():
            env = EpisodeNavigationEnv(mode=args.mode, split="train", base_seed=args.seed+index,
                                       difficulty_range=(args.difficulty_min, args.difficulty_max),
                                       randomize=True, max_episode_seconds=args.seconds,
                                       avatar_profile=avatar_profile)
            env.reset(seed=args.seed+index)
            return env
        return make

    contract_env = factory(0)()
    vector = None
    try:
        vector_class = SubprocVecEnv if args.envs > 1 else DummyVecEnv
        kwargs = {"start_method": "fork"} if args.envs > 1 else {}
        vector = vector_class([factory(index) for index in range(args.envs)], **kwargs)
        source = PPO.load(source_path, device="cpu")
        transfer_report = None
        if args.initialize_from59:
            policy, transfer_report = initialize_from59(source, vector, seed=args.seed,
                                                        learning_rate=args.learning_rate, entropy=args.entropy,
                                                        body_env=contract_env)
        else:
            assert_navigation_policy(source, contract_env)
            policy = resume_from110(source_path, vector, body_env=contract_env, seed=args.seed,
                                    learning_rate=args.learning_rate, entropy=args.entropy)
        # Vector environments do not expose unwrapped body metadata. Check and
        # bind against the real independently constructed navigation simulator.
        assert_policy_body(policy, contract_env.unwrapped)
        bind_policy_body(policy, contract_env.unwrapped)
        policy.navigation_observation_contract = copy.deepcopy(NAVIGATION_OBSERVATION_CONTRACT)
        policy.navigation_training_mode = args.mode
        if args.reset_log_std is not None:
            with torch.no_grad():
                policy.policy.log_std.fill_(args.reset_log_std)
        initial_timesteps = policy.num_timesteps
        started = time.monotonic()
        common = {"algorithm": "PPO", "task": "phase2_navigation", "mode": args.mode,
                  "role": "learned 20-joint residual over physical footstep prior",
                  "motionReference": "cached T2M BVH walking cadence and stride",
                  "steeringPrior": "bounded hip yaw joint targets; no root actuator or fixed route",
                  "sceneSplit": "train", "heldOutScenesUsedForTraining": False,
                  "requestedSteps": args.steps, "initialTimesteps": initial_timesteps,
                  "sourceCheckpoint": str(source_path), "sourceCheckpointSha256": source_sha,
                  "initialization": "explicit59to110" if args.initialize_from59 else "resume110",
                  "observationTransfer": transfer_report,
                  "navigationObservationContract": NAVIGATION_OBSERVATION_CONTRACT,
                  "avatarProfile": avatar_profile,
                  "avatarProfileSha256": policy.avatar_profile_sha256,
                  "bodyModelSha256": policy.body_model_sha256,
                  "physicalAssumptions": contract_env.unwrapped.physical_assumptions,
                  "sourceSha256": hashes,
                  "assetSha256": asset_hashes,
                  "seed": args.seed, "envCount": args.envs,
                  "maxEpisodeSeconds": args.seconds,
                  "trainingDifficultyRange": [args.difficulty_min, args.difficulty_max],
                  "learningRate": args.learning_rate, "entropy": args.entropy,
                  "logStdOverrideAfterTransferAudit": args.reset_log_std,
                  "policyDevice": "cpu", "physicsDevice": "cpu", "torchThreads": 1,
                  "rootPositionOverwrittenDuringStep": False,
                  "bodyTransferAllowed": False, "isT2M": False,
                  "neuralWorldModelTrained": False,
                  "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__,
                               "gymnasium": gym.__version__, "stable_baselines3": stable_baselines3.__version__}}
        _dump_json(output/"initialization.json", common)
        requested_remaining = args.steps
        while requested_remaining > 0:
            amount = min(requested_remaining, args.checkpoint_every)
            policy.learn(total_timesteps=amount, reset_num_timesteps=False)
            temporary_checkpoint = output/"policy.pending.zip"
            policy.save(temporary_checkpoint)
            temporary_checkpoint.replace(output/"policy.zip")
            requested_remaining -= amount
            if (_sha256(source_path) != source_sha or _source_hashes() != hashes
                    or _asset_hashes(avatar_profile) != asset_hashes):
                raise RuntimeError("source checkpoint or simulator code changed during this training run")
            progress = {**common, "timesteps": policy.num_timesteps,
                        "phaseTimesteps": policy.num_timesteps-initial_timesteps,
                        "actualTimesteps": policy.num_timesteps,
                        "wallSeconds": time.monotonic()-started,
                        "checkpointSha256": _sha256(output/"policy.zip"),
                        "complete": requested_remaining <= 0,
                        "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
            _dump_json(output/"training-progress.json", progress)
            _dump_json(output/"provenance.json", progress)
        print(json.dumps({key: progress[key] for key in
                          ("mode", "timesteps", "phaseTimesteps", "wallSeconds", "checkpointSha256")}), flush=True)
        return progress
    finally:
        if vector is not None:
            vector.close()
        contract_env.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["turn", "obstacle"], required=True)
    parser.add_argument("--steps", type=int, default=409600)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=8401)
    parser.add_argument("--output", required=True)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=.001)
    parser.add_argument("--reset-log-std", type=float)
    parser.add_argument("--avatar-profile", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize-from59", help="Explicitly widen a same-body 59-input PPO, verifying preserved outputs")
    source.add_argument("--resume110", help="Resume a checkpoint with the recorded navigation/body contract")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    try:
        _validated_args(args)
    except ValueError as error:
        parser.error(str(error))
    train(args)


if __name__ == "__main__":
    main()
