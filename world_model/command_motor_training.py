"""Explicitly adapt the joint motor to short-command gait with isolated PPO.

Nothing here adopts a checkpoint into the high-level controller. The source
110/20 final-goal motor is copied only after same-body/space verification;
the new command semantics, scaled value head and fresh Adam are recorded.
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
import shutil
import time

import numpy as np


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _source_hashes():
    from .command_navigation_training import _source_hashes as existing_sources
    hashes = existing_sources()
    directory = Path(__file__).parent
    hashes.update({name: _sha256(directory/name) for name in (
        "command_motor.py", "command_motor_training.py", "command_motor_curriculum.py")})
    return hashes


def _checkpoint_path(value):
    path = Path(value)
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _state_sha256(value):
    """Hash nested optimizer/tensor state independently of pickle storage IDs."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(f"tensor:{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}[".encode())
            for part in item:
                visit(part)
            digest.update(b"]")
        elif isinstance(item, (str, bool, int, float, type(None))):
            digest.update(f"{type(item).__name__}:{item!r};".encode())
        else:
            raise TypeError(f"unsupported state audit type {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def resume_command_motor(source_checkpoint, vector, *, body_env, seed):
    """Resume identical R3 actor/critic/std/Adam and audit actual train inputs.

    The sampled curriculum changes training-command coverage only. It is not
    a reward/observation transfer, and must not reset the critic or optimizer.
    """
    import torch
    from stable_baselines3 import PPO
    from .command_motor import assert_command_motor_policy

    if not 0 <= seed or seed+532 >= 1_000_000:
        raise ValueError("resume audit must use train-only seeds")
    checkpoint = _checkpoint_path(source_checkpoint)
    source = PPO.load(checkpoint, device="cpu")
    assert_command_motor_policy(source, body_env)
    actual = getattr(source, "command_motor_physical_control_timesteps", None)
    initial_v3 = getattr(source, "command_motor_reward_v3_initial_physical_controls", None)
    lineage = getattr(source, "command_motor_transfer_audit", None)
    ancestor = lineage.get("sourcePhysicalMotorTimesteps") if isinstance(lineage, dict) else None
    updates = getattr(source, "_n_updates", None)
    if (any(isinstance(value, bool) or not isinstance(value, int)
            for value in (actual, source.num_timesteps, initial_v3, ancestor, updates))
            or actual != source.num_timesteps
            or not 0 <= ancestor <= initial_v3 <= actual or updates < 0):
        raise ValueError("resume requires exact actual control and PPO-update lineage")
    policy = PPO.load(checkpoint, env=vector, device="cpu", seed=seed)
    assert_command_motor_policy(policy, body_env)
    source_tensor_sha = _state_sha256(source.policy.state_dict())
    source_optimizer_sha = _state_sha256(source.policy.optimizer.state_dict())
    if (_state_sha256(policy.policy.state_dict()) != source_tensor_sha
            or _state_sha256(policy.policy.optimizer.state_dict()) != source_optimizer_sha
            or policy.num_timesteps != actual or policy._n_updates != updates
            or policy.command_motor_physical_control_timesteps != actual):
        raise RuntimeError("resume changed joint actor/critic/std/Adam or actual counters")
    observations = []
    for index in range(32):
        observation, _ = body_env.reset(seed=seed+500+index, options={
            "scene_seed": seed+index, "physics_seed": seed+500+index})
        observations.append(observation)
    observations = np.asarray(observations, dtype=np.float32)
    source.policy.set_training_mode(False)
    policy.policy.set_training_mode(False)
    old_actions = source.predict(observations, deterministic=True)[0]
    new_actions = policy.predict(observations, deterministic=True)[0]
    action_error = float(np.max(np.abs(old_actions-new_actions)))
    with torch.no_grad():
        inputs = torch.as_tensor(observations, device="cpu")
        value_error = float((source.policy.predict_values(inputs)-policy.policy.predict_values(inputs)).abs().max())
    if action_error != 0. or value_error != 0.:
        raise RuntimeError("resume changed source outputs on actual train reset inputs")
    if (_state_sha256(policy.policy.state_dict()) != source_tensor_sha
            or _state_sha256(policy.policy.optimizer.state_dict()) != source_optimizer_sha):
        raise RuntimeError("read-only resume audit changed source parameter or Adam state")
    audit = {
        "sourceCheckpointSha256": _sha256(checkpoint),
        "sourceActualPhysicalControlTimesteps": actual,
        "sourcePpoUpdateEpochs": updates,
        "sourceAllPolicyTensorSha256": source_tensor_sha,
        "sourceAdamStateSha256": source_optimizer_sha,
        "allActorCriticLogStdTensorsPreservedExactly": True,
        "allAdamStatePreservedExactly": True,
        "criticReset": False, "freshAdam": False,
        "sourceControlRewardObservationAndBodyContractsVerifiedExactly": True,
        "sampleCount": 32, "samplesFromActualTrainResetObservations": True,
        "sceneSeedRange": [seed, seed+31],
        "physicsSeedRange": [seed+500, seed+531],
        "deterministicActionsMaxAbsError": action_error,
        "criticValuesMaxAbsError": value_error,
        "physicalControlsAddedByResumeAudit": 0,
        "ppoUpdateEpochsAddedByResumeAudit": 0,
        "trainingCommandDistributionChangeRecordedSeparately": True,
    }
    return policy, audit


def initialize_command_motor(source_checkpoint, vector, *, body_env, seed=90301,
                             learning_rate=.00005, entropy=.0002,
                             n_steps=512, batch_size=256, n_epochs=5,
                             initialize_from_command=False):
    """Copy the actor exactly; adapt only the declared final value head.

    Command reward v2 already has .01 value units. Its hidden critic is
    retained and only the final head is zeroed for the changed objective;
    applying another .01 factor would incorrectly change its units.
    """
    import torch
    from stable_baselines3 import PPO
    from .body_contract import bind_policy_body
    from .command_motor import (COMMAND_MOTOR_OBSERVATION_CONTRACT, COMMAND_MOTOR_REWARD_CONTRACT,
                                COMMAND_MOTOR_REWARD_V2_CONTRACT,
                                assert_command_motor_policy)
    from .navigation_training import assert_navigation_policy

    source = PPO.load(_checkpoint_path(source_checkpoint), device="cpu")
    if initialize_from_command:
        from .body_contract import assert_policy_body
        expected_v2_control = copy.deepcopy(body_env.control_contract)
        expected_v2_control["rewardContract"] = copy.deepcopy(COMMAND_MOTOR_REWARD_V2_CONTRACT)
        if (getattr(source, "command_motor_control_contract", None) != expected_v2_control
                or getattr(source, "command_motor_observation_contract", None) != COMMAND_MOTOR_OBSERVATION_CONTRACT
                or getattr(source, "command_motor_reward_contract", None) != COMMAND_MOTOR_REWARD_V2_CONTRACT
                or hasattr(source, "navigation_observation_contract")):
            raise ValueError("explicit command initialization requires reward-v2 with identical physical/input contracts; direct reward-v1 is unsupported")
        if (source.observation_space != body_env.observation_space
                or source.action_space != body_env.action_space):
            raise ValueError("explicit command initialization requires exact 110/20 spaces")
        assert_policy_body(source, body_env.unwrapped)
        lineage = getattr(source, "command_motor_transfer_audit", None)
        actual_controls = getattr(source, "command_motor_physical_control_timesteps", None)
        ancestor_controls = lineage.get("sourcePhysicalMotorTimesteps") if isinstance(lineage, dict) else None
        v2_initial_controls = getattr(source, "command_motor_reward_v2_initial_physical_controls", None)
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in (actual_controls, source.num_timesteps, ancestor_controls, v2_initial_controls))
                or actual_controls != source.num_timesteps
                or not 0 <= ancestor_controls <= v2_initial_controls <= actual_controls):
            raise ValueError("source command motor lacks exact actual physical-control lineage")
    else:
        assert_navigation_policy(source, body_env.navigation_env)
    if source.gamma != .99:
        raise ValueError("source motor must use gamma .99 per physical control")
    if (source.policy.__class__.__name__ != "ActorCriticPolicy" or source.use_sde
            or source.policy.features_extractor.__class__.__name__ != "FlattenExtractor"
            or getattr(source.policy, "squash_output", False)):
        raise ValueError("command motor transfer requires unsquashed standard MlpPolicy without SDE")
    if seed < 0 or seed+532 >= 1_000_000:
        raise ValueError("initialization audit must use train-only seeds")
    policy = PPO(source.policy_class, vector, device="cpu", seed=seed,
        learning_rate=learning_rate, ent_coef=entropy, gamma=.99,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        policy_kwargs=copy.deepcopy(source.policy_kwargs))
    policy.policy.load_state_dict(copy.deepcopy(source.policy.state_dict()))
    for name, value in source.policy.state_dict().items():
        if not torch.equal(value, policy.policy.state_dict()[name]):
            raise RuntimeError(f"motor transfer changed source tensor {name}")
    # These are actual exposed command observations from 32 real train resets,
    # not saved synthetic input or held-out evaluation rooms.
    observations = []
    for index in range(32):
        observation, _ = body_env.reset(seed=seed+500+index, options={
            "scene_seed": seed+index, "physics_seed": seed+500+index})
        observations.append(observation)
    observations = np.asarray(observations, dtype=np.float32)
    source.policy.set_training_mode(False)
    policy.policy.set_training_mode(False)
    old_actions = source.predict(observations, deterministic=True)[0]
    new_actions = policy.predict(observations, deterministic=True)[0]
    action_error = float(np.max(np.abs(old_actions-new_actions)))
    if action_error != 0. or not torch.equal(source.policy.log_std, policy.policy.log_std):
        raise RuntimeError("motor actor or log-std changed before declared adaptation")
    with torch.no_grad():
        inputs = torch.as_tensor(observations, device="cpu")
        values = source.policy.predict_values(inputs).clone()
        if initialize_from_command:
            policy.policy.value_net.weight.zero_()
            policy.policy.value_net.bias.zero_()
            value_error = float(policy.policy.predict_values(inputs).abs().max().item())
        else:
            policy.policy.value_net.weight.mul_(.01)
            policy.policy.value_net.bias.mul_(.01)
            scaled_values = policy.policy.predict_values(inputs)
            value_error = float(torch.max(torch.abs(scaled_values-.01*values)).item())
    if value_error > 1e-5:
        raise RuntimeError("motor critic scaling did not preserve its value-head semantics")
    policy.policy.optimizer = policy.policy.optimizer_class(policy.policy.parameters(),
        lr=learning_rate, **policy.policy.optimizer_kwargs)
    policy.num_timesteps = int(source.num_timesteps)
    policy.command_motor_physical_control_timesteps = int(source.num_timesteps)
    policy.command_motor_observation_contract = copy.deepcopy(COMMAND_MOTOR_OBSERVATION_CONTRACT)
    policy.command_motor_control_contract = copy.deepcopy(body_env.control_contract)
    policy.command_motor_reward_contract = copy.deepcopy(COMMAND_MOTOR_REWARD_CONTRACT)
    policy.command_motor_reward_v3_initial_physical_controls = int(source.num_timesteps)
    if hasattr(source, "command_motor_reward_v2_initial_physical_controls"):
        policy.command_motor_reward_v2_initial_physical_controls = int(source.command_motor_reward_v2_initial_physical_controls)
    # A fresh PPO intentionally does not inherit navigation_observation_contract.
    bind_policy_body(policy, body_env.unwrapped)
    assert_command_motor_policy(policy, body_env)
    audit = {"sourceCheckpointSha256": _sha256(_checkpoint_path(source_checkpoint)),
        "sourcePhysicalMotorTimesteps": (int(source.command_motor_transfer_audit["sourcePhysicalMotorTimesteps"])
                                         if initialize_from_command else int(source.num_timesteps)),
        "initialSourceActualPhysicalControlTimesteps": int(source.num_timesteps),
        "sourceRewardContractVersion": 2 if initialize_from_command else None,
        "targetRewardContractVersion": 3, "explicitCommandRewardTransfer": initialize_from_command,
        "sameBodyVerified": True, "sampleCount": 32,
        "samplesFromActualTrainResetObservations": True,
        "sceneSeedRange": [seed, seed+31], "physicsSeedRange": [seed+500, seed+531],
        "deterministicActionsMaxAbsErrorBeforeAdaptation": action_error,
        "actorAllTensorsCopiedExactly": True, "logStdCopiedExactlyBeforeConfiguredReset": True,
        "criticAdaptation": ("zero final linear value head weight and bias for changed reward objective; hidden critic unchanged, no second .01 rescaling"
                             if initialize_from_command else
                             "multiply final linear value head weight and bias by .01; hidden critic unchanged"),
        "criticValueScale": 1. if initialize_from_command else .01,
        "criticScaledValueMaxAbsError": value_error,
        "freshAdam": not bool(policy.policy.optimizer.state),
        "legacyNavigationContractRemoved": not hasattr(policy, "navigation_observation_contract"),
        "trainingTimestepsAddedByTransfer": 0, "bodyTransferAllowed": False}
    if initialize_from_command:
        audit["sourceCommandMotorTransferAudit"] = copy.deepcopy(source.command_motor_transfer_audit)
    policy.command_motor_transfer_audit = copy.deepcopy(audit)
    return policy, audit


def make_counter():
    from stable_baselines3.common.callbacks import BaseCallback
    from .humanoid import CONTROL_SUBSTEPS

    class MotorCounter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.controls = 0
            self.worker_scenes = {}
            self.curriculum_coverage = {"requestedOffsetEpisodeCounts": {},
                "observedEpisodeCount": 0, "actualConsumedControlCount": 0,
                "actualConsumedNegativeControls": 0, "actualConsumedZeroControls": 0,
                "actualConsumedPositiveControls": 0,
                "actualConsumedSaturatedNegativeControls": 0,
                "actualConsumedSaturatedPositiveControls": 0,
                "consumedCommandDistanceSumMeters": 0.,
                "consumedCommandDistanceMinMeters": None, "consumedCommandDistanceMaxMeters": None,
                "requestedEpisodeTargetsAreNotActualConsumedControlCounts": True}
            self._seen_curriculum_episodes = set()

        def _on_step(self):
            for worker, info in enumerate(self.locals["infos"]):
                metadata = info.get("command_motor", {})
                count = metadata.get("physicalControlsExecuted")
                if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                    raise RuntimeError("motor PPO requires exactly one real physical control per decision")
                substeps = metadata.get("observedPhysicsSubstepsThisControl")
                if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps != CONTROL_SUBSTEPS:
                    raise RuntimeError("motor PPO requires invariant monitoring every actual 2ms substep")
                navigation = info["navigation"]
                scene_seed = navigation["sceneSeed"]
                if (navigation["split"] != "train" or isinstance(scene_seed, bool)
                        or not isinstance(scene_seed, (int, np.integer)) or not 0 <= scene_seed < 1_000_000):
                    raise RuntimeError("held-out scene leaked into command motor training")
                if navigation["collision_audit_timestep_seconds"] != .002:
                    raise RuntimeError("motor contact audit must preserve every 2ms substep")
                self.controls += 1
                self.model.command_motor_physical_control_timesteps += 1
                command = metadata.get("nextCommand", {})
                if "curriculumEpisodeIndex" in command:
                    coverage = self.curriculum_coverage
                    requested = float(command["requestedHeadingOffsetRadians"])
                    episode = (worker, int(command["curriculumEpisodeIndex"]))
                    if episode not in self._seen_curriculum_episodes:
                        self._seen_curriculum_episodes.add(episode)
                        label = f"{requested:+.2f}"
                        counts = coverage["requestedOffsetEpisodeCounts"]
                        counts[label] = counts.get(label, 0)+1
                        coverage["observedEpisodeCount"] += 1
                    tracking = metadata["trackingReward"]
                    angle = float(tracking["consumedCommandRelativeHeadingRadians"])
                    distance = float(tracking["measurements"]["commandDistanceMeters"])
                    if not (math.isfinite(angle) and math.isfinite(distance)
                            and abs(angle) <= .45+1e-6 and 0 <= distance <= .5+1e-6):
                        raise RuntimeError("curriculum actual consumed command escaped the unchanged motor contract")
                    coverage["actualConsumedControlCount"] += 1
                    sign = "Negative" if angle < -1e-6 else "Positive" if angle > 1e-6 else "Zero"
                    coverage[f"actualConsumed{sign}Controls"] += 1
                    if abs(angle) >= .45-1e-6 and distance > 1e-6:
                        coverage[f"actualConsumedSaturated{sign}Controls"] += 1
                    coverage["consumedCommandDistanceSumMeters"] += distance
                    for suffix, function in (("Min", min), ("Max", max)):
                        key = f"consumedCommandDistance{suffix}Meters"
                        previous = coverage[key]
                        coverage[key] = distance if previous is None else function(previous, distance)
                scenes = self.worker_scenes.setdefault(worker, {})
                scenes.setdefault(navigation["sceneSeed"], {"sceneSeed": navigation["sceneSeed"],
                    "physicsSeed": navigation["physics_seed"], "sceneSha256": navigation["scene_sha256"],
                    "firstObservedCumulativeTimesteps": int(self.model.num_timesteps), "split": "train"})
            return True

    return MotorCounter()


def _validated_args(args):
    if getattr(args, "initialize_only", False) and args.resume:
        raise ValueError("initialize-only requires an explicit source transfer, not resume")
    if args.steps < 1 or args.checkpoint_every < 1 or not 1 <= args.envs <= 16:
        raise ValueError("steps/checkpoint-every must be positive; envs must be 1..16")
    if not 0 <= args.seed < 999_000 or args.seed+args.envs+532 >= 1_000_000:
        raise ValueError("motor training and transfer audit seeds must remain in the train split")
    if not 0 <= args.difficulty_min <= args.difficulty_max <= 1:
        raise ValueError("difficulty bounds must lie within [0, 1]")
    if not .35 <= args.lookahead_min <= args.lookahead_max <= .5:
        raise ValueError("command lookahead must remain in [.35, .5]")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("episode seconds must be finite and positive")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.entropy) or args.entropy < 0):
        raise ValueError("learning rate must be positive and entropy nonnegative")
    if args.reset_log_std is not None and not math.isfinite(args.reset_log_std):
        raise ValueError("configured motor log-std must be finite")
    curriculum = getattr(args, "local_command_curriculum", False)
    if curriculum:
        if not args.resume or args.reset_log_std is not None:
            raise ValueError("local-command curriculum requires strict R3 resume and --keep-log-std")
        if not 6. <= args.seconds <= 8.:
            raise ValueError("local-command curriculum requires 6..8 second train episodes")
        if getattr(args, "evaluate_only", False):
            raise ValueError("train-only local-command curriculum cannot be used for held-out evaluation")
    if getattr(args, "resume_audit_only", False):
        if not args.resume or args.reset_log_std is not None:
            raise ValueError("resume-audit-only requires --resume and --keep-log-std")
        if getattr(args, "initialize_only", False) or getattr(args, "evaluate_only", False):
            raise ValueError("resume-audit-only cannot be combined with initialize/evaluate-only")
    return args


def train(args):
    _validated_args(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import gymnasium
    import mujoco
    import torch
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .avatar_profile import load_avatar_profile
    from .command_motor import CommandMotorEnv, assert_command_motor_policy
    from .navigation_training import _asset_hashes
    from .humanoid import CONTROL_SUBSTEPS

    torch.set_num_threads(1)
    profile = load_avatar_profile(args.avatar_profile)
    if profile is None:
        raise ValueError("command motor adaptation requires the measured avatar profile")
    initialize_from_command = getattr(args, "initialize_from_command", None)
    source_path = _checkpoint_path(args.resume or initialize_from_command or args.source_policy)
    output = Path(args.output)
    if (output/"policy.zip").exists() or source_path.resolve() == (output/"policy.zip").resolve():
        raise ValueError("isolated motor adaptation must not overwrite an existing/source checkpoint")
    source_sha, hashes, assets = _sha256(source_path), _source_hashes(), _asset_hashes(profile)
    output.mkdir(parents=True, exist_ok=True)

    curriculum = getattr(args, "local_command_curriculum", False)
    environment_class = CommandMotorEnv
    if curriculum:
        from .command_motor_curriculum import CommandMotorCurriculumEnv
        environment_class = CommandMotorCurriculumEnv

    def factory(index):
        def make():
            return environment_class(mode="turn", split="train", base_seed=args.seed+index,
                difficulty_range=(args.difficulty_min, args.difficulty_max), randomize=True,
                avatar_profile=profile, max_episode_seconds=args.seconds,
                command_lookahead_range=(args.lookahead_min, args.lookahead_max))
        return make

    contract_env = factory(0)()
    vector = None
    try:
        vector_class = SubprocVecEnv if args.envs > 1 else DummyVecEnv
        kwargs = {"start_method": "fork"} if args.envs > 1 else {}
        vector = vector_class([factory(index) for index in range(args.envs)], **kwargs)
        if args.resume:
            policy, resume_audit = resume_command_motor(source_path, vector,
                body_env=contract_env, seed=args.seed)
            policy.learning_rate = args.learning_rate
            policy.lr_schedule = lambda _: args.learning_rate
            policy.ent_coef = args.entropy
            transfer = copy.deepcopy(getattr(policy, "command_motor_transfer_audit", None))
        else:
            resume_audit = None
            policy, transfer = initialize_command_motor(source_path, vector, body_env=contract_env,
                seed=args.seed, learning_rate=args.learning_rate, entropy=args.entropy,
                initialize_from_command=bool(initialize_from_command))
        if args.reset_log_std is not None:
            with torch.no_grad():
                policy.policy.log_std.fill_(args.reset_log_std)
        initial = int(policy.num_timesteps)
        initial_controls = int(policy.command_motor_physical_control_timesteps)
        initial_updates = int(policy._n_updates)
        if curriculum:
            if not hasattr(policy, "command_motor_curriculum_initial_physical_controls"):
                policy.command_motor_curriculum_initial_physical_controls = initial_controls
            policy.command_motor_training_command_generator_contract = copy.deepcopy(
                contract_env.training_command_generator_contract)
        counter = make_counter()
        common = {"algorithm": "PPO joint residual command-motor adaptation", "mode": "turn",
            "role": "isolated command-conditioned joint motor; disclosed training command generator is not a learned navigator",
            "observationCount": 110, "actionCount": 20, "commandMotorControlContract": contract_env.control_contract,
            "trainingCommandGeneratorContract": contract_env.training_command_generator_contract,
            "commandMotorObservationContract": policy.command_motor_observation_contract,
            "sourceCheckpoint": str(source_path), "sourceCheckpointSha256": source_sha,
            "initialTimesteps": initial, "initialPhysicalControlTimesteps": initial_controls,
            "sameBodyTransferAudit": transfer,
            "strictResumeAudit": resume_audit,
            "initialPpoUpdateEpochs": initial_updates,
            "localCommandCurriculumUsed": curriculum,
            "rewardChangedDuringThisTrainingPhase": not bool(args.resume),
            "criticValueScaleAtInitialization": (transfer["criticValueScale"] if transfer else None),
            "commandMotorRewardContract": policy.command_motor_reward_contract,
            "rewardV3InitialPhysicalControlTimesteps": policy.command_motor_reward_v3_initial_physical_controls,
            "configuredLogStdResetAfterCopyAudit": args.reset_log_std,
            "sourceSha256": hashes, "assetSha256": assets, "avatarProfile": profile,
            "avatarProfileSha256": policy.avatar_profile_sha256, "bodyModelSha256": policy.body_model_sha256,
            "physicalAssumptions": contract_env.unwrapped.physical_assumptions,
            "seed": args.seed, "envCount": args.envs, "trainingSceneSplit": "train",
            "difficultyRange": [args.difficulty_min, args.difficulty_max],
            "commandLookaheadRangeMeters": [args.lookahead_min, args.lookahead_max],
            "maxEpisodeSeconds": args.seconds, "learningRate": args.learning_rate, "entropy": args.entropy,
            "nStepsPerWorker": policy.n_steps, "gammaPer20msControl": policy.gamma,
            "rewardScale": .01, "physicalControlDtSeconds": .02, "physicsSubstepsPerControl": 10,
            "teacherDriverUsedDuringMotorTraining": True, "geometryPathTeacherUsed": False,
            "humanMotionDataCollected": False, "heldOutScenesUsedForTraining": False,
            "rootPositionOverwrittenDuringStep": False, "externalForceApplied": False,
            "automaticallyAdoptedIntoHighController": False,
            "policyDevice": "cpu", "physicsDevice": "cpu", "torchThreads": 1,
            "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__,
                "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__}}
        _dump(output/"initialization.json", common)
        archive = output/"simulation-sources"
        archive.mkdir(exist_ok=True)
        for name, expected in hashes.items():
            source = Path(__file__).parent/name
            if _sha256(source) != expected:
                raise RuntimeError("source changed before motor archive")
            shutil.copyfile(source, archive/name)
        if getattr(args, "initialize_only", False) or getattr(args, "resume_audit_only", False):
            if args.resume and not getattr(args, "resume_audit_only", False):
                raise ValueError("initialize-only requires an explicit source transfer, not resume")
            if (_source_hashes() != hashes or _asset_hashes(profile) != assets
                    or _sha256(source_path) != source_sha):
                raise RuntimeError("motor source changed during explicit baseline initialization")
            policy.save(output/"policy.zip")
            baseline = {**common, "timesteps": initial, "actualDecisionTimesteps": initial,
                "physicalControlTimesteps": initial_controls, "phaseTimesteps": 0,
                "phasePhysicalControlTimesteps": 0, "phasePhysicsSubsteps": 0,
                "physicalControlStepCountExact": True, "initializationOnly": True,
                "resumeAuditOnly": getattr(args, "resume_audit_only", False),
                "ppoUpdateEpochs": int(policy._n_updates), "phasePpoUpdateEpochs": 0,
                "checkpointSha256": _sha256(output/"policy.zip"), "complete": True,
                "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
            _dump(output/"provenance.json", baseline)
            print(json.dumps({"initializationOnly": True, "phaseTimesteps": 0,
                              "checkpointSha256": baseline["checkpointSha256"]}), flush=True)
            return baseline
        started = time.monotonic()
        progress = None
        while int(policy.num_timesteps)-initial < args.steps:
            amount = min(args.steps-(int(policy.num_timesteps)-initial), args.checkpoint_every)
            policy.learn(total_timesteps=amount, reset_num_timesteps=False, callback=counter)
            if (_source_hashes() != hashes or _asset_hashes(profile) != assets
                    or _sha256(source_path) != source_sha):
                raise RuntimeError("isolated motor source, asset or source checkpoint changed")
            assert_command_motor_policy(policy, contract_env)
            controls = int(policy.command_motor_physical_control_timesteps)-initial_controls
            decisions = int(policy.num_timesteps)-initial
            if controls != counter.controls or decisions != controls:
                raise RuntimeError("motor training did not preserve exact 1:1 decision/control counts")
            temporary = output/"policy.pending.zip"
            policy.save(temporary)
            temporary.replace(output/"policy.zip")
            progress = {**common, "timesteps": int(policy.num_timesteps),
                "actualDecisionTimesteps": int(policy.num_timesteps),
                "physicalControlTimesteps": int(policy.command_motor_physical_control_timesteps),
                "phaseTimesteps": decisions, "phasePhysicalControlTimesteps": controls,
                "phasePhysicsSubsteps": controls*CONTROL_SUBSTEPS, "physicalControlStepCountExact": True,
                "sceneSplitAuditPassed": True, "workerSceneAudit": counter.worker_scenes,
                "ppoUpdateEpochs": int(policy._n_updates),
                "phasePpoUpdateEpochs": int(policy._n_updates)-initial_updates,
                "localCommandCurriculumCoverage": copy.deepcopy(counter.curriculum_coverage) if curriculum else None,
                "checkpointSha256": _sha256(output/"policy.zip"), "wallSeconds": time.monotonic()-started,
                "complete": decisions >= args.steps, "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
            _dump(output/"training-progress.json", progress)
            _dump(output/"provenance.json", progress)
        print(json.dumps({key: progress[key] for key in ("phaseTimesteps", "phasePhysicalControlTimesteps",
            "phasePhysicsSubsteps", "wallSeconds", "checkpointSha256")}), flush=True)
        return progress
    finally:
        if vector is not None:
            vector.close()
        contract_env.close()


def evaluate_command_motor(policy, *, avatar_profile, output, scene_seed_base=1_170_100,
                           physics_seed_base=3_170_100, episodes=8, seconds=30.,
                           difficulty_range=(0., .5), command_lookahead_range=(.5, .5),
                           checkpoint_sha256=None):
    """Audit joint-motor command following, never claim learned room navigation."""
    from .body_contract import trace_body_metadata
    from .command_motor import CommandMotorEnv, assert_command_motor_policy
    from .command_navigation_evaluation import append_control_frames
    from .generalization import validate_scene_seed
    from .navigation_evaluation import audit_navigation
    if episodes < 1 or episodes > 32:
        raise ValueError("motor evaluation requires 1..32 distinct development rooms")
    seeds = [validate_scene_seed(scene_seed_base+i, "heldout") for i in range(episodes)]
    if any(seed >= 1_800_000 for seed in seeds):
        raise ValueError("isolated motor development must not consume reserved final navigation cohorts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = _source_hashes()
    rows, recordings = [], []
    transfer = getattr(policy, "command_motor_transfer_audit", None)
    if not isinstance(transfer, dict) or "sourcePhysicalMotorTimesteps" not in transfer:
        raise ValueError("motor evaluation requires the explicit original motor lineage")
    source_controls = int(transfer["sourcePhysicalMotorTimesteps"])
    adaptation_controls = int(policy.command_motor_physical_control_timesteps)-source_controls
    if adaptation_controls < 0 or int(policy.num_timesteps)-source_controls != adaptation_controls:
        raise ValueError("motor adaptation counters disagree with source motor lineage")
    adapted = adaptation_controls > 0
    reward_v3_controls = int(policy.command_motor_physical_control_timesteps)-int(
        getattr(policy, "command_motor_reward_v3_initial_physical_controls", policy.num_timesteps))
    if reward_v3_controls < 0:
        raise ValueError("reward-v3 adaptation counter precedes its explicit initialization")
    controller = ("reward-v3 adapted command-conditioned joint motor + analytic final-goal bearing driver"
                  if reward_v3_controls > 0 else
                  "copied joint motor baseline + analytic final-goal bearing driver; no reward-v3 adaptation")
    env = CommandMotorEnv(mode="turn", split="heldout", base_seed=seeds[0], randomize=True,
        avatar_profile=avatar_profile, max_episode_seconds=seconds,
        difficulty_range=difficulty_range, command_lookahead_range=command_lookahead_range,
        record_trajectory=True)
    try:
        assert_command_motor_policy(policy, env)
        for index, seed in enumerate(seeds):
            observation, _ = env.reset(seed=physics_seed_base+index, options={
                "scene_seed": seed, "physics_seed": physics_seed_base+index})
            assert_command_motor_policy(policy, env)
            body = env.unwrapped
            trace = {"version": 1, "task": "walk", "dt": body.dt,
                "sizeConvention": "mujoco", "geometryScope": "complete-world", "scene": env.scene,
                "controller": controller,
                "navigation": {"mode": "turn", "sceneSeed": seed, "physicsSeed": physics_seed_base+index,
                    "split": "heldout", "evaluationRole": "motor_development",
                    "checkpointSha256": checkpoint_sha256, "isLearnedRoomNavigator": False,
                    "sourcePhysicalMotorTimesteps": source_controls,
                    "motorAdaptationPhysicalControls": adaptation_controls,
                    "motorAdaptationTrained": adapted,
                    "rewardV3AdaptationPhysicalControls": reward_v3_controls,
                    "rewardV3AdaptationTrained": reward_v3_controls > 0,
                    "analyticCommandDriverUsed": True, "fixedWaypointsSupplied": False,
                    "commandMotorControlContract": env.control_contract,
                    "trainingCommandGeneratorContract": env.training_command_generator_contract},
                "frames": [env.snapshot()]}
            trace.update(trace_body_metadata(body))
            terminated = truncated = False
            reward_sum, controls, observed_substeps = 0., 0, 0
            while not (terminated or truncated):
                action = policy.predict(observation, deterministic=True)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                append_control_frames(trace, env.last_control_frames, expected_count=1, control_dt=body.dt)
                observed_substeps += info["command_motor"]["observedPhysicsSubstepsThisControl"]
                controls += 1
                reward_sum += reward
            quality = audit_navigation(trace)
            trace["navigationQuality"] = quality
            final = trace["frames"][-1]["stats"]
            if observed_substeps != controls*10:
                raise RuntimeError("motor evaluation missed a physical substep")
            row = {"sceneSeed": seed, "physicsSeed": physics_seed_base+index,
                "seconds": float(body.data.time), "physical20msControlSteps": controls,
                "physical2msSubstepsObserved": observed_substeps, "return": reward_sum,
                "fell": bool(final["fall"]), "collisionFree": quality["gates"]["collisionFree"],
                "arrivalStopped": all(quality["gates"][key] for key in
                    ("goalReached", "stopped", "bothFeetSupport", "sustainedStop")),
                "physicalGaitVerified": quality["gates"]["physicalRepeatedGait"],
                "commandTaskVerified": quality["navigationVerified"],
                "failedGates": quality["failedGates"], "quality": quality, "final": final,
                "maximumExternalForce": 0., "actualFinalGoalFixedEvery2ms": True,
                "sceneSha256": env.navigation_env._scene_sha256}
            rows.append(row)
            kind = "verified" if quality["navigationVerified"] else "failed"
            if index == 0 or not any(r["kind"] == kind for r in recordings):
                filename = f"command-motor-{seed}-{kind}.json"
                _dump(output/filename, trace)
                recordings.append({"sceneSeed": seed, "kind": kind, "filename": filename,
                    "traceSha256": _sha256(output/filename)})
            if sources != _source_hashes():
                raise RuntimeError("motor evaluation source changed")
            _dump(output/"evaluation-progress.json", {"episodes": rows, "evaluationRole": "motor_development"})
            print(json.dumps({"sceneSeed": seed, "seconds": row["seconds"],
                "commandTaskVerified": row["commandTaskVerified"], "failedGates": row["failedGates"]}), flush=True)
        result = {"role": controller,
            "evaluationRole": "motor_development", "isLearnedRoomNavigator": False,
            "episodeCount": len(rows), "episodes": rows, "recordings": recordings,
            "checkpointSha256": checkpoint_sha256, "sourceSha256": sources,
            "commandMotorControlContract": env.control_contract,
            "trainingCommandGeneratorContract": env.training_command_generator_contract,
            "avatarProfileSha256": body.avatar_profile_sha256, "bodyModelSha256": body.body_model_sha256,
            "actualModelRlPhysicalTimesteps": int(policy.num_timesteps),
            "sourcePhysicalMotorTimesteps": source_controls,
            "motorAdaptationPhysicalControls": adaptation_controls,
            "motorAdaptationTrained": adapted,
            "rewardV3AdaptationPhysicalControls": reward_v3_controls,
            "rewardV3AdaptationTrained": reward_v3_controls > 0,
            "commandTaskVerifiedRate": sum(row["commandTaskVerified"] for row in rows)/len(rows),
            "survivalRate": sum(not row["fell"] for row in rows)/len(rows),
            "collisionFreeRate": sum(row["collisionFree"] for row in rows)/len(rows),
            "arrivalStoppedRate": sum(row["arrivalStopped"] for row in rows)/len(rows),
            "physicalGaitPassRate": sum(row["physicalGaitVerified"] for row in rows)/len(rows),
            "proficiencyPassed": False, "anyRoomNavigationVerified": False,
            "automaticallyAdoptedIntoHighController": False}
        _dump(output/"evaluation.json", result)
        return result
    finally:
        env.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-policy", help="Explicitly copy a legacy same-body final-goal 110/20 motor")
    source.add_argument("--resume", help="Resume a checkpoint with the new command-motor semantic contract")
    source.add_argument("--initialize-from-command",
                        help="Explicitly copy a reward-v2 command actor to v3; zero final value head and reset Adam. Direct v1 is unsupported")
    parser.add_argument("--steps", type=int, default=102400)
    parser.add_argument("--checkpoint-every", type=int, default=25600)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90301)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.5)
    parser.add_argument("--lookahead-min", type=float, default=.5)
    parser.add_argument("--lookahead-max", type=float, default=.5)
    parser.add_argument("--learning-rate", type=float, default=.00005)
    parser.add_argument("--entropy", type=float, default=.0002)
    exploration = parser.add_mutually_exclusive_group()
    exploration.add_argument("--reset-log-std", type=float, default=-2.4)
    exploration.add_argument("--keep-log-std", action="store_const", dest="reset_log_std", const=None,
                             help="Preserve the source log-std after exact actor transfer")
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate an explicitly transferred/new motor; no actor updates")
    parser.add_argument("--initialize-only", action="store_true",
                        help="Save an audited transfer baseline and source archive without PPO learning")
    parser.add_argument("--local-command-curriculum", action="store_true",
                        help="Train-only balanced 6..8 second local-command holds; requires strict R3 --resume --keep-log-std")
    parser.add_argument("--resume-audit-only", action="store_true",
                        help="Save an exact actor/critic/std/Adam/counter resume baseline with zero additional PPO updates")
    parser.add_argument("--evaluation-scene-seed-base", type=int, default=1_170_100)
    parser.add_argument("--evaluation-physics-seed-base", type=int, default=3_170_100)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    return parser


def main():
    args = argument_parser().parse_args()
    _validated_args(args)
    if args.evaluate_only and args.initialize_only:
        raise ValueError("evaluate-only and initialize-only cannot be combined")
    if args.evaluate_only:
        if not args.resume:
            raise ValueError("evaluation requires a new-semantic motor checkpoint via --resume")
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        checkpoint = _checkpoint_path(args.resume)
        policy = PPO.load(checkpoint, device="cpu")
        evaluate_command_motor(policy, avatar_profile=args.avatar_profile, output=args.output,
            scene_seed_base=args.evaluation_scene_seed_base, physics_seed_base=args.evaluation_physics_seed_base,
            episodes=args.evaluation_episodes, seconds=args.seconds,
            difficulty_range=(args.difficulty_min, args.difficulty_max),
            command_lookahead_range=(args.lookahead_min, args.lookahead_max),
            checkpoint_sha256=_sha256(checkpoint))
    else:
        train(args)


if __name__ == "__main__":
    main()
