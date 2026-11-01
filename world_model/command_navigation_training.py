"""Train a two-action room navigator over an explicitly frozen walking PPO.

Navigation decisions run every five 20ms physical control ticks by default.
The motor is a separately hashed, same-body 110-input/20-joint checkpoint;
training changes only the 112-input/2-action navigation policy. No scene from
the held-out range is used here and no route or root pose is supplied.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import gymnasium as gym

from .avatar_profile import load_avatar_profile
from .body_contract import bind_policy_body
from .generalization import validate_scene_seed
from .humanoid import CONTROL_SUBSTEPS
from .navigation_training import _asset_hashes, _dump_json, _sha256


def _checkpoint_path(value):
    path = Path(value)
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _source_hashes():
    names = ["command_navigation_training.py", "command_navigation.py", "command_gait.py", "command_motor_limits.py",
             "navigation_training.py", "navigation.py", "navigation_scenes.py",
             "scenes.py", "sensors.py", "generalization.py", "environment.py",
             "humanoid.py", "avatar_profile.py", "body_contract.py", "locomotion.py",
             "steering.py", "imitation.py", "references.py", "contact_metrics.py"]
    directory = Path(__file__).parent
    # A missing implementation must never become a silently omitted contract.
    return {name: _sha256(directory/name) for name in names}


def _validated_args(args):
    if isinstance(args.steps, bool) or args.steps < 1:
        raise ValueError("steps must be positive")
    if not 1 <= args.envs <= 32 or args.checkpoint_every < 1:
        raise ValueError("envs must be 1..32 and checkpoint-every must be positive")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if not (0 <= args.difficulty_min <= args.difficulty_max <= 1):
        raise ValueError("difficulty limits must be ordered within 0..1")
    if not 0 <= args.seed < 1_000_000 or args.seed + args.envs > 1_000_000:
        raise ValueError("training worker seeds must remain below the held-out boundary 1000000")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.entropy) or args.entropy < 0:
        raise ValueError("entropy must be finite and nonnegative")
    if args.reset_log_std is not None and not math.isfinite(args.reset_log_std):
        raise ValueError("reset-log-std must be finite")
    if not 1 <= args.decision_steps <= 10:
        raise ValueError("decision-steps must be 1..10 physical control ticks")
    if not math.isfinite(args.max_heading_radians) or not 0 < args.max_heading_radians <= .6:
        raise ValueError("max-heading-radians must lie within (0,.6]")
    if not math.isfinite(args.max_lookahead_m) or not .35 <= args.max_lookahead_m <= .5:
        raise ValueError("max-lookahead-m must lie within the verified walking range [.35,.5]")
    if (not math.isfinite(args.heading_slew_radians_per_second)
            or not 0 < args.heading_slew_radians_per_second <= .35):
        raise ValueError("heading-slew-radians-per-second must lie within (0,.35]")
    if args.mode not in ("turn", "obstacle"):
        raise ValueError("mode must be turn or obstacle")
    yaw_limit = getattr(args, "motor_hip_yaw_action_limit", None)
    if yaw_limit is not None and (isinstance(yaw_limit, bool)
            or not math.isfinite(yaw_limit) or not 0 < yaw_limit <= 1):
        raise ValueError("motor-hip-yaw-action-limit must be finite and in (0,1]")
    return args


def resume_command(checkpoint, vector, *, body_env, seed, learning_rate, entropy):
    """Preserve Adam and explicitly reseed newly constructed training workers."""
    from stable_baselines3 import PPO
    from .command_navigation import assert_command_policy

    policy = PPO.load(checkpoint, env=vector, device="cpu", seed=seed)
    assert_command_policy(policy, body_env)
    count = getattr(policy, "command_physical_control_timesteps", None)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("resumed command checkpoint lacks an exact physical-control step count")
    policy.learning_rate = learning_rate
    policy.lr_schedule = lambda progress_remaining: learning_rate
    policy.ent_coef = entropy
    return policy


def make_control_counter(decision_steps, *, audit_scenes=False):
    """Count real motor ticks, including a terminal decision shorter than five."""
    from stable_baselines3.common.callbacks import BaseCallback

    class PhysicalControlCounter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.run_physical_control_steps = 0
            self.worker_scenes = {}
            self._scene_keys = {}

        def _on_step(self):
            infos = self.locals.get("infos", [])
            if len(infos) != self.training_env.num_envs:
                raise ValueError("physical step audit requires one info record per training worker")
            counts = [info.get("command_control_steps") for info in infos]
            if any(isinstance(count, bool) or not isinstance(count, int)
                   or not 1 <= count <= decision_steps for count in counts):
                raise ValueError("command_control_steps must record exact executed motor ticks")
            if audit_scenes:
                for worker, info in enumerate(infos):
                    navigation = info.get("navigation", {})
                    if navigation.get("split") != "train":
                        raise ValueError("training received a scene outside the train split")
                    scene_seed = validate_scene_seed(navigation.get("scene_seed"), "train")
                    key = (navigation.get("episode_index"), scene_seed,
                           navigation.get("physics_seed"), navigation.get("scene_sha256"))
                    if key not in self._scene_keys.setdefault(worker, set()):
                        self._scene_keys[worker].add(key)
                        self.worker_scenes.setdefault(worker, []).append({
                            "sceneSeed": scene_seed, "physicsSeed": navigation.get("physics_seed"),
                            "episodeIndex": navigation.get("episode_index"),
                            "sceneSha256": navigation.get("scene_sha256"),
                            "difficulty": navigation.get("difficulty"),
                            "firstObservedDecisionTimesteps": int(self.model.num_timesteps),
                            "split": "train"})
            amount = sum(counts)
            self.run_physical_control_steps += amount
            self.model.command_physical_control_timesteps += amount
            return True

    return PhysicalControlCounter()


def train(args):
    _validated_args(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    import mujoco
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from .command_navigation import (COMMAND_OBSERVATION_CONTRACT,
                                     CommandNavigationEnv, assert_command_policy)

    torch.set_num_threads(1)
    profile = load_avatar_profile(args.avatar_profile)
    if profile is None:
        raise ValueError("command navigation requires an explicit measured avatar profile")
    motor_path = _checkpoint_path(args.motor_policy)
    motor_sha = _sha256(motor_path)
    source_path = _checkpoint_path(args.resume) if args.resume else None
    source_sha = _sha256(source_path) if source_path else None
    output = Path(args.output)
    if (output/"policy.zip").exists():
        raise ValueError("output already contains a checkpoint; use a new output directory")
    if motor_path.resolve() == (output/"policy.zip").resolve() or (
            source_path and source_path.resolve() == (output/"policy.zip").resolve()):
        raise ValueError("output must not overwrite the frozen motor or resumed navigator")
    hashes, assets = _source_hashes(), _asset_hashes(profile)
    output.mkdir(parents=True, exist_ok=True)

    def factory(index):
        def make():
            env = CommandNavigationEnv(motor_checkpoint=motor_path,
                decision_steps=args.decision_steps, max_heading_radians=args.max_heading_radians,
                max_lookahead_m=args.max_lookahead_m,
                heading_slew_radians_per_second=args.heading_slew_radians_per_second,
                motor_hip_yaw_action_limit=getattr(args, "motor_hip_yaw_action_limit", None),
                record_trajectory=False,
                mode=args.mode, split="train", base_seed=args.seed+index,
                difficulty_range=(args.difficulty_min, args.difficulty_max),
                randomize=True, max_episode_seconds=args.seconds, avatar_profile=profile)
            env.reset(seed=args.seed+index)
            return env
        return make

    contract_env = factory(0)()
    vector = None
    try:
        vector_class = SubprocVecEnv if args.envs > 1 else DummyVecEnv
        kwargs = {"start_method": "fork"} if args.envs > 1 else {}
        vector = vector_class([factory(index) for index in range(args.envs)], **kwargs)
        if args.resume:
            policy = resume_command(source_path, vector, body_env=contract_env, seed=args.seed,
                                    learning_rate=args.learning_rate, entropy=args.entropy)
        else:
            policy = PPO("MlpPolicy", vector, seed=args.seed, device="cpu", verbose=1,
                learning_rate=args.learning_rate, ent_coef=args.entropy, gamma=.99,
                n_steps=256, batch_size=256, n_epochs=5,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}})
            bind_policy_body(policy, contract_env.unwrapped)
            policy.command_observation_contract = copy.deepcopy(COMMAND_OBSERVATION_CONTRACT)
            policy.command_control_contract = copy.deepcopy(contract_env.control_contract)
            policy.command_physical_control_timesteps = 0
        assert_command_policy(policy, contract_env)
        if args.reset_log_std is not None:
            with torch.no_grad():
                policy.policy.log_std.fill_(args.reset_log_std)
        policy.command_training_mode = args.mode
        initial = int(policy.num_timesteps)
        initial_control = policy.command_physical_control_timesteps
        counter = make_control_counter(args.decision_steps, audit_scenes=True)
        common = {"algorithm": "PPO", "task": "phase2_command_navigation", "mode": args.mode,
            "role": "learned heading/lookahead commands over a frozen learned joint motor",
            "motorCheckpoint": str(motor_path), "motorCheckpointSha256": motor_sha,
            "motorObservationCount": 110, "motorActionCount": 20, "motorFrozen": True,
            "motorTrainingTimesteps": int(contract_env._motor.num_timesteps),
            "observationCount": 112, "actionCount": 2,
            "commandObservationContract": COMMAND_OBSERVATION_CONTRACT,
            "commandControlContract": contract_env.control_contract,
            "controlDtSeconds": float(contract_env.unwrapped.dt),
            "maximumDecisionDtSeconds": float(contract_env.unwrapped.dt*args.decision_steps),
            "decisionSteps": args.decision_steps, "gammaPerDecision": float(policy.gamma),
            "sceneSplit": "train", "heldOutScenesUsedForTraining": False,
            "requestedDecisionSteps": args.steps, "initialTimesteps": initial,
            "initialPhysicalControlTimesteps": initial_control,
            "initialization": "resume112" if args.resume else "fresh112to2",
            "sourceCheckpoint": str(source_path) if source_path else None,
            "sourceCheckpointSha256": source_sha,
            "avatarProfile": profile, "avatarProfileSha256": policy.avatar_profile_sha256,
            "bodyModelSha256": policy.body_model_sha256,
            "physicalAssumptions": contract_env.unwrapped.physical_assumptions,
            "sourceSha256": hashes, "assetSha256": assets, "seed": args.seed,
            "envCount": args.envs, "maxEpisodeSeconds": args.seconds,
            "trainingDifficultyRange": [args.difficulty_min, args.difficulty_max],
            "learningRate": args.learning_rate, "entropy": args.entropy,
            "logStdOverride": args.reset_log_std,
            "policyDevice": "cpu", "physicsDevice": "cpu", "torchThreads": 1,
            "rootPositionOverwrittenDuringStep": False, "fixedWaypointsSupplied": False,
            "bodyTransferAllowed": False, "isT2M": False, "neuralWorldModelTrained": False,
            "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__,
                "gymnasium": gym.__version__, "stable_baselines3": stable_baselines3.__version__}}
        _dump_json(output/"initialization.json", common)
        started = time.monotonic()
        while policy.num_timesteps-initial < args.steps:
            amount = min(args.steps-(policy.num_timesteps-initial), args.checkpoint_every)
            policy.learn(total_timesteps=amount, reset_num_timesteps=False, callback=counter)
            if (_source_hashes() != hashes or _asset_hashes(profile) != assets
                    or _sha256(motor_path) != motor_sha
                    or (source_path and _sha256(source_path) != source_sha)):
                raise RuntimeError("frozen motor, source checkpoint, simulator or assets changed during training")
            assert_command_policy(policy, contract_env)
            temporary = output/"policy.pending.zip"
            policy.save(temporary)
            temporary.replace(output/"policy.zip")
            actual_decisions = int(policy.num_timesteps-initial)
            actual_controls = int(policy.command_physical_control_timesteps-initial_control)
            if actual_controls != counter.run_physical_control_steps:
                raise RuntimeError("physical control counts disagree with training callbacks")
            progress = {**common, "timesteps": int(policy.num_timesteps),
                "phaseTimesteps": actual_decisions, "actualDecisionTimesteps": int(policy.num_timesteps),
                "phaseDecisionTimesteps": actual_decisions,
                "physicalControlTimesteps": int(policy.command_physical_control_timesteps),
                "phasePhysicalControlTimesteps": actual_controls,
                "phasePhysicsSubsteps": actual_controls*CONTROL_SUBSTEPS,
                "physicalControlStepCountExact": True,
                "phasePhysicalControlStepsUpperBound": actual_decisions*args.decision_steps,
                "sceneSplitAuditPassed": True,
                "workerSceneAudit": [{"workerIndex": worker, "episodes": episodes}
                    for worker, episodes in sorted(counter.worker_scenes.items())],
                "phaseSimulatedSecondsAcrossWorkers": actual_controls*contract_env.unwrapped.dt,
                "wallSeconds": time.monotonic()-started,
                "checkpointSha256": _sha256(output/"policy.zip"),
                "complete": actual_decisions >= args.steps,
                "generatedAtUtc": datetime.now(timezone.utc).isoformat()}
            _dump_json(output/"training-progress.json", progress)
            _dump_json(output/"provenance.json", progress)
        print(json.dumps({key: progress[key] for key in ("mode", "phaseDecisionTimesteps",
            "phasePhysicalControlTimesteps", "wallSeconds", "checkpointSha256")}), flush=True)
        return progress
    finally:
        if vector is not None:
            vector.close()
        contract_env.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("turn", "obstacle"), required=True)
    parser.add_argument("--motor-policy", required=True, help="Frozen same-body 110-input/20-joint walking PPO ZIP")
    parser.add_argument("--resume", help="Resume a compatible 112-input/2-command navigator")
    parser.add_argument("--steps", type=int, default=102400, help="Requested high-level decision steps, rounded to full PPO rollouts")
    parser.add_argument("--checkpoint-every", type=int, default=20480)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=8701)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=.001)
    parser.add_argument("--reset-log-std", type=float)
    parser.add_argument("--motor-hip-yaw-action-limit", type=float,
                        help="Explicit version-2 limit on the frozen motor's two hip-yaw residual actions")
    parser.add_argument("--decision-steps", type=int, default=5)
    parser.add_argument("--max-heading-radians", type=float, default=.45)
    parser.add_argument("--max-lookahead-m", type=float, default=.5)
    parser.add_argument("--heading-slew-radians-per-second", type=float, default=.175)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
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
