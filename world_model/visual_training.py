"""Isolated CUDA depth-CNN navigation training over a frozen physical motor.

Geometry is allowed only in offline train-label generation and reward. Actor
inputs are genuine camera frames plus the declared eleven-value local state.
Supervised optimizer updates, PPO decisions and actual physics ticks are
counted separately. This module never publishes a checkpoint to the avatar.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .command_motor_training import _dump, _sha256, _state_sha256


def source_hashes():
    from .command_motor_training import _source_hashes
    hashes = _source_hashes()
    for name in ("depth_perception.py", "visual_navigation.py", "visual_policy.py",
                 "visual_teacher.py", "visual_training.py", "visual_evaluation.py"):
        hashes[name] = _sha256(Path(__file__).parent/name)
    return hashes


def make_env(motor, profile, seed, seconds=30., difficulty=(0., .5), recording=False):
    from .visual_navigation import VisualNavigationEnv
    return VisualNavigationEnv(motor_checkpoint=motor, decision_steps=5,
        depth_stack=3, record_trajectory=recording, mode="obstacle", split="train",
        base_seed=seed, difficulty_range=difficulty, randomize=True,
        max_episode_seconds=seconds, avatar_profile=profile)


def collect_labels(motor, profile, *, episodes, seed, seconds, difficulty):
    """Actual physical teacher rollouts, including every unsuccessful episode."""
    from .visual_teacher import TrainingGeometryTeacher
    env = make_env(motor, profile, seed, seconds, difficulty)
    images, states, actions, records = [], [], [], []
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed+index, options={
                "scene_seed": seed+index, "physics_seed": seed+10000+index})
            teacher = TrainingGeometryTeacher(env)
            done, count, substeps, decisions = False, 0, 0, 0
            while not done:
                action = teacher.action()
                images.append(obs["depth"].copy())
                states.append(obs["state"].copy())
                actions.append(action)
                obs, _, terminated, truncated, info = env.step(action)
                count += int(info["command_control_steps"])
                substeps += int(info["command_physics_substeps"])
                decisions += 1
                done = terminated or truncated
            records.append({"sceneSeed": seed+index, "physicsSeed": seed+10000+index,
                "decisions": decisions, "physicalControls": count, "physicsSubsteps": substeps,
                "finalNavigation": copy.deepcopy(info.get("navigation", {}))})
            print(json.dumps({"phase": "teacher-collection", **records[-1]}), flush=True)
    finally:
        env.close()
    return {"depth": np.asarray(images, dtype=np.float32),
            "state": np.asarray(states, dtype=np.float32)}, np.asarray(actions, dtype=np.float32), records


def fit_teacher(model, observations, actions, *, epochs=25, batch_size=256, seed=0):
    """Supervised CNN/actor warm start; neither PPO nor physics is advanced."""
    import torch
    rng, losses, updates = np.random.default_rng(seed), [], 0
    tensors = {key: torch.as_tensor(value, device=model.device) for key, value in observations.items()}
    targets = torch.as_tensor(actions, device=model.device)
    before = _state_sha256(model.policy.state_dict())
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        epoch_losses = []
        for indices in np.array_split(rng.permutation(len(actions)), max(1, math.ceil(len(actions)/batch_size))):
            obs = {key: value[indices] for key, value in tensors.items()}
            mean = model.policy.get_distribution(obs).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        if epoch % 5 == 0:
            print(json.dumps({"phase": "cuda-supervised", "epoch": epoch+1, "mse": losses[-1]}), flush=True)
    with torch.no_grad():
        model.policy.value_net.weight.zero_()
        model.policy.value_net.bias.zero_()
        model.policy.log_std.copy_(torch.tensor([-1.5, -3.], device=model.device))
    # PPO starts with a fresh optimizer; supervised Adam is not represented as
    # on-policy reinforcement-learning updates in any saved counter.
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(), lr=float(model.lr_schedule(1.)), **model.policy.optimizer_kwargs)
    return {"examples": len(actions), "epochs": epochs, "optimizerSteps": updates,
        "lossMse": losses, "policyTensorShaBefore": before,
        "policyTensorShaAfter": _state_sha256(model.policy.state_dict()),
        "device": str(model.device), "ppoDecisionsAdded": 0, "ppoAdamResetAfterSupervision": True,
        "geometryPrivilegedLabelsTrainOnly": True,
        "labelsAreNotClaimsOfPhysicallyVerifiedTeacherSuccess": True}


def depth_dependence(model, observations):
    """Hold state fixed and replace only pixels; this is not a success metric."""
    actual = model.predict(observations, deterministic=True)[0]
    blank = {key: value.copy() for key, value in observations.items()}
    blank["depth"][:] = 1.
    alternative = model.predict(blank, deterministic=True)[0]
    return {"stateHeldExactlyFixed": True, "blankDepthValue": 1.,
        "actionMaxAbsDifference": float(np.max(np.abs(actual-alternative))),
        "actionMeanAbsDifference": float(np.mean(np.abs(actual-alternative))),
        "sampleCount": len(actual), "isNavigationSuccessEvidence": False}


def resume_visual_policy(checkpoint, vector, *, body_env, seed):
    """Resume actor/critic/std/Adam and exact physical counters without transfer."""
    from stable_baselines3 import PPO
    from .visual_navigation import assert_visual_policy
    source = PPO.load(checkpoint, device="cpu")
    assert_visual_policy(source, body_env)
    count = getattr(source, "visual_physical_control_timesteps", None)
    substeps = getattr(source, "visual_physics_substeps", None)
    if (type(count) is not int or count < 0 or type(substeps) is not int
            or substeps != 10*count or type(source.num_timesteps) is not int
            or source.num_timesteps < 0 or type(source._n_updates) is not int
            or source._n_updates < 0):
        raise ValueError("visual resume requires exact native control and PPO counters")
    model = PPO.load(checkpoint, env=vector, device="cuda", seed=seed)
    assert_visual_policy(model, body_env)
    weights, adam = _state_sha256(source.policy.state_dict()), _state_sha256(source.policy.optimizer.state_dict())
    if (_state_sha256(model.policy.state_dict()) != weights
            or _state_sha256(model.policy.optimizer.state_dict()) != adam
            or (model.num_timesteps, model._n_updates, model.visual_physical_control_timesteps,
                model.visual_physics_substeps) != (source.num_timesteps, source._n_updates, count, substeps)):
        raise RuntimeError("visual resume changed actor/critic/std/Adam or counters")
    return model, {"sourceCheckpointSha256": _sha256(checkpoint),
        "sourcePolicyTensorSha256": weights, "sourceAdamStateSha256": adam,
        "actorCriticLogStdAndAdamPreservedExactly": True,
        "sourcePpoDecisions": source.num_timesteps, "sourcePpoUpdateEpochs": source._n_updates,
        "sourcePhysicalControls": count, "sourcePhysicsSubsteps": substeps,
        "criticReset": False, "freshAdam": False, "teacherFittingRepeated": False}


def train(args):
    import torch
    import mujoco
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from .avatar_profile import load_avatar_profile
    from .body_contract import bind_policy_body
    from .navigation_training import _asset_hashes
    from .visual_policy import visual_policy_kwargs
    from .visual_navigation import assert_visual_policy

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA is required; refusing a silent CPU fallback")
    if not 1 <= args.envs <= 32 or args.steps < 1 or args.bc_episodes < 0:
        raise ValueError("invalid workers/steps/teacher count")
    if (not math.isfinite(args.seconds) or args.seconds <= 0 or args.bc_epochs < 1
            or not 0 <= args.difficulty_min <= args.difficulty_max <= 1
            or args.n_steps < 2 or args.batch_size < 2
            or (args.n_steps*args.envs) % args.batch_size != 0
            or args.checkpoint_every < 1 or args.eval_episodes < 0
            or not 1_000_000 <= args.eval_seed < 1_800_000
            or args.eval_seed+args.eval_episodes > 1_800_000):
        raise ValueError("invalid run sizes or development evaluation cohort")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not 0 <= args.seed < 900000 or args.seed+100+args.bc_episodes+10000 >= 1000000:
        raise ValueError("all teaching/training seeds must remain in train domain")
    if args.resume and args.bc_episodes:
        raise ValueError("resumed policy must not repeat teacher fitting; set --bc-episodes 0")
    output, motor = Path(args.output), Path(args.motor_policy)
    if (output/"policy.zip").exists() or (output/"started.json").exists():
        raise ValueError("use a fresh output directory to preserve training provenance")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    profile = load_avatar_profile(args.avatar_profile)
    sources, motor_sha = source_hashes(), _sha256(motor)
    assets = _asset_hashes(profile)
    for name in sources:
        destination = output/"source"/name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, destination)
    started = {"version": 1, "startedUtc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "sourceSha256": sources, "assetSha256": assets,
        "motorCheckpointSha256": motor_sha, "cudaDeviceName": torch.cuda.get_device_name(),
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torchVersion": torch.__version__, "mujocoVersion": mujoco.__version__,
        "sb3Version": stable_baselines3.__version__, "sourceMotorFrozen": True,
        "renderer": "CPU MuJoCo pinhole ray query; actual CUDA CNN and PPO",
        "highLevelInput": "3 genuine depth frames and 11 local state values",
        "lowLevelExistingRangeSensorsRetained": True, "publishedToLiveAvatar": False,
        "finalReservedCohortUsed": False}
    _dump(output/"started.json", started)
    probe = make_env(motor, profile, args.seed, args.seconds,
                     (args.difficulty_min, args.difficulty_max))
    probe.reset(seed=args.seed)
    if args.resume:
        # Validate a persisted contract before paying to spawn all workers.
        try:
            assert_visual_policy(PPO.load(args.resume, device="cpu"), probe)
        except Exception:
            probe.close()
            raise
    def factory(index):
        def create():
            import torch
            torch.set_num_threads(1)
            return make_env(motor, profile, args.seed+1000+index, args.seconds,
                            (args.difficulty_min, args.difficulty_max))
        return create
    vector = (SubprocVecEnv([factory(i) for i in range(args.envs)], start_method="spawn")
              if args.envs > 1 else DummyVecEnv([factory(0)]))
    try:
        if args.resume:
            model, audit = resume_visual_policy(args.resume, vector, body_env=probe, seed=args.seed)
            if (model.n_steps != args.n_steps or model.batch_size != args.batch_size
                    or float(model.lr_schedule(1.)) != args.learning_rate):
                raise ValueError("resume must preserve declared rollout, batch and learning rate")
            _dump(output/"resume-audit.json", audit)
        else:
            model = PPO("MultiInputPolicy", vector, device="cuda", seed=args.seed,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=4,
                learning_rate=args.learning_rate, ent_coef=.0003, gamma=.99,
                policy_kwargs=visual_policy_kwargs(), verbose=1)
            bind_policy_body(model, probe.unwrapped)
            model.visual_observation_contract = copy.deepcopy(probe.observation_contract)
            model.visual_control_contract = copy.deepcopy(probe.control_contract)
            model.visual_physical_control_timesteps = 0
            model.visual_physics_substeps = 0
        assert_visual_policy(model, probe)
        if not all(parameter.device.type == "cuda" for parameter in model.policy.parameters()):
            raise RuntimeError("all high policy tensors must actually reside on CUDA")
        model.save(output/("resumed-policy.zip" if args.resume else "untrained-policy.zip"))
        initial_decisions = int(model.num_timesteps)
        initial_controls = model.visual_physical_control_timesteps
        initial_substeps = model.visual_physics_substeps
        initial_updates = int(model._n_updates)
        observations = None
        if args.bc_episodes:
            observations, actions, teaching = collect_labels(motor, profile,
                episodes=args.bc_episodes, seed=args.seed+100,
                seconds=args.seconds, difficulty=(args.difficulty_min, args.difficulty_max))
            np.savez_compressed(output/"teacher-observations.npz", **observations, actions=actions)
            _dump(output/"teacher-collection.json", {"episodes": teaching,
                "split": "train", "ppoTimestepsAdded": 0,
                "actualPhysicalControls": sum(x["physicalControls"] for x in teaching),
                "actualPhysicsSubsteps": sum(x["physicsSubsteps"] for x in teaching)})
            _dump(output/"supervision.json", fit_teacher(model, observations, actions,
                epochs=args.bc_epochs, batch_size=args.batch_size, seed=args.seed))
        model.save(output/"before-ppo-policy.zip")
        from .visual_evaluation import evaluate
        evaluation_kwargs = dict(motor_checkpoint=motor,
            scene_seeds=range(args.eval_seed, args.eval_seed+args.eval_episodes),
            physics_seed_base=args.eval_seed+2000000, avatar_profile=profile,
            seconds=args.seconds, difficulty_range=(args.difficulty_min, args.difficulty_max))
        if args.eval_episodes:
            evaluate(model, output=output/"before", checkpoint_sha256=_sha256(output/"before-ppo-policy.zip"), **evaluation_kwargs)
        before_weights = {key: tensor.detach().cpu().clone() for key, tensor in model.policy.state_dict().items()}
        baseline_std = model.policy.log_std.detach().cpu().tolist()
        start, checkpoints = time.monotonic(), []
        class Counter(BaseCallback):
            def __init__(self):
                super().__init__()
                self.controls = self.substeps = self.episodes = 0
                self.scenes = set()
                self.last_report = 0.
                self.last_checkpoint = initial_decisions
            def _on_step(self):
                for info in self.locals["infos"]:
                    controls, substeps = info["command_control_steps"], info["command_physics_substeps"]
                    if type(controls) is not int or not 1 <= controls <= 5 or substeps != 10*controls:
                        raise ValueError("actual 20ms/2ms accounting violated")
                    nav = info["navigation"]
                    if nav["split"] != "train" or not 0 <= nav["scene_seed"] < 1000000:
                        raise ValueError("non-training scene entered PPO")
                    self.scenes.add(nav["scene_seed"])
                    self.controls += controls
                    self.substeps += substeps
                self.episodes += int(np.sum(self.locals["dones"]))
                self.model.visual_physical_control_timesteps = initial_controls+self.controls
                self.model.visual_physics_substeps = initial_substeps+self.substeps
                elapsed = time.monotonic()-start
                if elapsed-self.last_report >= 20:
                    self.last_report = elapsed
                    progress = {"phase": "actual-cuda-ppo", "decisions": self.num_timesteps,
                        "physicalControls": self.controls, "physicsSubsteps": self.substeps,
                        "episodes": self.episodes, "distinctTrainScenes": len(self.scenes),
                        "elapsedSeconds": elapsed, "decisionsPerSecond": (self.num_timesteps-initial_decisions)/elapsed,
                        "ppoUpdateEpochs": int(self.model._n_updates),
                        "cudaAllocatedBytes": torch.cuda.memory_allocated()}
                    _dump(output/"progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if self.num_timesteps-self.last_checkpoint >= args.checkpoint_every:
                    self.last_checkpoint = self.num_timesteps
                    path = output/f"checkpoint-{self.num_timesteps}.zip"
                    self.model.save(path)
                    checkpoints.append({"decisions": self.num_timesteps, "path": path.name,
                                        "sha256": _sha256(path)})
                return True
        counter = Counter()
        model.learn(total_timesteps=args.steps, callback=counter, reset_num_timesteps=False)
        elapsed = time.monotonic()-start
        model.save(output/"policy.zip")
        changed = {key: float((tensor.detach().cpu()-before_weights[key]).abs().max())
            for key, tensor in model.policy.state_dict().items()
            if not torch.equal(tensor.detach().cpu(), before_weights[key])}
        cnn_changed = {key: value for key, value in changed.items() if "depth_encoder" in key}
        adam_steps = [float(state["step"].detach().cpu()) for state in model.policy.optimizer.state.values()]
        if not cnn_changed or not adam_steps or min(adam_steps) < 1 or model.device.type != "cuda":
            raise RuntimeError("actual CUDA PPO CNN/Adam update was not proven")
        report = {"version": 1, "actualPpoDecisions": int(model.num_timesteps),
            "actualPpoPhysicalControls": model.visual_physical_control_timesteps,
            "actualPpoPhysicsSubsteps": model.visual_physics_substeps,
            "runPpoDecisions": int(model.num_timesteps)-initial_decisions,
            "runPpoPhysicalControls": counter.controls, "runPpoPhysicsSubsteps": counter.substeps,
            "initialPpoDecisions": initial_decisions, "initialPpoUpdateEpochs": initial_updates,
            "actualPpoEpisodes": counter.episodes, "distinctTrainScenes": sorted(counter.scenes),
            "ppoUpdateEpochs": int(model._n_updates), "ppoAdamStepsMin": min(adam_steps),
            "ppoAdamStepsMax": max(adam_steps), "changedTensorCount": len(changed),
            "cnnChangedTensorsMaxAbs": cnn_changed, "cudaDevice": str(model.device),
            "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated(),
            "elapsedPpoSeconds": elapsed, "decisionsPerSecond": (model.num_timesteps-initial_decisions)/elapsed,
            "initialPpoLogStd": baseline_std, "finalLogStd": model.policy.log_std.detach().cpu().tolist(),
            "checkpointSha256": _sha256(output/"policy.zip"), "checkpoints": checkpoints,
            "motorCheckpointUnchanged": _sha256(motor) == motor_sha,
            "sourceBytesUnchanged": source_hashes() == sources,
            "assetsUnchanged": _asset_hashes(profile) == assets,
            "publishedToLiveAvatar": False, "phase2Complete": False,
            "wholeControllerVisionOnly": False}
        if not all(report[key] for key in ("motorCheckpointUnchanged", "sourceBytesUnchanged", "assetsUnchanged")):
            raise RuntimeError("frozen sources, body assets or motor changed during learning")
        if observations is None and args.resume:
            previous_data = Path(args.resume).parent/"teacher-observations.npz"
            if previous_data.exists():
                with np.load(previous_data) as data:
                    observations = {key: data[key][:128].copy() for key in ("depth", "state")}
        if observations is not None:
            _dump(output/"depth-dependence.json", depth_dependence(model,
                {key: value[:128] for key, value in observations.items()}))
        _dump(output/"training.json", report)
        if args.eval_episodes:
            evaluate(model, output=output/"after", checkpoint_sha256=report["checkpointSha256"], **evaluation_kwargs)
            evaluate(model, output=output/"after-blank-depth", blank_depth=True,
                     checkpoint_sha256=report["checkpointSha256"], **evaluation_kwargs)
        _dump(output/"completed.json", {**report, "completedUtc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(report), flush=True)
    finally:
        vector.close()
        probe.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-policy", required=True)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Existing visual checkpoint; preserves all weights, Adam and counters")
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90801)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.5)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--bc-episodes", type=int, default=16)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=8192)
    parser.add_argument("--eval-seed", type=int, default=1180100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    return parser


if __name__ == "__main__":
    train(argument_parser().parse_args())
