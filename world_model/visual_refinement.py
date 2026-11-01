"""Conservative physical PPO refinement of an existing visual checkpoint.

The encoder and action noise are frozen to preserve the supervised visual
representation while the value MLP learns its initially unknown return scale.
Only the actor/value MLPs and heads adapt. This is a separate experiment: no
source checkpoint, running experiment, motor or published avatar is modified.
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


def configure_conservative_refinement(model, *, learning_rate=1e-5, target_kl=.01):
    """Change declared optimization controls without resetting weights/Adam/counts.

    Frozen parameters remain in the optimizer, so existing Adam parameter
    identities and moments survive. Adam never updates a tensor with grad=None.
    A subsequent standard PPO.load() must call this function again because
    torch state_dict does not serialize requires_grad.
    """
    import torch
    from stable_baselines3.common.utils import FloatSchedule

    if (not math.isfinite(learning_rate) or not 0 < learning_rate <= 3e-5
            or not math.isfinite(target_kl) or not 0 < target_kl <= .02):
        raise ValueError("conservative learning rate and target KL must be finite and bounded")
    counts = (model.num_timesteps, model._n_updates,
              getattr(model, "visual_physical_control_timesteps", None),
              getattr(model, "visual_physics_substeps", None))
    if (any(type(value) is not int or value < 0 for value in counts)
            or counts[3] != 10*counts[2]):
        raise ValueError("refinement requires exact native PPO and physical counters")
    weights_before = _state_sha256(model.policy.state_dict())
    adam_before = _state_sha256(model.policy.optimizer.state_dict())
    previous_lr = float(model.lr_schedule(1.))
    previous_target_kl = model.target_kl
    features = model.policy.features_extractor
    frozen_names = []
    for name, parameter in model.policy.named_parameters():
        if name.startswith(("features_extractor.", "pi_features_extractor.",
                            "vf_features_extractor.")) or name == "log_std":
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen_names.append(name)
    if not frozen_names or any(parameter.requires_grad for parameter in features.parameters()):
        raise RuntimeError("the complete trained depth/state representation must be frozen")
    # lr_schedule is called by PPO before every update. Existing Adam moments
    # are untouched; only the next step size and trust-region stop are changed.
    model.learning_rate = float(learning_rate)
    model.lr_schedule = FloatSchedule(float(learning_rate))
    model.target_kl = float(target_kl)
    model.visual_refinement_contract = {
        "version": 1, "frozenFeatureExtractor": True, "frozenActionLogStd": True,
        "learningRate": float(learning_rate), "targetKl": float(target_kl),
        "requiresReapplyAfterPpoLoad": True, "auxiliaryTeacherLoss": False,
        "privilegedInputAddedToActor": False,
    }
    if (_state_sha256(model.policy.state_dict()) != weights_before
            or _state_sha256(model.policy.optimizer.state_dict()) != adam_before
            or counts != (model.num_timesteps, model._n_updates,
                          model.visual_physical_control_timesteps, model.visual_physics_substeps)):
        raise RuntimeError("configuration unexpectedly changed weights, Adam or physical counters")
    return {"sourcePolicyTensorSha256": weights_before, "sourceAdamStateSha256": adam_before,
        "weightsPreservedExactly": True, "adamMomentsPreservedExactly": True,
        "optimizerReset": False, "criticReset": False, "supervisionRepeated": False,
        "initialPpoDecisions": counts[0], "initialPpoUpdateEpochs": counts[1],
        "initialPhysicalControls": counts[2], "initialPhysicsSubsteps": counts[3],
        "previousLearningRate": previous_lr, "previousTargetKl": previous_target_kl,
        "frozenParameterNames": frozen_names, **model.visual_refinement_contract}


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from .avatar_profile import load_avatar_profile
    from .navigation_training import _asset_hashes
    from .visual_evaluation import evaluate
    from .visual_navigation import assert_visual_policy
    from .visual_training import make_env, resume_visual_policy, source_hashes

    if not torch.cuda.is_available():
        raise RuntimeError("actual CUDA is required; refusing CPU-only refinement")
    if (not 1 <= args.envs <= 32 or args.steps < 1 or not 0 <= args.seed < 900000
            or not math.isfinite(args.seconds) or args.seconds <= 0
            or not 0 <= args.difficulty_min <= args.difficulty_max <= 1
            or args.checkpoint_every < 1
            or args.eval_episodes < 0 or not 1_000_000 <= args.eval_seed < 1_800_000
            or args.eval_seed+args.eval_episodes > 1_800_000):
        raise ValueError("invalid run sizes or train/development domains")
    output, source, motor = Path(args.output), Path(args.resume), Path(args.motor_policy)
    if (output/"started.json").exists() or (output/"policy.zip").exists():
        raise ValueError("use a fresh output directory to preserve experiment provenance")
    # Validate checkpoint and declared changes before creating an output.
    preliminary = PPO.load(source, device="cpu")
    preliminary_audit = configure_conservative_refinement(preliminary,
        learning_rate=args.learning_rate, target_kl=args.target_kl)
    profile = load_avatar_profile(args.avatar_profile)
    probe = make_env(motor, profile, args.seed, args.seconds,
                     (args.difficulty_min, args.difficulty_max))
    try:
        assert_visual_policy(preliminary, probe)
    except Exception:
        probe.close()
        raise
    del preliminary
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    sources = source_hashes()
    sources["visual_refinement.py"] = _sha256(Path(__file__))
    assets, motor_sha, source_sha = _asset_hashes(profile), _sha256(motor), _sha256(source)
    output.mkdir(parents=True, exist_ok=True)
    for name in sources:
        target = output/"source"/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, target)
    _dump(output/"started.json", {"version": 1, "arguments": vars(args),
        "startedUtc": datetime.now(timezone.utc).isoformat(), "sourceSha256": sources,
        "assetSha256": assets, "motorCheckpointSha256": motor_sha,
        "sourceCheckpointSha256": source_sha, "sourcePpoDecisions": preliminary_audit["initialPpoDecisions"],
        "device": "cuda:0", "cudaDeviceName": torch.cuda.get_device_name(),
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "publishedToLiveAvatar": False, "finalReservedCohortUsed": False})
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
        model, resume_audit = resume_visual_policy(source, vector, body_env=probe, seed=args.seed)
        audit = configure_conservative_refinement(model,
            learning_rate=args.learning_rate, target_kl=args.target_kl)
        _dump(output/"resume-audit.json", {"nativeResume": resume_audit, "refinement": audit})
        initial = (model.num_timesteps, model._n_updates,
                   model.visual_physical_control_timesteps, model.visual_physics_substeps)
        weights = {k:v.detach().cpu().clone() for k,v in model.policy.state_dict().items()}
        if not all(p.device.type == "cuda" for p in model.policy.parameters()):
            raise RuntimeError("all high policy tensors must reside on real CUDA")
        model.save(output/"before-ppo-policy.zip")
        evaluation_kwargs = dict(motor_checkpoint=motor,
            scene_seeds=range(args.eval_seed, args.eval_seed+args.eval_episodes),
            physics_seed_base=args.eval_seed+2000000, avatar_profile=profile,
            seconds=args.seconds, difficulty_range=(args.difficulty_min, args.difficulty_max))
        if args.eval_episodes:
            evaluate(model, output=output/"before",
                     checkpoint_sha256=_sha256(output/"before-ppo-policy.zip"), **evaluation_kwargs)
        start = time.monotonic()
        class Counter(BaseCallback):
            def __init__(self):
                super().__init__()
                self.controls = self.substeps = self.episodes = 0
                self.scenes = set()
                self.last_report = 0.
                self.last_checkpoint = initial[0]
            def _on_step(self):
                for info in self.locals["infos"]:
                    c, s, nav = info["command_control_steps"], info["command_physics_substeps"], info["navigation"]
                    if type(c) is not int or not 1 <= c <= 5 or s != 10*c:
                        raise RuntimeError("actual native physical control accounting violated")
                    if nav["split"] != "train" or not 0 <= nav["scene_seed"] < 1_000_000:
                        raise RuntimeError("non-training observation entered PPO")
                    self.controls += c
                    self.substeps += s
                    self.scenes.add(nav["scene_seed"])
                self.episodes += int(np.sum(self.locals["dones"]))
                self.model.visual_physical_control_timesteps = initial[2]+self.controls
                self.model.visual_physics_substeps = initial[3]+self.substeps
                elapsed = time.monotonic()-start
                if elapsed-self.last_report >= 20:
                    self.last_report = elapsed
                    report = {"phase": "conservative-actual-cuda-ppo", "runDecisions": self.num_timesteps-initial[0],
                        "runPhysicalControls": self.controls, "runPhysicsSubsteps": self.substeps,
                        "episodes": self.episodes, "distinctTrainScenes": len(self.scenes),
                        "elapsedSeconds": elapsed, "ppoUpdateEpochs": int(self.model._n_updates),
                        "cudaAllocatedBytes": torch.cuda.memory_allocated()}
                    _dump(output/"progress.json", report)
                    print(json.dumps(report), flush=True)
                if self.num_timesteps-self.last_checkpoint >= args.checkpoint_every:
                    self.last_checkpoint = self.num_timesteps
                    self.model.save(output/f"checkpoint-{self.num_timesteps}.zip")
                return True
        counter = Counter()
        model.learn(total_timesteps=args.steps, callback=counter, reset_num_timesteps=False)
        model.save(output/"policy.zip")
        changed = {k:float((v.detach().cpu()-weights[k]).abs().max())
                   for k,v in model.policy.state_dict().items() if not torch.equal(v.detach().cpu(),weights[k])}
        if any(name in changed for name in audit["frozenParameterNames"]):
            raise RuntimeError("frozen visual representation or action noise changed")
        if not any(k.startswith("action_net.") or k.startswith("mlp_extractor.policy_net.") for k in changed):
            raise RuntimeError("no actual actor update was proven")
        unchanged = {"sourceBytesUnchanged": all(_sha256(Path(__file__).parent/name)==sha for name,sha in sources.items()),
                     "motorCheckpointUnchanged": _sha256(motor)==motor_sha,
                     "sourceCheckpointUnchanged": _sha256(source)==source_sha,
                     "assetsUnchanged": _asset_hashes(profile)==assets}
        if not all(unchanged.values()):
            raise RuntimeError("frozen input source, motor or assets changed")
        adam = [float(s["step"].detach().cpu()) for s in model.policy.optimizer.state.values()]
        if not adam or max(adam) < 1:
            raise RuntimeError("no actual Adam updates were proven")
        report = {"version": 1, **unchanged, "refinementContract": model.visual_refinement_contract,
            "runPpoDecisions": int(model.num_timesteps)-initial[0], "runPpoPhysicalControls": counter.controls,
            "runPpoPhysicsSubsteps": counter.substeps, "runPpoUpdateEpochs": int(model._n_updates)-initial[1],
            "actualPpoDecisions": int(model.num_timesteps),
            "actualPpoPhysicalControls": model.visual_physical_control_timesteps,
            "actualPpoPhysicsSubsteps": model.visual_physics_substeps,
            "actualPpoEpisodes": counter.episodes, "distinctTrainScenes": sorted(counter.scenes),
            "changedTensorsMaxAbs": changed, "frozenFeaturesAndLogStdByteIdentical": True,
            "ppoAdamStepsMin": min(adam), "ppoAdamStepsMax": max(adam), "device": str(model.device),
            "cudaPeakAllocatedBytes": torch.cuda.max_memory_allocated(),
            "elapsedPpoSeconds": time.monotonic()-start, "checkpointSha256": _sha256(output/"policy.zip"),
            "publishedToLiveAvatar": False, "phase2Complete": False, "wholeControllerVisionOnly": False}
        _dump(output/"training.json", report)
        if args.eval_episodes:
            evaluate(model, output=output/"after", checkpoint_sha256=report["checkpointSha256"], **evaluation_kwargs)
            evaluate(model, output=output/"after-blank-depth", blank_depth=True,
                     checkpoint_sha256=report["checkpointSha256"], **evaluation_kwargs)
        _dump(output/"completed.json", {**report, "completedUtc":datetime.now(timezone.utc).isoformat()})
        print(json.dumps(report), flush=True)
    finally:
        vector.close()
        probe.close()


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--motor-policy", required=True)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=92801)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--difficulty-min", type=float, default=0.)
    parser.add_argument("--difficulty-max", type=float, default=.5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--target-kl", type=float, default=.01)
    parser.add_argument("--checkpoint-every", type=int, default=8192)
    parser.add_argument("--eval-seed", type=int, default=1180100)
    parser.add_argument("--eval-episodes", type=int, default=8)
    return parser


if __name__ == "__main__":
    train(argument_parser().parse_args())
