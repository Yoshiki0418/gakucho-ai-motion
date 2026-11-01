"""Train and audit bounded residual policies with held-out physics rollouts.

The neural policy controls joints; the free root remains in MuJoCo dynamics.
Evaluation scenes/seeds are never used for selecting training checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import gymnasium as gym

from .environment import HumanoidRoomEnv
from .body_contract import assert_policy_body, bind_policy_body, trace_body_metadata


class RandomPushes(gym.Wrapper):
    """Training-only domain variation: finite force pulses on the free pelvis."""

    def __init__(self, env, maximum_force, start_range=(1.4, 3.0)):
        super().__init__(env)
        self.maximum_force = maximum_force
        self.start_range = start_range

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        random = self.unwrapped.np_random
        self.push_start = random.uniform(*self.start_range)
        direction = random.uniform(-np.pi, np.pi)
        force = random.uniform(.2, 1.0) * self.maximum_force
        self.push = np.asarray([np.cos(direction) * force, np.sin(direction) * force, 0.])
        return observation, info

    def step(self, action):
        time = self.unwrapped.data.time
        self.unwrapped.apply_push(self.push if self.push_start <= time < self.push_start+.2 else [0, 0, 0])
        return self.env.step(action)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def evaluate(policy=None, *, task="stand", scene=None, seeds=range(10000, 10008),
             seconds=6, randomize=True, push_force=0, trace_path=None,
             reference=False, reference_start=True, physical_footsteps=False, audit_walk=False,
             strict_gait_reward=False, audit_sit=False, avatar_profile=None):
    """Compare the learned residual with the zero-residual PD baseline."""
    episodes = []
    trace = None
    for episode_index, seed in enumerate(seeds):
        env_class = HumanoidRoomEnv
        kwargs = {}
        if reference:
            from .imitation import ReferenceHumanoidEnv
            env_class = ReferenceHumanoidEnv
            kwargs["reference_start"] = reference_start
            if physical_footsteps:
                from .locomotion import FootstepHumanoidEnv
                env_class = FootstepHumanoidEnv
                kwargs["strict_gait_reward"] = strict_gait_reward
        env = env_class(task=task, scene=scene, randomize=randomize,
                        max_episode_seconds=seconds, avatar_profile=avatar_profile, **kwargs)
        assert_policy_body(policy, env)
        observation, _ = env.reset(seed=int(seed))
        episode_trace = None
        if (trace_path and episode_index == 0) or audit_walk:
            episode_trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world", "dt": .02,
                     "controller": "PPO residual" if policy else "PD baseline",
                     "task": task, "seed": int(seed), "scene": scene,
                     "motionReference": "cached T2M BVH" if reference else "analytic stance",
                     "referenceStart": reference and reference_start,
                     "physicalFootstepPrior": physical_footsteps,
                     "gravity": env.model.opt.gravity.tolist(), "frames": [env.snapshot()]}
            episode_trace.update(trace_body_metadata(env))
        rewards, heights, upright, action_norms, seat_samples = [], [], [], [], []
        weight_newtons = float(env.model.body_mass.sum()*abs(env.model.opt.gravity[2]))
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action = (np.zeros(env.action_space.shape, dtype=np.float32) if policy is None
                      else policy.predict(observation, deterministic=True)[0])
            # A real external force, with no position or velocity correction.
            env.data.xfrc_applied[:] = 0
            if push_force and 2 <= env.data.time < 2.2:
                body = getattr(env, "pelvis_body_id", 1)
                env.data.xfrc_applied[body, 0] = push_force
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            heights.append(float(env.data.qpos[2]))
            upright.append(float(info.get("upright", 0)))
            action_norms.append(float(np.linalg.norm(action)))
            if audit_sit:
                seat_samples.append((float(env.data.time), float(info["pelvis_support_force"]),
                                     bool(info["pelvis_support_contact"]), float(info["upright"])))
            if episode_trace is not None:
                episode_trace["frames"].append(env.snapshot())
        episodes.append({"seed": int(seed), "seconds": float(env.data.time),
                         "return": float(sum(rewards)), "fell": bool(terminated),
                         "success": bool(info.get("success", not terminated)),
                         "minRootHeight": min(heights), "meanUpright": float(np.mean(upright)),
                         "meanActionNorm": float(np.mean(action_norms)), "final": info})
        if audit_walk:
            from .gait_audit import audit_gait
            episodes[-1]["gaitQuality"] = audit_gait(episode_trace)
        if audit_sit:
            final_window = [sample for sample in seat_samples if sample[0] >= env.data.time-2-1e-9]
            enough = len(final_window) >= round(2/env.dt)
            minimum_load = min((sample[1] for sample in final_window), default=0.)
            supported = (task == "sit" and enough and not terminated and info["success"]
                         and all(sample[2] and sample[3] > .8 for sample in final_window)
                         and minimum_load >= .3*weight_newtons)
            episodes[-1]["seatSupportQuality"] = {
                "verified": bool(supported), "bodyWeightNewtons": weight_newtons,
                "minimumFinal2SecondsPelvisLoadNewtons": minimum_load,
                "minimumLoadFraction": .3, "sustainedSeconds": 2.,
                "scope": "actual pelvis support on semantic seat near the prepared chair"}
        if episode_index == 0:
            trace = episode_trace
        env.close()
    result = {"controller": "PPO residual" if policy else "PD baseline", "task": task,
              "episodeCount": len(episodes), "pushForceN": push_force,
              "randomized": randomize, "episodes": episodes,
              "motionReference": "cached T2M BVH" if reference else "analytic stance",
              "referenceStart": reference and reference_start,
              "physicalFootstepPrior": physical_footsteps,
              "survivalRate": sum(not e["fell"] for e in episodes) / len(episodes),
              "successRate": sum(e["success"] for e in episodes) / len(episodes),
              "meanReturn": float(np.mean([e["return"] for e in episodes]))}
    if avatar_profile is not None:
        result.update(avatarProfileSha256=env.avatar_profile_sha256,
                      bodyModelSha256=env.body_model_sha256)
    if audit_walk:
        result["walkingVerifiedRate"] = sum(e["gaitQuality"]["walkingVerified"] for e in episodes)/len(episodes)
    if audit_sit:
        result["seatSupportVerifiedRate"] = sum(e["seatSupportQuality"]["verified"] for e in episodes)/len(episodes)
    if trace_path:
        trace["evaluation"] = result
        dump_json(trace_path, trace)
    return result


def train(args):
    # Avoid inheriting a large BLAS/thread pool in each simulator process.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    import mujoco
    import gymnasium
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads(Path(args.scene).read_text()) if args.scene else None
    from .avatar_profile import load_avatar_profile
    avatar_profile = load_avatar_profile(getattr(args, "avatar_profile", None))
    contract_env = HumanoidRoomEnv(task=args.task, scene=scene, avatar_profile=avatar_profile)
    source_names = ["humanoid.py", "environment.py", "training.py"]
    if avatar_profile:
        source_names += ["avatar_profile.py", "body_contract.py"]
    if args.reference:
        source_names += ["imitation.py", "references.py"]
        if args.physical_footsteps:
            source_names.append("locomotion.py")
            if args.strict_gait_reward:
                source_names.append("contact_metrics.py")
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in source_names}

    def factory(index):
        def make():
            env_class = HumanoidRoomEnv
            kwargs = {}
            if args.reference:
                from .imitation import ReferenceHumanoidEnv
                env_class = ReferenceHumanoidEnv
                kwargs["reference_start"] = not args.standing_start
                if args.physical_footsteps:
                    from .locomotion import FootstepHumanoidEnv
                    env_class = FootstepHumanoidEnv
                    kwargs["strict_gait_reward"] = args.strict_gait_reward
                if args.task == "walk" and args.walk_distance_min is not None:
                    kwargs["walking_distance_range"] = (args.walk_distance_min, args.walk_distance_max)
            env = env_class(task=args.task, scene=scene, randomize=True,
                            max_episode_seconds=args.seconds, avatar_profile=avatar_profile, **kwargs)
            if args.push_max:
                env = RandomPushes(env, args.push_max, (args.push_start_min, args.push_start_max))
            env.reset(seed=args.seed + index)
            return env
        return make

    vector_class = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    kwargs = {"start_method": "fork"} if args.envs > 1 else {}
    vector = vector_class([factory(i) for i in range(args.envs)], **kwargs)
    if args.resume:
        policy = PPO.load(args.resume, env=vector, device="cpu")
        try:
            assert_policy_body(policy, contract_env)
        except ValueError:
            if not getattr(args, "allow_body_transfer", False):
                vector.close()
                contract_env.close()
                raise
        if args.learning_rate is not None:
            policy.learning_rate = args.learning_rate
            policy.lr_schedule = lambda progress_remaining: args.learning_rate
        if args.entropy is not None:
            policy.ent_coef = args.entropy
        if args.reset_log_std is not None:
            with torch.no_grad():
                policy.policy.log_std.fill_(args.reset_log_std)
    else:
        policy = PPO("MlpPolicy", vector, seed=args.seed, device="cpu", verbose=1,
                     learning_rate=args.learning_rate or 3e-4, n_steps=512, batch_size=256, n_epochs=5,
                     gamma=.99, gae_lambda=.95, ent_coef=.003 if args.entropy is None else args.entropy,
                     policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                    "log_std_init": -1.5 if args.reset_log_std is None else args.reset_log_std})
    source_profile_sha256 = getattr(policy, "avatar_profile_sha256", None)
    source_body_sha256 = getattr(policy, "body_model_sha256", None)
    bind_policy_body(policy, contract_env)
    contract_env.close()
    start = time.monotonic()
    initial_timesteps = policy.num_timesteps
    try:
        # Periodic durable checkpoints make a long curriculum resumable.
        remaining = args.steps
        while remaining > 0:
            amount = min(remaining, args.checkpoint_every)
            policy.learn(total_timesteps=amount, reset_num_timesteps=False)
            policy.save(out / "policy")
            remaining -= amount
            dump_json(out / "training-progress.json", {
                "task": args.task, "seed": args.seed, "requestedSteps": args.steps,
                "actualTimesteps": policy.num_timesteps, "wallSeconds": time.monotonic()-start,
                "complete": remaining <= 0})
    finally:
        vector.close()
    checkpoint = out / "policy.zip"
    provenance = {"algorithm": "PPO", "role": "bounded joint target residual over torque-limited PD",
                  "task": args.task, "seed": args.seed, "timesteps": policy.num_timesteps,
                  "phaseTimesteps": policy.num_timesteps-initial_timesteps,
                  "initialTimesteps": initial_timesteps, "resumeCheckpoint": args.resume,
                  "trainingPushMaximumN": args.push_max,
                  "trainingPushStartRangeSeconds": [args.push_start_min, args.push_start_max],
                  "motionReference": "cached T2M BVH" if args.reference else "analytic stance",
                  "referenceStart": args.reference and not args.standing_start,
                  "walkingTrainingDistanceRange": [args.walk_distance_min, args.walk_distance_max],
                  "physicalPriorAdaptation": "symmetric swing clearance and measured joint balance" if args.reference and args.task == "walk" else None,
                  "physicalFootstepPrior": args.physical_footsteps,
                  "strictGaitReward": args.strict_gait_reward,
                  "maxEpisodeSeconds": args.seconds,
                  "learningRateOverride": args.learning_rate, "entropyOverride": args.entropy,
                  "initialLogStdOverride": args.reset_log_std,
                  "requestedSteps": args.steps, "envCount": args.envs,
                  "wallSeconds": time.monotonic()-start, "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "versions": {"torch": torch.__version__, "mujoco": mujoco.__version__,
                               "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__},
                  "policyDevice": "cpu", "physicsDevice": "cpu",
                  "avatarProfile": avatar_profile,
                  "avatarProfileSha256": policy.avatar_profile_sha256,
                  "bodyModelSha256": policy.body_model_sha256,
                  "physicalAssumptions": contract_env.physical_assumptions,
                  "bodyTransfer": {"explicitlyAllowed": getattr(args, "allow_body_transfer", False),
                                   "sourceAvatarProfileSha256": source_profile_sha256,
                                   "sourceBodyModelSha256": source_body_sha256},
                  "sourceSha256": source_hashes,
                  "scene": scene, "rootPositionOverwrittenDuringStep": False,
                  "isT2M": False, "neuralWorldModelTrained": False}
    dump_json(out / "provenance.json", provenance)
    # Fixed evaluation seeds are disjoint from all training seeds.
    common = {"task": args.task, "scene": scene, "reference": args.reference,
              "reference_start": not args.standing_start, "seconds": args.seconds,
              "avatar_profile": avatar_profile}
    common["physical_footsteps"] = args.physical_footsteps
    common["strict_gait_reward"] = args.strict_gait_reward
    reports = {"provenance": provenance, "baseline": evaluate(None, **common),
               "policy": evaluate(policy, **common, trace_path=out / "rollout.json"),
               "baselinePush": evaluate(None, **common, push_force=25),
               "policyPush": evaluate(policy, **common, push_force=25)}
    dump_json(out / "evaluation.json", reports)
    print(json.dumps({k: {m: v[m] for m in ("survivalRate", "successRate", "meanReturn")}
                      for k, v in reports.items() if k != "provenance"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["stand", "walk", "sit"], default="stand")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--output", default=".runtime/world-model/stage1")
    parser.add_argument("--scene")
    parser.add_argument("--resume")
    parser.add_argument("--avatar-profile", help="Measured avatar body profile JSON; legacy policies require explicit transfer")
    parser.add_argument("--allow-body-transfer", action="store_true", help="Explicitly initialize fine-tuning from a different physical body")
    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--push-max", type=float, default=0, help="Random training-only pelvis pushes in Newtons")
    parser.add_argument("--push-start-min", type=float, default=1.4)
    parser.add_argument("--push-start-max", type=float, default=3.0)
    parser.add_argument("--reference", action="store_true", help="T2M major-joint reference; root is never commanded")
    parser.add_argument("--standing-start", action="store_true", help="Evaluate and train a transition from an actual standing reset")
    parser.add_argument("--walk-distance-min", type=float)
    parser.add_argument("--walk-distance-max", type=float)
    parser.add_argument("--physical-footsteps", action="store_true", help="Torque-driven two-link footstep prior, with T2M timing/stride")
    parser.add_argument("--strict-gait-reward", action="store_true", help="Actual repeated unloaded swings and low loaded-foot slip, instead of arrival shortcuts")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy", type=float)
    parser.add_argument("--reset-log-std", type=float)
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.envs <= 32 or args.seconds <= 0:
        parser.error("steps/seconds must be positive and envs must be 1..32")
    if not 0 <= args.push_start_min <= args.push_start_max or args.push_max < 0:
        parser.error("push strength and timing must be nonnegative and ordered")
    if ((args.walk_distance_min is None) != (args.walk_distance_max is None) or
            (args.walk_distance_min is not None and not 0 < args.walk_distance_min <= args.walk_distance_max)):
        parser.error("walk distance limits must both be supplied, positive and ordered")
    if args.physical_footsteps and not (args.reference and args.standing_start and args.task == "walk"):
        parser.error("physical footsteps require --task walk --reference --standing-start")
    if args.strict_gait_reward and not args.physical_footsteps:
        parser.error("strict gait reward requires --physical-footsteps")
    train(args)


if __name__ == "__main__":
    main()
