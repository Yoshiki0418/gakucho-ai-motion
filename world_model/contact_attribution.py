"""Matched physical contact attribution; geometry is an explicit diagnostic oracle.

The geometric controller is never a learned actor input and is not qualified
training data unless the independent full-trajectory audit passes in train rooms.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .body_contract import trace_body_metadata
from .command_navigation_evaluation import append_control_frames
from .navigation_evaluation import audit_navigation
from .scenes import point_box_distance
from .training import dump_json
from .visual_teacher import oracle_route, route_target


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DiagnosticGeometryController:
    """Full-map *evaluation* oracle; no train-only teacher guard is bypassed."""

    def __init__(self, env, *, margin=.5, carrot=.55, pace=-.6):
        self.env, self.carrot, self.pace = env, carrot, pace
        self.route, self.progress = oracle_route(env.scene, margin), 0.

    def action(self):
        body = self.env.unwrapped
        target, self.progress = route_target(self.route, body.data.qpos[:2], self.carrot, self.progress)
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = target-body.data.qpos[:2]
        desired = math.atan2(delta[0], -delta[1])
        return np.asarray([np.clip(_wrapped(desired-heading)/.45, -1., 1.), self.pace], dtype=np.float32)


def actual_body_clearance(body, furniture):
    """Minimum actual collision-geometry surface distance, not root clearance."""
    import mujoco
    obstacle = body.model.geom(furniture["id"]).id
    distances = []
    for geom in range(body.model.ngeom):
        if body.model.geom_bodyid[geom] and (body.model.geom_contype[geom] or body.model.geom_conaffinity[geom]):
            distance = float(mujoco.mj_geomDistance(body.model, body.data, obstacle, geom, 10., None))
            distances.append((distance, body.model.geom(geom).name))
    minimum, name = min(distances)
    return {"actualCollisionGeomSurfaceClearanceMeters": minimum,
            "closestAvatarCollisionGeom": name,
            "rootPointClearanceMeters": point_box_distance(body.data.qpos[:2], furniture)}


def _measurement(frame, furniture):
    stats = frame["stats"]
    motor = stats.get("command_motor", {})
    reward = motor.get("trackingReward", {})
    physical = reward.get("measurements", {})
    command = motor.get("nextCommand", {})
    return {"timeSeconds": frame["time"],
        "rootPosition": stats["root_position"], "rootVelocity": stats["root_velocity"],
        "rootPointClearanceMeters": point_box_distance(stats["root_position"], furniture),
        "requestedHeadingOffsetRadians": command.get("requestedHeadingOffsetRadians"),
        "nextEffectiveHeadingOffsetRadians": command.get("effectiveHeadingOffsetRadians"),
        "consumedCommandRelativeHeadingRadians": reward.get("consumedCommandRelativeHeadingRadians"),
        "actualYawRateRadiansPerSecond": reward.get("actualYawRateRadiansPerSecond"),
        "desiredYawRateRadiansPerSecond": reward.get("desiredYawRateRadiansPerSecond"),
        "actualRootHeadingRadians": physical.get("rootHeadingRadians"),
        "consumedCommandWorldXYZ": physical.get("consumedCommandWorldXYZ"),
        "consumedCommandDistanceMeters": physical.get("commandDistanceMeters"),
        "collisionOccurred": stats.get("collision_occurred", False),
        "contactForceNewtons": stats.get("collision_force_newtons", 0.)}


def summarize_measurements(samples, collision_time):
    active = [x for x in samples if x["actualYawRateRadiansPerSecond"] is not None]
    precontact = [x for x in active if collision_time is None or x["timeSeconds"] <= collision_time]
    turning = [x for x in precontact if abs(x["consumedCommandRelativeHeadingRadians"]) >= .2]
    def first(predicate):
        return next((x["timeSeconds"] for x in active if predicate(x)), None)
    return {"firstRequestedTurnOverPoint2RadiansSeconds": first(lambda x: abs(x["requestedHeadingOffsetRadians"]) >= .2),
        "firstConsumedTurnOverPoint2RadiansSeconds": first(lambda x: abs(x["consumedCommandRelativeHeadingRadians"]) >= .2),
        "firstRootClearanceBelow1MeterSeconds": first(lambda x: x["rootPointClearanceMeters"] < 1.),
        "firstRootClearanceBelowPoint5MetersSeconds": first(lambda x: x["rootPointClearanceMeters"] < .5),
        "firstPhysicalCollisionSeconds": collision_time,
        "precontactTurningControlCount": len(turning),
        "precontactTurningMeanRequestedYawRate": float(np.mean([x["desiredYawRateRadiansPerSecond"] for x in turning])) if turning else None,
        "precontactTurningMeanActualYawRate": float(np.mean([x["actualYawRateRadiansPerSecond"] for x in turning])) if turning else None,
        "precontactTurningMeanYawTrackingAbsoluteError": float(np.mean([abs(x["actualYawRateRadiansPerSecond"]-x["desiredYawRateRadiansPerSecond"]) for x in turning])) if turning else None,
        "precontactTurningWrongSignFraction": float(np.mean([x["actualYawRateRadiansPerSecond"]*x["desiredYawRateRadiansPerSecond"] < 0 for x in turning])) if turning else None}


def rollout_controller(controller, *, scene_seed, physics_seed, motor_checkpoint, avatar_profile,
                       output, policy=None, seconds=30., difficulty_range=(0., .5),
                       margin=.5, carrot=.55, pace=-.6, split="heldout", collect_observations=False,
                       env_factory=None):
    """Retain every native frame and optional raw train-only depth/state labels."""
    from .generalization import validate_scene_seed
    from .visual_navigation import VisualNavigationEnv, assert_visual_policy
    validate_scene_seed(scene_seed, split)
    if controller not in ("visual", "geometry") or (controller == "visual" and policy is None):
        raise ValueError("controller must be visual with a policy, or geometry")
    if collect_observations and split != "train":
        raise ValueError("held-out diagnostic observations cannot become training labels")
    output = Path(output)
    if (output/"episode.json").exists():
        raise ValueError("preserve completed contact diagnostic; use a new directory")
    factory = VisualNavigationEnv if env_factory is None else env_factory
    env = factory(motor_checkpoint=motor_checkpoint, record_trajectory=True,
        mode="obstacle", split=split, base_seed=scene_seed, difficulty_range=difficulty_range,
        randomize=True, max_episode_seconds=seconds, avatar_profile=avatar_profile)
    try:
        observation, _ = env.reset(seed=physics_seed, options={"scene_seed": scene_seed, "physics_seed": physics_seed})
        body, scene = env.unwrapped, copy.deepcopy(env.scene)
        if controller == "visual":
            assert_visual_policy(policy, env)
        oracle = DiagnosticGeometryController(env, margin=margin, carrot=carrot, pace=pace) if controller == "geometry" else None
        initial = {"qpos": body.data.qpos.copy().tolist(), "qvel": body.data.qvel.copy().tolist(),
            "qposSha256": hashlib.sha256(body.data.qpos.tobytes()).hexdigest(),
            "qvelSha256": hashlib.sha256(body.data.qvel.tobytes()).hexdigest(),
            "sceneSha256": hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        furniture = next(g for g in scene["geoms"] if g.get("entityId") == "navigation-furniture")
        trace = {"version": 1, "sizeConvention": "mujoco", "geometryScope": "complete-world",
            "dt": float(body.dt), "task": "walk", "scene": scene, "gravity": body.model.opt.gravity.tolist(),
            "motionReference": "cached T2M BVH", "physicalFootstepPrior": True,
            "controller": controller+" contact attribution with immutable 5aa motor",
            "navigation": {"mode": "obstacle", "sceneSeed": scene_seed, "physicsSeed": physics_seed,
                "split": split, "evaluationRole": "development-diagnostic" if split == "heldout" else "teacher-qualification",
                "privilegedTeacherUsed": oracle is not None, "fixedWaypointsSupplied": oracle is not None,
                "rootPositionOverwrittenDuringStep": False, "motorCheckpointSha256": _sha(motor_checkpoint),
                "geometryScope": "full actual scene for diagnostic oracle only" if oracle else "depth/state actor only"},
            "initialPhysicalState": initial, "frames": [env.snapshot()]}
        trace.update(trace_body_metadata(body))
        done = truncated = False
        samples, decisions, clearance, data_depth, data_state, data_action = [], [], [], [], [], []
        controls = substeps = 0
        while not (done or truncated):
            action = oracle.action() if oracle else policy.predict(observation, deterministic=True)[0]
            clearance.append({"timeSeconds": float(body.data.time), **actual_body_clearance(body, furniture)})
            decisions.append({"timeSeconds": float(body.data.time), "action": action.tolist()})
            if collect_observations:
                data_depth.append(observation["depth"].copy()); data_state.append(observation["state"].copy()); data_action.append(action.copy())
            observation, _, done, truncated, info = env.step(action)
            count, native = info["command_control_steps"], info["command_physics_substeps"]
            if native != count*10:
                raise RuntimeError("contact attribution missed native solver substeps")
            append_control_frames(trace, env.last_control_frames, expected_count=count, control_dt=body.dt)
            samples.extend(_measurement(frame, furniture) for frame in env.last_control_frames)
            controls += count; substeps += native
        env.assert_frozen_motor(check_checkpoint=True)
        quality = audit_navigation(trace)
        metrics = trace["frames"][-1]["stats"]["navigation"]["metrics"]
        trace["navigationQuality"] = quality
        output.mkdir(parents=True, exist_ok=True)
        dump_json(output/"trace.json", trace)
        dump_json(output/"measurements.json", {"native20msMeasurements": samples, "decision100msActions": decisions,
            "decision100msActualCollisionSurfaceClearance": clearance,
            "clearanceScope": "mj_geomDistance actual avatar collision geoms vs furniture every 100ms; sticky contacts still observed every 2ms"})
        result = {"version": 1, "controller": controller, "sceneSeed": scene_seed, "physicsSeed": physics_seed,
            "split": split, "initialPhysicalState": initial, "seconds": float(body.data.time),
            "quality": quality, "physicalControls": controls, "physical2msSubsteps": substeps,
            "finalGoalDistance": trace["frames"][-1]["stats"]["goal_distance"],
            "contactGeometryIds": metrics["obstacle_contact_geom_ids"],
            "minimumActualBodySurfaceClearanceMetersAt10Hz": min(x["actualCollisionGeomSurfaceClearanceMeters"] for x in clearance),
            "attribution": summarize_measurements(samples, metrics["first_collision_time_seconds"]),
            "motorCheckpointSha256": _sha(motor_checkpoint), "sourceSha256": _sha(__file__),
            "explicitEnvironmentFactory": factory.__module__+"."+factory.__qualname__,
            "traceSha256": _sha(output/"trace.json"), "measurementsSha256": _sha(output/"measurements.json"),
            "privilegedTeacherUsed": oracle is not None, "qualifiedTrainingData": bool(split == "train" and quality["navigationVerified"])}
        if collect_observations:
            np.savez_compressed(output/"observations.npz", depth=np.stack(data_depth), state=np.stack(data_state), action=np.stack(data_action))
            result["observationSha256"] = _sha(output/"observations.npz")
            result["observationCount"] = len(data_action)
        dump_json(output/"episode.json", result)
        print(json.dumps({"controller": controller, "sceneSeed": scene_seed, "navigationVerified": quality["navigationVerified"],
            "failedGates": quality["failedGates"], "attribution": result["attribution"]}), flush=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--motor-policy", required=True)
    parser.add_argument("--avatar-profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="1180100,1180102,1180104,1180106")
    parser.add_argument("--carrot", type=float, default=.55)
    parser.add_argument("--margin", type=float, default=.5)
    args = parser.parse_args()
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    policy = PPO.load(args.policy, device="cpu")
    episodes = []
    for seed in map(int, args.seeds.split(",")):
        pair = []
        for controller in ("visual", "geometry"):
            pair.append(rollout_controller(controller, scene_seed=seed, physics_seed=seed+2_000_000,
                motor_checkpoint=args.motor_policy, avatar_profile=args.avatar_profile,
                output=Path(args.output)/f"{controller}-{seed}", policy=policy,
                margin=args.margin, carrot=args.carrot))
        if pair[0]["initialPhysicalState"] != pair[1]["initialPhysicalState"]:
            raise RuntimeError("paired controllers received different actual initial physics")
        episodes.extend(pair)
        dump_json(Path(args.output)/"comparison.json", {"episodes": episodes, "allPairInitialPhysicsExact": True,
            "heldoutDiagnosticNotTrainingData": True, "policySha256": _sha(args.policy)})


if __name__ == "__main__":
    main()
