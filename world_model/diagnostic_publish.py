"""Publish teacher-assisted physics diagnostics without promoting learned skills."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from .navigation_evaluation import audit_navigation
from .avatar_profile import avatar_profile_sha256


SOURCE_FILES = {
    "command_navigation_training.py", "command_navigation.py", "command_gait.py",
    "command_motor_limits.py", "navigation_training.py", "navigation.py",
    "navigation_scenes.py", "scenes.py", "sensors.py", "generalization.py",
    "environment.py", "humanoid.py", "avatar_profile.py", "body_contract.py",
    "locomotion.py", "steering.py", "imitation.py", "references.py",
    "contact_metrics.py", "command_motor.py", "command_motor_training.py",
}
VARIANT = "slew-0p5-yaw-0.15"
CONTROLLER = "private geometry teacher + copied learned joint motor; zero diagnostic RL"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_checkpoint_pair(source, private, transfer):
    """Recheck the archived transfer, including weights and unchanged RL counters."""
    import torch
    from stable_baselines3 import PPO

    source_policy = PPO.load(source, device="cpu")
    private_policy = PPO.load(private, device="cpu")
    a, b = source_policy.policy.state_dict(), private_policy.policy.state_dict()
    _require(a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a),
             "Diagnostic transfer changed policy tensors")
    count = transfer["physicalControlTimestepsPreserved"]
    for policy in (source_policy, private_policy):
        _require(policy.num_timesteps == count and policy.command_motor_physical_control_timesteps == count,
                 "Diagnostic transfer changed physical RL counters")
        _require(policy._n_updates == transfer["sourceUpdateEpochsPreserved"],
                 "Diagnostic transfer changed RL update count")
        _require(policy.command_motor_observation_contract == transfer["sourceControlContract"]["observationContract"],
                 "Diagnostic raw actor observation contract differs")
    _require(source_policy.command_motor_control_contract == transfer["sourceControlContract"]
             and private_policy.command_motor_control_contract == transfer["targetControlContract"],
             "Diagnostic checkpoint controller contract differs")
    _require(not private_policy.policy.optimizer.state, "Diagnostic inference populated optimizer state")


def _validate_trace(trace, row, transfer, profile, profile_sha, body_sha, *, collection=False):
    _require(trace.get("version") == 1 and trace.get("task") == "walk"
             and trace.get("geometryScope") == "complete-world" and trace.get("dt") == .02,
             "Diagnostic requires complete native 20 ms walking frames")
    expected_controller = "frozen private combined geometry expert and learned joint motor; zero collection training" if collection else CONTROLLER
    _require(trace.get("controller") == expected_controller, "Diagnostic controller role differs")
    _require(trace.get("avatarProfileSha256") == profile_sha and trace.get("bodyModelSha256") == body_sha
             and trace.get("avatarProfile") == profile and len(trace.get("physicsRestBodies", [])) >= 13,
             "Diagnostic measured avatar/body/rest pose differs")
    nav = trace["navigation"]
    _require(nav.get("evaluationRole") == ("private_teacher_collection" if collection else "private_causal_diagnostic") and nav.get("split") == "train"
             and nav.get("mode") == "obstacle" and nav.get("sceneSeed") == row["sceneSeed"]
             and nav.get("physicsSeed") == row["physicsSeed"]
             and nav.get("checkpointSha256") == transfer["checkpointSha256"]
             and nav.get("sourceCheckpointSha256") == transfer["sourceCheckpointSha256"]
             and nav.get("commandMotorControlContract") == transfer["targetControlContract"],
             "Diagnostic recording belongs to a different source/controller/cohort")
    for key in ("isLearnedRoomNavigator", "highLevelLearnedActorUsed", "usedInLearnedInference"):
        _require(nav.get(key) is False, "Teacher diagnostic cannot claim learned room navigation")
    _require(nav.get("plannerUsedForTraining") is True and nav.get("additionalTrainingTimesteps") == 0,
             "Teacher diagnostic must record zero additional RL")
    teacher = trace["teacher"]
    recording_contract = teacher.get("recordingContract", {})
    _require(teacher.get("diagnosticHeadingSlewRadiansPerSecond") == .5
             and teacher.get("motorHipYawResidualActionLimit") == .15
             and (recording_contract.get("geometryPathInPolicyObservation") is False if collection else teacher.get("pathExposedToMotorObservations") is False)
             and (teacher.get("acceptedForFutureSupervision") is False if collection else teacher.get("acceptedForSupervision") is False)
             and len(teacher.get("route", [])) >= 3,
             "Diagnostic teacher/action role differs")
    frames = trace["frames"]
    _require(len(frames) == row["actual20msControls"]+1 and row["actual2msSubsteps"] == 10*row["actual20msControls"],
             "Diagnostic native control/substep counts differ")
    goal = trace["scene"]["goals"][0]
    body_names = {body["name"] for body in trace["physicsRestBodies"]}
    world = {geom["id"]: geom for geom in frames[0]["geoms"] if geom.get("body") == "world"}
    _require("navigation-furniture" in world and len(world) >= 6, "Diagnostic physical furniture/walls are missing")
    for index, frame in enumerate(frames):
        _require(math.isclose(frame["time"], index*.02, abs_tol=1e-8)
                 and frame.get("avatarProfileSha256") == profile_sha and frame.get("bodyModelSha256") == body_sha
                 and {body["name"] for body in frame.get("bodies", [])} == body_names,
                 "Diagnostic native timestamps/body pose metadata differ")
        _require({geom["id"]: geom for geom in frame["geoms"] if geom.get("body") == "world"} == world,
                 "Diagnostic physical world changed during replay")
        stats = frame["stats"]
        _require(stats["navigation"]["goal"] == goal and stats["navigation"]["scene_sha256"] == row["sceneSha256"]
                 and stats["navigation"].get("proficiencyPassed") is False,
                 "Diagnostic actual final goal or proficiency claim changed")
        if index == 0:
            continue
        motor = stats["command_motor"]
        _require(motor.get("actualFinalGoalFixed") is True and motor.get("rootStateOverwrittenDuringStep") is False
                 and motor.get("isLearnedRoomNavigator") is False and motor.get("teacherDriverUsed") is True
                 and motor.get("physicalControlsExecuted") == 1
                 and motor.get("observedPhysicsSubstepsThisControl") == 10
                 and motor.get("observedPhysicsSubstepsThisEpisode") == index*10
                 and motor.get("controlContract") == transfer["targetControlContract"]
                 and motor.get("privateDiagnosticControllerContract") == transfer["privateControllerContract"],
                 "Diagnostic physical control/substep/teacher invariants differ")
        raw, applied = motor["rawLearnedMotorAction"], motor["appliedMotorAction"]
        _require(len(raw) == len(applied) == 20 and all(raw[j] == applied[j] for j in range(20) if j not in (4, 10))
                 and all(math.isclose(applied[j], max(-.15, min(.15, raw[j])), abs_tol=1e-7) for j in (4, 10)),
                 "Diagnostic applied action adapter changed more than hip yaw")
    quality = audit_navigation(trace)
    _require(quality == trace["navigationQuality"] and quality["navigationVerified"] == row["verified"]
             and quality["failedGates"] == row["failedGates"], "Diagnostic native physics/gait gates differ")
    final = frames[-1]["stats"]
    _require(math.isclose(final["goal_distance"], row["goalDistanceMeters"], abs_tol=1e-12)
             and final["navigation"]["terminal_reason"] == row["terminalReason"]
             and math.isclose(frames[-1]["time"], row["seconds"], abs_tol=1e-8),
             "Diagnostic final outcome differs")
    return quality


def _validate_followup(directory, transfer, profile, profile_sha, body_sha):
    directory = Path(directory)
    collection = _read(directory/"collection.json")
    episodes = collection["episodes"]
    _require(collection.get("checkpointSha256") == transfer["checkpointSha256"]
             and collection.get("sourceParentCheckpointSha256") == transfer["sourceCheckpointSha256"]
             and collection.get("sourceCumulativePhysicalTimesteps") == transfer["physicalControlTimestepsPreserved"]
             and collection.get("sourceSha256") == transfer["sourceSha256"]
             and collection.get("assetSha256") == transfer["assetSha256"], "Diagnostic follow-up source/controller differs")
    _require(collection.get("episodeCount") == len(episodes) == 16
             and len({episode["sceneSeed"] for episode in episodes}) == 16
             and collection.get("additionalTrainingTimesteps") == collection.get("additionalSupervisedUpdates") == 0
             and collection.get("heldOutOrFinalScenesUsed") is False and collection.get("isLearnedRoomNavigator") is False
             and collection.get("usedInLearnedInference") is False and collection.get("plannerUsedForTraining") is True
             and collection.get("sourceFreezeVerified") is True and collection.get("automaticAdoption") is False,
             "Diagnostic follow-up learning/cohort role differs")
    _require(_sha(directory/"collector-source.py") == collection["collectorSourceSha256"], "Diagnostic follow-up collector differs")
    records = []
    for episode in episodes:
        seed = episode["sceneSeed"]
        _require(0 <= seed < 1_000_000 and isinstance(episode.get("qualified"), bool)
                 and episode.get("maximumExternalForce") == 0 and episode.get("actualFinalGoalFixedEvery2ms") is True
                 and episode.get("additionalTrainingTimesteps") == 0,
                 "Diagnostic follow-up episode physics/cohort differs")
        name = episode["trace"]
        _require(Path(name).name == name and name.startswith(f"teacher-{seed}-") and name.endswith(".json"),
                 "Diagnostic follow-up recording path is outside its collection")
        _require(_sha(directory/name) == episode["traceSha256"], "Diagnostic follow-up recording checksum differs")
        row = {**episode, "verified": episode["qualified"], "actual2msSubsteps": episode["actual2msSubstepsObserved"]}
        quality = _validate_trace(_read(directory/name), row, transfer, profile, profile_sha, body_sha, collection=True)
        records.append({"sceneSeed": seed, "traceSha256": episode["traceSha256"], "verified": quality["navigationVerified"],
                        "failedGates": quality["failedGates"], "terminalReason": episode["terminalReason"]})
    count = sum(record["verified"] for record in records)
    _require(collection["qualifiedEpisodes"] == count and collection["actual20msControls"] == sum(episode["actual20msControls"] for episode in episodes)
             and collection["actual2msSubstepsObserved"] == sum(episode["actual2msSubstepsObserved"] for episode in episodes),
             "Diagnostic follow-up gate/counter totals differ")
    return {"trialCount": len(records), "verifiedTrialCount": count, "collectionSha256": _sha(directory/"collection.json"),
            "collectorSourceSha256": collection["collectorSourceSha256"], "additionalTrainingTimesteps": 0,
            "additionalSupervisedUpdates": 0, "isLearnedRoomNavigator": False, "evaluationRole": "additional_train_scenes",
            "sourceCheckpointSha256": transfer["sourceCheckpointSha256"], "privateCheckpointSha256": transfer["checkpointSha256"],
            "records": records}


def publish(*, diagnostic, source_model, output="frontend/public/world-model", dry_run=False, followup=None):
    """Validate all inputs before adding two isolated diagnostic gallery records."""
    diagnostic, source_model, output = Path(diagnostic), Path(source_model), Path(output)
    manifest = _read(output/"manifest.json")
    original = copy.deepcopy(manifest)
    stage = next(stage for stage in manifest["stages"] if stage["stage"] == 2)
    _require(stage.get("proficiencyPassed") is False and manifest["phase2Scope"]["onlineAvatarControl"] is False,
             "Diagnostic must preserve an unfinished offline phase two")
    profile_sha = manifest["avatarProfileSha256"]
    body_sha = manifest["trainingModels"]["walking"]["bodyModelSha256"]
    profile_path = output/"gakucho-avatar-profile.json"
    profile = _read(profile_path)
    _require(avatar_profile_sha256(profile) == profile_sha, "Measured avatar profile checksum differs")
    result = _read(diagnostic/"feasibility.json")
    transfer = _read(diagnostic/VARIANT/"transfer.json")
    source = _read(source_model/"provenance.json")
    _require(_sha(source_model/"policy.zip") == result["sourceCheckpointSha256"] == transfer["sourceCheckpointSha256"]
             == source["checkpointSha256"] and _sha(diagnostic/VARIANT/"policy.zip") == transfer["checkpointSha256"],
             "Diagnostic source/private checkpoint checksum differs")
    _require(source["avatarProfileSha256"] == profile_sha and source["bodyModelSha256"] == body_sha
             and source["physicalControlTimesteps"] == transfer["physicalControlTimestepsPreserved"]
             == result["sourceCumulativePhysicalTimesteps"], "Diagnostic source body/cumulative training count differs")
    for evidence in (result, transfer):
        _require(evidence.get("additionalTrainingTimesteps") == 0 and evidence.get("additionalSupervisedUpdates") == 0
                 and evidence.get("isLearnedRoomNavigator") is False and evidence.get("automaticAdoption") is False,
                 "Diagnostic must remain zero-learning and unadopted")
    _require(result.get("heldOutScenesUsed") is False and result.get("sourceFreezeVerified") is True
             and result.get("bodyAndProfileUnchanged") is True, "Diagnostic source/body/cohort guard missing")
    _require(transfer.get("allActorCriticLogStdTensorsCopiedExactly") is True
             and transfer.get("rawActionMaxAbsError") == transfer.get("criticMaxAbsError") == 0
             and transfer.get("actualTrainResetObservationCount") == 32 and transfer.get("sameBodyVerified") is True,
             "Diagnostic exact same-body policy transfer audit missing")
    hashes = result["sourceSha256"]
    _require(set(hashes) == SOURCE_FILES and transfer["sourceSha256"] == hashes and source["sourceSha256"] == hashes,
             "Diagnostic simulation source archive is incomplete or differs from trained source")
    for name, expected in hashes.items():
        _require(Path(name).name == name and _sha(diagnostic/"simulation-sources"/name) == expected,
                 "Diagnostic archived simulation source checksum differs")
        if name not in ("command_motor.py", "command_motor_training.py"):
            _require(_sha(Path(__file__).parent/name) == expected, "Diagnostic protected physical source checksum differs")
    _require(transfer["assetSha256"] == result["assetSha256"], "Diagnostic asset provenance differs")
    root = Path(__file__).resolve().parent.parent
    for name, expected in result["assetSha256"].items():
        path = (root/name).resolve()
        _require(path.is_relative_to(root) and _sha(path) == expected, "Diagnostic physical/reference/avatar asset checksum differs")
    for file, key in (("probe-source.py", "probeSourceSha256"), ("planner-source.py", "plannerSourceSha256")):
        _require(_sha(diagnostic/file) == result[key] == transfer[key], "Diagnostic private helper checksum differs")
    private = transfer["privateControllerContract"]
    _require(private.get("commandHeadingSlewRadiansPerSecond") == .5 and private.get("motorHipYawResidualActionLimit") == .15
             and private.get("isLearnedRoomNavigator") is False and private.get("additionalTrainingTimesteps") == 0
             and private.get("otherEighteenActionsBitIdentical") is True
             and private.get("rootStateOverwrittenDuringStep") is False and private.get("externalForceApplied") is False
             and private["appliedActionAdapterContract"]["helperSourceSha256"] == hashes["command_motor_limits.py"],
             "Diagnostic private applied-action controller contract differs")
    expected_target = copy.deepcopy(transfer["sourceControlContract"])
    expected_target.update(allowedCommandHeadingSlewRadiansPerSecond=.5, motorHipYawResidualActionLimit=.15)
    _require(transfer["targetControlContract"] == expected_target, "Diagnostic changed undeclared body/torque/reference controls")
    _verify_checkpoint_pair(source_model/"policy.zip", diagnostic/VARIANT/"policy.zip", transfer)
    episodes = result["episodes"]
    _require(len(episodes) == 8 and len({(episode["variant"], episode["sceneSeed"]) for episode in episodes}) == 8
             and all(isinstance(episode["verified"], bool) and 0 <= episode["sceneSeed"] < 1_000_000 for episode in episodes)
             and result["verifiedEpisodes"] == sum(episode["verified"] for episode in episodes),
             "Diagnostic complete training-only factor cohort differs")
    selected = [episode for episode in episodes if episode["variant"] == VARIANT]
    _require(len(selected) == 2 and len({episode["sceneSeed"] for episode in selected}) == 2,
             "Diagnostic gallery must include both selected variant trials")
    records = []
    for row in selected:
        _require(row.get("headingSlewRadiansPerSecond") == .5 and row.get("hipYawResidualActionLimit") == .15
                 and row.get("maximumExternalForce") == 0 and row.get("actualFinalGoalFixedEvery2ms") is True
                 and row.get("additionalTrainingTimesteps") == 0 and row.get("isLearnedRoomNavigator") is False,
                 "Diagnostic episode control/force/learning role differs")
        expected_path = f"{VARIANT}/geometry-{row['sceneSeed']}.json"
        _require(row["recording"] == expected_path, "Diagnostic recording path is outside its variant")
        path = diagnostic/expected_path
        _require(_sha(path) == row["traceSha256"], "Diagnostic native recording checksum differs")
        quality = _validate_trace(_read(path), row, transfer, profile, profile_sha, body_sha)
        filename = f"stage2-diagnostic-teacher-obstacle-{row['sceneSeed']}.json"
        records.append({"title": f"教師の経路補助で家具回避・条件{row['sceneSeed']}（{'通過' if row['verified'] else '未通過'}）",
                        "url": "/world-model/"+filename, "sceneSeed": row["sceneSeed"], "verified": row["verified"],
                        "traceSha256": row["traceSha256"], "seconds": row["seconds"], "failedGates": quality["failedGates"],
                        "filename": filename, "contents": path.read_bytes()})
    public_records = [{key: value for key, value in record.items() if key not in ("filename", "contents")} for record in records]
    manifest["phase2Diagnostics"] = {
        "version": 1, "role": "teacher-assisted physics feasibility; not learned room navigation",
        "sceneScope": "two generated single-furniture physics test rooms; not the photoreal home",
        "teacherRole": "pure geometry expert provides direction commands; learned joint motor controls actual torques",
        "trialCount": len(records), "verifiedTrialCount": sum(record["verified"] for record in records),
        "evaluationRole": "diagnostic_train_scenes", "additionalTrainingTimesteps": 0, "additionalSupervisedUpdates": 0,
        "sourceCumulativePhysicalTimesteps": transfer["physicalControlTimestepsPreserved"],
        "sourceCheckpointSha256": transfer["sourceCheckpointSha256"], "privateCheckpointSha256": transfer["checkpointSha256"],
        "privateControllerContract": private, "avatarProfileSha256": profile_sha, "bodyModelSha256": body_sha,
        "sourceSha256": hashes, "probeSourceSha256": result["probeSourceSha256"], "plannerSourceSha256": result["plannerSourceSha256"],
        "publicationSourceSha256": _sha(__file__), "isLearnedRoomNavigator": False, "proficiencyPassed": False,
        "anyRoomNavigationVerified": False, "onlineAvatarControl": False, "automaticAdoption": False, "records": public_records,
        "validationSourceSha256": {name: _sha(Path(__file__).parent/name) for name in ("navigation_evaluation.py", "gait_audit.py")},
    }
    if followup is not None:
        manifest["phase2Diagnostics"]["followup"] = _validate_followup(followup, transfer, profile, profile_sha, body_sha)
    stage["diagnosticOptions"] = public_records
    protected = copy.deepcopy(manifest)
    protected.pop("phase2Diagnostics", None)
    next(stage for stage in protected["stages"] if stage["stage"] == 2).pop("diagnosticOptions", None)
    before = copy.deepcopy(original)
    before.pop("phase2Diagnostics", None)
    next(stage for stage in before["stages"] if stage["stage"] == 2).pop("diagnosticOptions", None)
    _require(protected == before, "Diagnostic publication changed primary results")
    if not dry_run:
        for record in records:
            (output/record["filename"]).write_bytes(record["contents"])
        pending = output/".manifest.diagnostic.pending.json"
        try:
            pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
            pending.replace(output/"manifest.json")
        finally:
            if pending.exists():
                pending.unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output", default="frontend/public/world-model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--followup")
    args = parser.parse_args()
    manifest = publish(diagnostic=args.diagnostic, source_model=args.source_model, output=args.output, dry_run=args.dry_run, followup=args.followup)
    print(json.dumps({"dryRun": args.dry_run, "diagnosticTrials": manifest["phase2Diagnostics"]["trialCount"],
                      "verifiedTrials": manifest["phase2Diagnostics"]["verifiedTrialCount"], "primaryResultsUnchanged": True}))


if __name__ == "__main__":
    main()
