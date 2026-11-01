"""Synthetic gallery fixtures exercise evidence guards, never physical proficiency."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from . import diagnostic_publish as publisher
from .avatar_profile import avatar_profile_sha256
from .navigation_evaluation import audit_navigation
from .test_gait_audit import _fixture


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def gallery(tmp_path, monkeypatch):
    output, diagnostic, source = (tmp_path/name for name in ("public", "diagnostic", "source"))
    output.mkdir()
    profile = {"version": 1, "fixture": "synthetic publisher fixture"}
    profile_sha = avatar_profile_sha256(profile)
    dump(output/"gakucho-avatar-profile.json", profile)
    manifest = {"avatarProfileSha256": profile_sha, "trainingModels": {"walking": {"bodyModelSha256": "body"}},
        "failureExamples": {"examples": [{"seed": 1}]}, "gates": {"walking": "passed"},
        "phase2CommandTrainingModels": {"turn": {"checkpointSha256": "existing-model"}},
        "phase2Scope": {"onlineAvatarControl": False, "allPhase2TasksPassed": False, "publicationSourceSha256": "unchanged"},
        "stages": [{"stage": 1, "gates": {"walking": "passed"}},
                   {"stage": 2, "proficiencyPassed": False, "gates": {"turning": "passed", "singleFurnitureAvoidance": "failed"},
                    "traceUrl": "/world-model/old.json", "sceneUrl": "/world-model/old-scene.json",
                    "traceOptions": [{"title": "existing", "url": "/world-model/old.json"}],
                    "metrics": {"turningEvaluation": {"episodeCount": 32, "navigationVerifiedRate": 29/32},
                                "commandNavigationEvaluations": {"turn": {"episodeCount": 8, "navigationVerifiedRate": 5/8},
                                                                 "obstacle": {"episodeCount": 8, "navigationVerifiedRate": 0}}}},
                   {"stage": 3}, {"stage": 4}]}
    dump(output/"manifest.json", manifest)
    (output/"old-scene.json").write_bytes(b"preserve existing scene")
    source.mkdir()
    (source/"policy.zip").write_bytes(b"synthetic source checkpoint")
    variant = diagnostic/publisher.VARIANT
    variant.mkdir(parents=True)
    (variant/"policy.zip").write_bytes(b"synthetic exact transfer checkpoint")
    hashes = {}
    for name in publisher.SOURCE_FILES:
        raw = (Path(publisher.__file__).parent/name).read_bytes()
        (diagnostic/"simulation-sources").mkdir(exist_ok=True)
        (diagnostic/"simulation-sources"/name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    for name in ("probe-source.py", "planner-source.py"):
        (diagnostic/name).write_text("# synthetic publisher fixture\n")
    source_sha, clone_sha = publisher._sha(source/"policy.zip"), publisher._sha(variant/"policy.zip")
    source_contract = {"observationContract": {"observationCount": 110, "actionCount": 20}, "bodyModelSha256": "body"}
    target = {**source_contract, "allowedCommandHeadingSlewRadiansPerSecond": .5, "motorHipYawResidualActionLimit": .15}
    private = {"commandHeadingSlewRadiansPerSecond": .5, "motorHipYawResidualActionLimit": .15,
               "isLearnedRoomNavigator": False, "additionalTrainingTimesteps": 0,
               "otherEighteenActionsBitIdentical": True, "rootStateOverwrittenDuringStep": False,
               "externalForceApplied": False, "appliedActionAdapterContract": {"helperSourceSha256": hashes["command_motor_limits.py"]}}
    assets = {"frontend/public/world-model/gakucho-avatar-profile.json": publisher._sha(Path(publisher.__file__).parent.parent/"frontend/public/world-model/gakucho-avatar-profile.json")}
    common = {"sourceCheckpointSha256": source_sha, "sourceSha256": hashes, "assetSha256": assets,
              "probeSourceSha256": publisher._sha(diagnostic/"probe-source.py"), "plannerSourceSha256": publisher._sha(diagnostic/"planner-source.py"),
              "additionalTrainingTimesteps": 0, "additionalSupervisedUpdates": 0, "isLearnedRoomNavigator": False, "automaticAdoption": False}
    transfer = {**common, "checkpointSha256": clone_sha, "sourceControlContract": source_contract, "targetControlContract": target,
                "privateControllerContract": private, "physicalControlTimestepsPreserved": 123,
                "allActorCriticLogStdTensorsCopiedExactly": True, "rawActionMaxAbsError": 0., "criticMaxAbsError": 0.,
                "actualTrainResetObservationCount": 32, "sameBodyVerified": True, "sourceUpdateEpochsPreserved": 1}
    dump(variant/"transfer.json", transfer)
    dump(source/"provenance.json", {"checkpointSha256": source_sha, "avatarProfileSha256": profile_sha, "sourceSha256": hashes,
                                  "bodyModelSha256": "body", "physicalControlTimesteps": 123})
    rows = []
    for seed, failed in ((101, True), (102, False)):
        trace = _fixture()
        goal = {"id": "goal", "position": [0, -1, 0]}
        world = [{"id": name, "type": "box", "position": [0, -.5, .5], "size": [.1, .1, 1], "body": "world"}
                 for name in ("floor", "wall-west", "wall-east", "wall-north", "wall-south", "navigation-furniture")]
        world[0].update(position=[0, 0, -.04], size=[3, 3, .04])
        trace.update(version=1, geometryScope="complete-world", controller=publisher.CONTROLLER,
                     avatarProfileSha256=profile_sha, avatarProfile=profile, bodyModelSha256="body",
                     physicsRestBodies=[{"name": f"body{j}", "position": [0, 0, 0], "quaternion": [1, 0, 0, 0]} for j in range(13)],
                     scene={"spawn": [0, 0, 0], "goals": [goal], "metadata": {"directRouteBlocked": True},
                            "geoms": [{**world[-1], "entityId": "navigation-furniture", "yaw": 0}]},
                     navigation={"mode": "obstacle", "sceneSeed": seed, "physicsSeed": seed+10, "split": "train",
                                 "evaluationRole": "private_causal_diagnostic", "checkpointSha256": clone_sha, "sourceCheckpointSha256": source_sha,
                                 "commandMotorControlContract": target, "isLearnedRoomNavigator": False, "highLevelLearnedActorUsed": False,
                                 "usedInLearnedInference": False, "plannerUsedForTraining": True, "additionalTrainingTimesteps": 0},
                     teacher={"diagnosticHeadingSlewRadiansPerSecond": .5, "motorHipYawResidualActionLimit": .15,
                              "pathExposedToMotorObservations": False, "acceptedForSupervision": False, "route": [[0, 0], [-1, -.5], [0, -1]]})
        for index, frame in enumerate(trace["frames"]):
            frame.update(avatarProfileSha256=profile_sha, bodyModelSha256="body", bodies=copy.deepcopy(trace["physicsRestBodies"]))
            frame["geoms"].extend(copy.deepcopy(world))
            stats = frame["stats"]
            loads = [stats[side+"_foot_force"] for side in ("left", "right")]
            stats.update(root_velocity=[0, 0, 0], navigation_floor_support=[True, True], navigation_success_seconds=.5,
                         collision_occurred=failed and index == 20,
                         navigation={"goal": goal, "scene_sha256": f"scene{seed}", "proficiencyPassed": False,
                                     "terminal_reason": "collision" if failed else "success",
                                     "metrics": {"floor_support_forces_newtons": loads, "goal_floor_matches": True, "root_height_error_m": .01}})
            if index >= len(trace["frames"])-26:
                stats["goal_distance"] = .1
                stats["left_foot_force"] = stats["right_foot_force"] = 300.
                stats["navigation"]["metrics"]["floor_support_forces_newtons"] = [300., 300.]
            if index:
                stats["command_motor"] = {"actualFinalGoalFixed": True, "rootStateOverwrittenDuringStep": False,
                    "isLearnedRoomNavigator": False, "teacherDriverUsed": True, "physicalControlsExecuted": 1,
                    "observedPhysicsSubstepsThisControl": 10, "observedPhysicsSubstepsThisEpisode": index*10,
                    "controlContract": target, "privateDiagnosticControllerContract": private,
                    "rawLearnedMotorAction": [0.]*20, "appliedMotorAction": [0.]*20}
        trace["navigationQuality"] = audit_navigation(trace)
        assert trace["navigationQuality"]["navigationVerified"] is not failed, trace["navigationQuality"]["failedGates"]
        path = variant/f"geometry-{seed}.json"
        dump(path, trace)
        rows.append({"variant": publisher.VARIANT, "sceneSeed": seed, "physicsSeed": seed+10, "sceneSha256": f"scene{seed}",
                     "verified": not failed, "failedGates": trace["navigationQuality"]["failedGates"],
                     "actual20msControls": len(trace["frames"])-1, "actual2msSubsteps": 10*(len(trace["frames"])-1),
                     "goalDistanceMeters": .1, "seconds": trace["frames"][-1]["time"], "terminalReason": "collision" if failed else "success",
                     "headingSlewRadiansPerSecond": .5, "hipYawResidualActionLimit": .15, "maximumExternalForce": 0,
                     "actualFinalGoalFixedEvery2ms": True, "additionalTrainingTimesteps": 0, "isLearnedRoomNavigator": False,
                     "recording": f"{publisher.VARIANT}/geometry-{seed}.json", "traceSha256": publisher._sha(path)})
    for variant_name in ("a", "b", "c"):
        rows.extend({**copy.deepcopy(row), "variant": variant_name, "verified": False} for row in rows[:2])
    result = {**common, "sourceCumulativePhysicalTimesteps": 123, "sourceFreezeVerified": True,
              "bodyAndProfileUnchanged": True, "heldOutScenesUsed": False, "verifiedEpisodes": 1, "episodes": rows}
    dump(diagnostic/"feasibility.json", result)
    monkeypatch.setattr(publisher, "_verify_checkpoint_pair", lambda *args: None)
    return output, diagnostic, source, manifest


def publish(gallery, **kwargs):
    output, diagnostic, source, _ = gallery
    return publisher.publish(diagnostic=diagnostic, source_model=source, output=output, **kwargs)


def test_native_records_are_byte_identical_and_primary_results_are_preserved(gallery):
    output, diagnostic, _, before = gallery
    first = publish(gallery)
    assert publish(gallery) == first
    restored = copy.deepcopy(first)
    restored.pop("phase2Diagnostics")
    restored["stages"][1].pop("diagnosticOptions")
    assert restored == before
    assert first["phase2Diagnostics"]["verifiedTrialCount"] == 1
    assert first["phase2Diagnostics"]["proficiencyPassed"] is False
    for item in first["phase2Diagnostics"]["records"]:
        assert (output/Path(item["url"]).name).read_bytes() == (diagnostic/publisher.VARIANT/f"geometry-{item['sceneSeed']}.json").read_bytes()
    assert (output/"old-scene.json").read_bytes() == b"preserve existing scene"


def test_dry_run_validates_without_writing_any_public_files(gallery):
    output, _, _, _ = gallery
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert publish(gallery, dry_run=True)["phase2Diagnostics"]["trialCount"] == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def followup_fixture(gallery):
    _, diagnostic, _, _ = gallery
    directory = diagnostic/"followup"
    directory.mkdir()
    (directory/"collector-source.py").write_text("# synthetic collection fixture\n")
    transfer = publisher._read(diagnostic/publisher.VARIANT/"transfer.json")
    initial = publisher._read(diagnostic/"feasibility.json")["episodes"][0]
    template = publisher._read(diagnostic/initial["recording"])
    rows = []
    for seed in range(201, 217):
        trace = copy.deepcopy(template)
        trace["controller"] = "frozen private combined geometry expert and learned joint motor; zero collection training"
        trace["navigation"].update(evaluationRole="private_teacher_collection", sceneSeed=seed, physicsSeed=seed+10)
        trace["teacher"].update(recordingContract={"geometryPathInPolicyObservation": False}, acceptedForFutureSupervision=False)
        trace["navigationQuality"] = audit_navigation(trace)
        name = f"teacher-{seed}-excluded.json"
        dump(directory/name, trace)
        rows.append({**copy.deepcopy(initial), "sceneSeed": seed, "physicsSeed": seed+10,
                     "qualified": False, "trace": name, "traceSha256": publisher._sha(directory/name),
                     "actual2msSubstepsObserved": initial["actual2msSubsteps"]})
    collection = {"episodes": rows, "episodeCount": 16, "qualifiedEpisodes": 0,
                  "actual20msControls": sum(row["actual20msControls"] for row in rows),
                  "actual2msSubstepsObserved": sum(row["actual2msSubstepsObserved"] for row in rows),
                  "checkpointSha256": transfer["checkpointSha256"], "sourceParentCheckpointSha256": transfer["sourceCheckpointSha256"],
                  "sourceCumulativePhysicalTimesteps": transfer["physicalControlTimestepsPreserved"],
                  "sourceSha256": transfer["sourceSha256"], "assetSha256": transfer["assetSha256"],
                  "additionalTrainingTimesteps": 0, "additionalSupervisedUpdates": 0, "heldOutOrFinalScenesUsed": False,
                  "isLearnedRoomNavigator": False, "usedInLearnedInference": False, "plannerUsedForTraining": True,
                  "sourceFreezeVerified": True, "automaticAdoption": False,
                  "collectorSourceSha256": publisher._sha(directory/"collector-source.py")}
    dump(directory/"collection.json", collection)
    return directory


def test_further_sixteen_failed_conditions_remain_separate_from_initial_success(gallery):
    result = publish(gallery, followup=followup_fixture(gallery), dry_run=True)
    summary = result["phase2Diagnostics"]
    assert summary["verifiedTrialCount"] == 1
    assert summary["followup"]["trialCount"] == 16
    assert summary["followup"]["verifiedTrialCount"] == 0
    assert summary["followup"]["additionalTrainingTimesteps"] == 0
    assert summary["proficiencyPassed"] is False


def test_forged_followup_qualification_is_recomputed_from_native_frames(gallery):
    directory = followup_fixture(gallery)
    collection = publisher._read(directory/"collection.json")
    collection["qualifiedEpisodes"] = 1
    collection["episodes"][0]["qualified"] = True
    dump(directory/"collection.json", collection)
    with pytest.raises(ValueError, match="gait gates differ"):
        publish(gallery, followup=directory)


def test_historical_reward_sources_are_verified_in_archive_after_new_reward_work(gallery, monkeypatch):
    original_sha = publisher._sha
    current_motor = Path(publisher.__file__).parent/"command_motor.py"
    monkeypatch.setattr(publisher, "_sha", lambda path: "new reward source" if Path(path) == current_motor else original_sha(path))
    assert publish(gallery, dry_run=True)["phase2Diagnostics"]["verifiedTrialCount"] == 1


def test_model_provenance_must_match_the_complete_historical_source_archive(gallery):
    _, _, source, _ = gallery
    provenance = publisher._read(source/"provenance.json")
    provenance["sourceSha256"]["command_motor.py"] = "wrong archived reward source"
    dump(source/"provenance.json", provenance)
    with pytest.raises(ValueError, match="differs from trained source"):
        publish(gallery)


@pytest.mark.parametrize("mutation", ["trace-bytes", "gates", "final-goal", "substeps", "other-joint", "teacher-role", "source-archive", "clone-checkpoint", "rl-count", "hidden-failure"])
def test_corrupted_or_promoted_evidence_is_rejected_before_publication(gallery, mutation):
    output, diagnostic, _, _ = gallery
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    result = publisher._read(diagnostic/"feasibility.json")
    row = result["episodes"][1]
    path = diagnostic/row["recording"]
    trace = publisher._read(path)
    if mutation == "trace-bytes":
        path.write_text(path.read_text()+" ")
    elif mutation == "source-archive":
        (diagnostic/"simulation-sources/navigation.py").write_text("changed")
    elif mutation == "clone-checkpoint":
        (diagnostic/publisher.VARIANT/"policy.zip").write_bytes(b"changed")
    elif mutation == "rl-count":
        result["additionalTrainingTimesteps"] = 100
    elif mutation == "hidden-failure":
        result["episodes"].pop(0)
    else:
        if mutation == "gates": trace["navigationQuality"]["gates"]["stopped"] = False
        elif mutation == "final-goal": trace["frames"][10]["stats"]["navigation"]["goal"] = {"position": [99, 99, 99]}
        elif mutation == "substeps": trace["frames"][10]["stats"]["command_motor"]["observedPhysicsSubstepsThisControl"] = 9
        elif mutation == "other-joint": trace["frames"][10]["stats"]["command_motor"]["appliedMotorAction"][0] = .1
        elif mutation == "teacher-role": trace["navigation"]["isLearnedRoomNavigator"] = True
        dump(path, trace)
        row["traceSha256"] = publisher._sha(path)
    dump(diagnostic/"feasibility.json", result)
    with pytest.raises(ValueError): publish(gallery)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
