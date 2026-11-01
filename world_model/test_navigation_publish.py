"""Comparison publication must preserve accepted skills and reject changed evidence."""
import hashlib
import json

import pytest

from .navigation_publish import publish


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def publication(tmp_path):
    output = tmp_path/"public"
    output.mkdir()
    manifest = {"avatarProfileSha256": "profile", "gates": {"walking": "passed"},
        "trainingModels": {"walking": {"bodyModelSha256": "body", "checkpointSha256": "accepted-walk"}},
        "failureExamples": {"evaluationExcluded": True, "examples": [{"seed": 1}]},
        "phase2TrainingModels": {"turn": {"checkpointSha256": "accepted-turn"}},
        "stages": [{"stage": 1, "proficiencyPassed": True, "gates": {"walking": "passed"},
                    "metrics": {"walkingEvaluation": {"count": 32, "completionVerifiedRate": 31/32}}},
            {"stage": 2, "gates": {"turning": "passed", "singleFurnitureAvoidance": "failed", "multiFurnitureNavigation": "pending"},
             "proficiencyPassed": False, "status": "learning", "traceUrl": "/world-model/stage2-turn-accepted.json",
             "sceneUrl": "/world-model/stage2-navigation-scene.json",
             "traceOptions": [{"title": "Accepted turn", "url": "/world-model/stage2-turn-accepted.json"},
                              {"title": "Old command example", "url": "/world-model/stage2-command-turn-old.json"}],
             "metrics": {"turningEvaluation": {"episodeCount": 32, "navigationVerifiedRate": 29/32},
                         "navigationEvaluation": {"episodeCount": 8, "navigationVerifiedRate": 0}}},
            {"stage": 3, "proficiencyPassed": False}, {"stage": 4, "proficiencyPassed": False}]}
    dump(output/"manifest.json", manifest)
    (output/"stage2-navigation-scene.json").write_bytes(b'{"existing":true}')

    def evaluation(mode="turn", *, command=True, count=8, passed=0, seed=1_160_000):
        directory = tmp_path/f"evaluation-{mode}-{'command' if command else 'joint'}"
        model = tmp_path/f"model-{mode}-{'command' if command else 'joint'}"
        directory.mkdir(exist_ok=True)
        model.mkdir(exist_ok=True)
        (model/"policy.zip").write_bytes(b"evaluated-policy")
        motor = tmp_path/"motor.zip"
        motor.write_bytes(b"frozen-motor")
        sha = hashlib.sha256((model/"policy.zip").read_bytes()).hexdigest()
        motor_sha = hashlib.sha256(motor.read_bytes()).hexdigest()
        contract = {"motorCheckpointSha256": motor_sha, "motorWeightsFrozen": True,
                    "motorInputCount": 110, "motorJointActionCount": 20}
        common = {"mode": mode, "checkpointSha256": sha, "avatarProfileSha256": "profile", "bodyModelSha256": "body"}
        interface = {"motorCheckpointSha256": motor_sha, "commandControlContract": contract,
            "observationCount": 112, "actionCount": 2, "motorObservationCount": 110, "motorActionCount": 20}
        provenance = {**common, **interface, "motorFrozen": True, "motorCheckpoint": str(motor),
                      "actualDecisionTimesteps": 32768, "physicalControlTimesteps": 163376}
        dump(model/"provenance.json", provenance)
        success = passed > 0
        filename = f"stage2-{'command-' if command else ''}{mode}-{seed}-{'verified' if success else 'failed'}.json"
        trace = {"avatarProfileSha256": "profile", "bodyModelSha256": "body", "scene": {"mode": mode},
                 "navigation": {**common, **interface, "sceneSeed": seed, "motorFrozen": True},
                 "navigationQuality": {"navigationVerified": success}}
        dump(directory/filename, trace)
        result = {**common, **interface, "episodeCount": count,
            "navigationVerifiedRate": passed/count, "evaluationRole": "final" if seed >= 1_800_000 else "development",
            "proficiencyPassed": seed >= 1_800_000 and count >= 32 and passed/count >= .9,
            "episodes": [{"sceneSeed": seed+index, "navigationVerified": index < passed} for index in range(count)],
            "recordings": [{"filename": filename, "sceneSeed": seed,
                            "traceSha256": hashlib.sha256((directory/filename).read_bytes()).hexdigest()}]}
        dump(directory/"evaluation.json", result)
        return directory, model, result, provenance
    return output, manifest, evaluation


def test_failed_development_is_idempotent_and_preserves_accepted_results(publication):
    output, before, evaluation = publication
    directory, model, result, provenance = evaluation()
    first = publish(command_turning=directory, command_turning_model=model, output=output)
    assert publish(command_turning=directory, command_turning_model=model, output=output) == first
    for key in ("gates", "trainingModels", "failureExamples", "phase2TrainingModels"):
        assert first[key] == before[key]
    assert [stage for stage in first["stages"] if stage["stage"] != 2] == [stage for stage in before["stages"] if stage["stage"] != 2]
    stage = first["stages"][1]
    assert stage["gates"] == before["stages"][1]["gates"]
    assert stage["metrics"]["turningEvaluation"] == before["stages"][1]["metrics"]["turningEvaluation"]
    assert stage["metrics"]["commandNavigationEvaluations"]["turn"] == result
    assert first["phase2CommandTrainingModels"]["turn"] == provenance
    assert stage["traceUrl"] == before["stages"][1]["traceUrl"]
    assert (output/"stage2-navigation-scene.json").read_bytes() == b'{"existing":true}'
    assert len(stage["traceOptions"]) == 2 and "方向・歩調PPO" in stage["traceOptions"][-1]["title"]
    assert all("command-turn-old" not in option["url"] for option in stage["traceOptions"])
    assert stage["proficiencyPassed"] is False
    assert stage["traceTask"] == "walk"
    assert first["phase2Scope"]["allPhase2TasksPassed"] is False
    assert first["phase2Scope"]["onlineAvatarControl"] is False


def test_even_eight_successful_development_trials_cannot_claim_proficiency(publication):
    output, _, evaluation = publication
    directory, model, result, _ = evaluation(passed=8)
    result["proficiencyPassed"] = True
    dump(directory/"evaluation.json", result)
    result = publish(command_turning=directory, command_turning_model=model, output=output)
    evaluation = result["stages"][1]["metrics"]["commandNavigationEvaluations"]["turn"]
    assert evaluation["navigationVerifiedRate"] == 1.
    assert evaluation["proficiencyPassed"] is False


def test_original_joint_publication_and_command_results_are_separate(publication):
    output, before, evaluation = publication
    turn, turn_model, _, _ = evaluation(command=False, count=32, passed=29, seed=1_851_000)
    obstacle, obstacle_model, _, _ = evaluation(mode="obstacle", command=False)
    command, command_model, command_result, _ = evaluation(passed=4)
    result = publish(turning=turn, obstacle=obstacle, turning_model=turn_model,
        obstacle_model=obstacle_model, command_turning=command, command_turning_model=command_model, output=output)
    stage = result["stages"][1]
    assert stage["gates"]["turning"] == "passed"
    assert stage["metrics"]["turningEvaluation"]["navigationVerifiedRate"] == 29/32
    assert stage["metrics"]["commandNavigationEvaluations"]["turn"] == command_result
    assert len(stage["traceOptions"]) == 3
    assert stage["traceUrl"].startswith("/world-model/stage2-obstacle-")
    assert result["trainingModels"] == before["trainingModels"]
    assert result["stages"][0] == before["stages"][0]


@pytest.mark.parametrize("fitting_rl_steps", [0, 750])
def test_verified_imitation_preserves_actual_rl_counts_without_complete_flag(publication, fitting_rl_steps):
    output, _, evaluation = publication
    directory, model, _, provenance = evaluation(passed=4)
    from pathlib import Path
    dataset = model/"teacher-data"
    dataset.mkdir()
    (dataset/"dataset.npz").write_bytes(b"verified-train-only-dataset")
    dataset_sha = hashlib.sha256((dataset/"dataset.npz").read_bytes()).hexdigest()
    provenance.update(teacherDataset=str(dataset), teacherDatasetSha256=dataset_sha,
        supervisedPretraining={"epochs": 150, "supervisedMinibatches": 750, "supervisedExamples": 2,
                               "rlDecisionTimestepsAdded": fitting_rl_steps, "physicsControlTimestepsAdded": 0},
        teacherCollection={"heldOutScenesUsed": False, "failedEpisodesExcluded": True,
            "avatarProfileSha256": "profile", "bodyModelSha256": "body",
            "motorCheckpointSha256": provenance["motorCheckpointSha256"],
            "commandControlContract": provenance["commandControlContract"], "datasetSha256": dataset_sha,
            "qualifiedEpisodeCount": 1, "qualifiedDecisionCount": 2,
            "episodes": [{"acceptedForSupervision": True, "split": "train", "sceneSeed": 1,
                          "navigationVerified": True, "realFinalGoalUnchangedEverySubstep": True,
                          "maximumExternalForce": 0, "decisions": 2}]})
    dump(model/"provenance.json", provenance)
    if fitting_rl_steps:
        before = (output/"manifest.json").read_bytes()
        with pytest.raises(ValueError, match="must not be added"):
            publish(command_turning=directory, command_turning_model=model, output=output)
        assert (output/"manifest.json").read_bytes() == before
    else:
        result = publish(command_turning=directory, command_turning_model=model, output=output)
        saved = result["phase2CommandTrainingModels"]["turn"]
        assert "complete" not in saved
        assert saved["actualDecisionTimesteps"] == 32768
        assert saved["physicalControlTimesteps"] == 163376
        assert saved["supervisedPretraining"]["supervisedMinibatches"] == 750


@pytest.mark.parametrize("corruption", ["trace_bytes", "policy_bytes", "motor_bytes", "trace_body", "trace_checkpoint", "missing_interface", "missing_motor_path"])
def test_rejected_evidence_does_not_partially_change_publication(publication, corruption):
    output, _, evaluation = publication
    joint, joint_model, _, _ = evaluation(command=False)
    command, command_model, result, provenance = evaluation()
    filename = result["recordings"][0]["filename"]
    if corruption == "trace_bytes":
        (command/filename).write_bytes((command/filename).read_bytes()+b" ")
    elif corruption == "policy_bytes":
        (command_model/"policy.zip").write_bytes(b"changed-policy")
    elif corruption == "motor_bytes":
        from pathlib import Path
        Path(provenance["motorCheckpoint"]).write_bytes(b"changed-motor")
    elif corruption == "missing_interface":
        del provenance["observationCount"]
        dump(command_model/"provenance.json", provenance)
    elif corruption == "missing_motor_path":
        del provenance["motorCheckpoint"]
        dump(command_model/"provenance.json", provenance)
    else:
        trace = json.loads((command/filename).read_text())
        if corruption == "trace_body": trace["bodyModelSha256"] = "different-body"
        else: trace["navigation"]["checkpointSha256"] = "different-checkpoint"
        dump(command/filename, trace)
        result["recordings"][0]["traceSha256"] = hashlib.sha256((command/filename).read_bytes()).hexdigest()
        dump(command/"evaluation.json", result)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError):
        publish(turning=joint, turning_model=joint_model, command_turning=command,
                command_turning_model=command_model, output=output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
