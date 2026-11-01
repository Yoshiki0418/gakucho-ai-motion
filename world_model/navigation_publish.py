"""Publish measured phase-two trials without promoting unfinished room skills."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .training import dump_json


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _command_result(result):
    """Development examples cannot certify a learned navigation skill."""
    episodes = result["episodes"]
    count = result["episodeCount"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count != len(episodes):
        raise ValueError("Command navigation episode count differs from its trials")
    seeds = [episode["sceneSeed"] for episode in episodes]
    if len(set(seeds)) != count or any(isinstance(seed, bool) or not isinstance(seed, int)
                                      or not 1_000_000 <= seed < 2_000_000 for seed in seeds):
        raise ValueError("Command navigation requires distinct held-out scene seeds")
    if any(not isinstance(episode["navigationVerified"], bool) for episode in episodes):
        raise ValueError("Command navigation trial verification must be boolean")
    rate = sum(episode["navigationVerified"] for episode in episodes)/count
    if not math.isclose(result["navigationVerifiedRate"], rate, abs_tol=1e-12):
        raise ValueError("Command navigation rate differs from its actual trial count")
    final = all(1_800_000 <= seed < 2_000_000 for seed in seeds)
    if result["evaluationRole"] != ("final" if final else "development"):
        raise ValueError("Command navigation evaluation role differs from its held-out cohort")
    if (result.get("observationCount"), result.get("actionCount"),
        result.get("motorObservationCount"), result.get("motorActionCount")) != (112, 2, 110, 20):
        raise ValueError("Command navigation has a different policy/motor interface")
    result["proficiencyPassed"] = bool(result.get("proficiencyPassed") and final and count >= 32
                                       and rate >= .9 and result.get("checkpointSha256"))
    return result


def _verify_supervision(provenance, profile_sha, body_sha):
    fitted, collection = provenance["supervisedPretraining"], provenance["teacherCollection"]
    if (fitted.get("rlDecisionTimestepsAdded"), fitted.get("physicsControlTimestepsAdded")) != (0, 0):
        raise ValueError("Supervised fitting must not be added to physical RL timesteps")
    if collection.get("heldOutScenesUsed") is not False or collection.get("failedEpisodesExcluded") is not True:
        raise ValueError("Supervised command training requires verified train-only examples")
    if (collection.get("avatarProfileSha256"), collection.get("bodyModelSha256"),
        collection.get("motorCheckpointSha256"), collection.get("commandControlContract")) != (
            profile_sha, body_sha, provenance["motorCheckpointSha256"], provenance["commandControlContract"]):
        raise ValueError("Supervised command examples have a different body or motor contract")
    accepted = [episode for episode in collection["episodes"] if episode["acceptedForSupervision"]]
    if not accepted or any(episode.get("split") != "train" or not 0 <= episode["sceneSeed"] < 1_000_000
            or episode.get("navigationVerified") is not True
            or episode.get("realFinalGoalUnchangedEverySubstep") is not True
            or episode.get("maximumExternalForce") != 0 for episode in accepted):
        raise ValueError("Supervised command training requires verified train-only examples")
    decisions = sum(episode["decisions"] for episode in accepted)
    if (collection["qualifiedEpisodeCount"], collection["qualifiedDecisionCount"], fitted["supervisedExamples"]) != (
            len(accepted), decisions, decisions):
        raise ValueError("Supervised command example counts differ from qualified physical trials")
    if collection.get("datasetSha256") != provenance.get("teacherDatasetSha256"):
        raise ValueError("Supervised command dataset and collection checksums differ")
    if provenance.get("teacherDataset") and _sha256(Path(provenance["teacherDataset"])/"dataset.npz") != provenance["teacherDatasetSha256"]:
        raise ValueError("Supervised command dataset differs from its recorded checksum")


def _load_evaluation(directory, model_dir, mode, profile_sha, body_sha, *, command=False):
    directory = Path(directory)
    result = json.loads((directory/"evaluation.json").read_text())
    if result["mode"] != mode or (result["avatarProfileSha256"], result["bodyModelSha256"]) != (profile_sha, body_sha):
        raise ValueError("Navigation evaluation mode or physical body differs")
    if command:
        result = _command_result(result)
    provenance = None
    if model_dir is not None:
        model_dir = Path(model_dir)
        provenance = json.loads((model_dir/"provenance.json").read_text())
        sha = _sha256(model_dir/"policy.zip")
        if sha != result["checkpointSha256"] or sha != provenance["checkpointSha256"]:
            raise ValueError("Navigation evaluation does not belong to this checkpoint")
        if (provenance["avatarProfileSha256"], provenance["bodyModelSha256"]) != (profile_sha, body_sha):
            raise ValueError("Navigation checkpoint provenance has a different physical body")
        if command:
            for key in ("motorCheckpointSha256", "commandControlContract", "observationCount",
                        "actionCount", "motorObservationCount", "motorActionCount"):
                if provenance.get(key) != result.get(key):
                    raise ValueError("Command navigation checkpoint and motor contract differ")
            if provenance.get("motorFrozen") is not True:
                raise ValueError("Command navigation must use a separately frozen motor")
            if "supervisedPretraining" in provenance:
                _verify_supervision(provenance, profile_sha, body_sha)
            if not provenance.get("motorCheckpoint"):
                raise ValueError("Command navigation must identify its actual frozen motor checkpoint")
            if _sha256(provenance["motorCheckpoint"]) != result["motorCheckpointSha256"]:
                raise ValueError("Command navigation frozen motor checkpoint differs")
    records = []
    prefix = f"stage2-{'command-' if command else ''}{mode}-"
    for recording in result["recordings"]:
        filename = recording["filename"]
        if Path(filename).name != filename or not filename.startswith(prefix) or not filename.endswith(".json"):
            raise ValueError("Navigation recording filename is outside its published mode")
        contents = (directory/filename).read_bytes()
        if hashlib.sha256(contents).hexdigest() != recording["traceSha256"]:
            raise ValueError("Navigation recording differs from the audited file")
        trace = json.loads(contents)
        if (trace["avatarProfileSha256"], trace["bodyModelSha256"]) != (profile_sha, body_sha):
            raise ValueError("Navigation recording has a different physical body")
        navigation = trace["navigation"]
        if (navigation["mode"], navigation["sceneSeed"], navigation["checkpointSha256"]) != (
                mode, recording["sceneSeed"], result["checkpointSha256"]):
            raise ValueError("Navigation recording does not belong to this evaluation checkpoint")
        if command:
            for key in ("motorCheckpointSha256", "commandControlContract", "observationCount",
                        "actionCount", "motorObservationCount", "motorActionCount"):
                if navigation.get(key) != result.get(key):
                    raise ValueError("Command navigation recording and motor contract differ")
            if navigation.get("motorFrozen") is not True:
                raise ValueError("Command navigation recording must preserve the frozen motor")
        success = trace["navigationQuality"]["navigationVerified"]
        if not isinstance(success, bool):
            raise ValueError("Navigation recording verification must be boolean")
        if command:
            episode = next((item for item in result["episodes"] if item["sceneSeed"] == recording["sceneSeed"]), None)
            if episode is None or episode["navigationVerified"] != success:
                raise ValueError("Command navigation recording differs from its evaluated trial")
        label = "家具1個を回避" if mode == "obstacle" else "方向を変えて歩行"
        if command:
            label = ("方向・歩調PPOで" if result.get("checkpointSha256") else "方向・歩調の基準制御で")+label
        records.append({"title": f"{label}・条件{recording['sceneSeed']}（{'通過' if success else '未通過'}）",
                        "url": "/world-model/"+filename, "verified": success, "mode": mode,
                        "filename": filename, "contents": contents, "trace": trace})
    return result, provenance, records


def publish(*, turning=None, obstacle=None, turning_model=None, obstacle_model=None,
            command_turning=None, command_obstacle=None, command_turning_model=None,
            command_obstacle_model=None, output="frontend/public/world-model"):
    output = Path(output)
    manifest = json.loads((output/"manifest.json").read_text())
    stage = next(stage for stage in manifest["stages"] if stage["stage"] == 2)
    expected_profile = manifest["avatarProfileSha256"]
    expected_body = manifest["trainingModels"]["walking"]["bodyModelSha256"]
    records, evaluations, models = [], {}, {}
    command_records, command_evaluations, command_models = [], {}, {}
    # Validate every input before writing a trace, scene or manifest.
    for command, inputs in ((False, (("turn", turning, turning_model), ("obstacle", obstacle, obstacle_model))),
                            (True, (("turn", command_turning, command_turning_model),
                                    ("obstacle", command_obstacle, command_obstacle_model)))):
        for mode, directory, model_dir in inputs:
            if directory is None:
                if model_dir is not None:
                    raise ValueError("A navigation checkpoint requires its evaluation directory")
                continue
            result, provenance, loaded_records = _load_evaluation(directory, model_dir, mode,
                expected_profile, expected_body, command=command)
            (command_evaluations if command else evaluations)[mode] = result
            if provenance is not None:
                (command_models if command else models)[mode] = provenance
            (command_records if command else records).extend(loaded_records)
    if not records and not command_records:
        raise ValueError("No actual phase-two recordings were supplied")

    if evaluations:
        gates = {"turning": "passed" if evaluations.get("turn", {}).get("proficiencyPassed") else "failed" if "turn" in evaluations else "pending",
                 "singleFurnitureAvoidance": "passed" if evaluations.get("obstacle", {}).get("proficiencyPassed") else "failed" if "obstacle" in evaluations else "pending",
                 "multiFurnitureNavigation": "pending"}
        stage["gates"] = gates
        stage["status"] = "家具1個の回避を検証済み／複数家具は次の課題" if gates["singleFurnitureAvoidance"] == "passed" else "家具1個の回避を学習・検証中"
        stage["description"] = "同じ学長の身体で、方向転換から家具1個の回避へ進めています。実際の重力・摩擦・関節トルクを使い、家具接触・転倒・歩行・到着後の停止を個別に検証します。"
        stage.setdefault("metrics", {})["turningEvaluation"] = evaluations.get("turn", {})
        stage["metrics"]["navigationEvaluation"] = evaluations.get("obstacle", {})
        stage["metrics"]["scope"] = "one blocking furniture item on flat floors; multiple furniture, multiple rooms and stairs remain unlearned"
        manifest["phase2TrainingModels"] = models
    gates = stage.setdefault("gates", {"turning": "pending", "singleFurnitureAvoidance": "pending"})
    gates["multiFurnitureNavigation"] = "pending"
    stage["proficiencyPassed"] = False
    stage["traceTask"] = "walk"
    if command_evaluations:
        stage.setdefault("metrics", {}).setdefault("commandNavigationEvaluations", {}).update(command_evaluations)
        registered = manifest.setdefault("phase2CommandTrainingModels", {})
        for mode in command_evaluations:
            registered.pop(mode, None)
        registered.update(command_models)

    def replaced(url):
        return (bool(evaluations) and url.startswith(("/world-model/stage2-turn-", "/world-model/stage2-obstacle-"))) or any(
            url.startswith(f"/world-model/stage2-command-{mode}-") for mode in command_evaluations)

    existing = [item for item in stage.get("traceOptions", []) if not replaced(item["url"])]
    options = existing+[{"title": item["title"], "url": item["url"]} for item in records+command_records]
    stage["traceOptions"] = list({item["url"]: item for item in options}.values())
    preferred_records = records or command_records
    preferred = next((item for item in preferred_records if item["mode"] == "obstacle" and item["verified"]),
                     next((item for item in preferred_records if item["mode"] == "obstacle"), preferred_records[0]))
    use_preferred = bool(records) or not stage.get("traceUrl") or replaced(stage["traceUrl"])
    if use_preferred:
        stage["traceUrl"] = preferred["url"]
        stage["sceneUrl"] = "/world-model/stage2-navigation-scene.json"
    manifest["phase2Scope"] = {"started": True, "singleFurnitureAvoidance": gates.get("singleFurnitureAvoidance", "pending"),
        "allPhase2TasksPassed": False, "multiRoomTasksPassed": False, "stairsPassed": False,
        "onlineAvatarControl": False, "publicationSourceSha256": _sha256(__file__)}
    for item in records+command_records:
        (output/item["filename"]).write_bytes(item["contents"])
    if use_preferred:
        dump_json(output/"stage2-navigation-scene.json", preferred["trace"]["scene"])
    dump_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("turning", "obstacle", "turning-model", "obstacle-model", "command-turning",
                 "command-obstacle", "command-turning-model", "command-obstacle-model"):
        parser.add_argument("--"+name)
    parser.add_argument("--output", default="frontend/public/world-model")
    args = parser.parse_args()
    result = publish(turning=args.turning, obstacle=args.obstacle, turning_model=args.turning_model,
                     obstacle_model=args.obstacle_model, command_turning=args.command_turning,
                     command_obstacle=args.command_obstacle, command_turning_model=args.command_turning_model,
                     command_obstacle_model=args.command_obstacle_model, output=args.output)
    print(json.dumps(result["phase2Scope"], ensure_ascii=False))


if __name__ == "__main__":
    main()
