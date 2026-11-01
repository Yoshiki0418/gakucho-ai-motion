"""Reproducibly publish measured outcomes, without promoting unverified skills."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .curriculum import export
from .gait_audit import audit_gait
from .home_scene import create_home_scene
from .training import dump_json, evaluate
from .avatar_profile import load_avatar_profile, avatar_profile_sha256
from .environment import HumanoidRoomEnv


def publish(output, standing, sitting, walking=None, physical_footsteps=False, wbc_trace=None,
            physical_approach=False, avatar_profile=None):
    from stable_baselines3 import PPO
    output, standing, sitting = map(Path, (output, standing, sitting))
    avatar_profile = load_avatar_profile(avatar_profile)
    evaluation_seed_base = 2_000_000 if avatar_profile is not None else 1_000_000
    stand_policy = PPO.load(standing / "policy.zip", device="cpu")
    sit_policy = PPO.load(sitting / "policy.zip", device="cpu")
    manifest = export(output, standing/"policy.zip", standing/"evaluation.json", avatar_profile=avatar_profile)
    if avatar_profile is not None:
        manifest["avatarProfile"] = avatar_profile
        manifest["avatarProfileSha256"] = avatar_profile_sha256(avatar_profile)
    scene = json.loads((output/"stage1-scene.json").read_text())
    holdout = {}
    for name, policy in (("baseline", None), ("policy", stand_policy)):
        holdout[name] = evaluate(policy, scene=scene, seeds=range(evaluation_seed_base, evaluation_seed_base+8),
                                seconds=20, push_force=100,
                                avatar_profile=avatar_profile,
                                trace_path=output/f"stage1-{name}-push.json")
    dump_json(output/"standing-heldout-evaluation.json", holdout)
    sit_scene = json.loads((sitting/"provenance.json").read_text())["scene"]
    seat_result = evaluate(sit_policy, task="sit", scene=sit_scene, reference=True, reference_start=False,
                           seconds=12, seeds=range(evaluation_seed_base+100, evaluation_seed_base+108), audit_sit=True,
                           avatar_profile=avatar_profile,
                           trace_path=output/"stage1-sit.json")
    dump_json(output/"sitting-heldout-evaluation.json", seat_result)
    home = create_home_scene()
    evaluate(stand_policy, scene=home, seeds=[evaluation_seed_base+300], randomize=False,
             avatar_profile=avatar_profile, trace_path=output/"stage1-home.json")
    dump_json(output/"current-home-scene.json", home)
    stage = manifest["stages"][0]
    standing_passed = (holdout["policy"]["survivalRate"] >= .9 and holdout["policy"]["successRate"] >= .9)
    stage["gates"]["standing"] = "passed" if standing_passed else "failed"
    manifest["gates"]["standing"] = stage["gates"]["standing"]
    stage["metrics"]["standingHeldoutPush"] = {
        "forceN": 100, "pulseSeconds": .2, "observationSeconds": 20,
        "count": 8, "baselineSurvivalRate": holdout["baseline"]["survivalRate"],
        "policySurvivalRate": holdout["policy"]["survivalRate"]}
    stage["metrics"]["nearChairSitting"] = {"count": 8, "seconds": 12, "successRate": seat_result["successRate"],
                                            "seatSupportVerifiedRate": seat_result["seatSupportVerifiedRate"],
                                            "scope": "standing start beside a prepared chair; navigation not included"}
    stage["gates"]["sitting"] = "passed" if seat_result["successRate"] >= .9 and seat_result["seatSupportVerifiedRate"] >= .9 else "failed"
    manifest["gates"]["sitting"] = stage["gates"]["sitting"]
    stage["gates"]["walking"] = "failed"
    manifest["gates"]["walking"] = "failed"
    stage["traceOptions"] += [
        {"title": "100 Nで押す・学習後の立位", "url": "/world-model/stage1-policy-push.json"},
        {"title": "100 Nで押す・基準制御との比較", "url": "/world-model/stage1-baseline-push.json"},
        {"title": "近傍の椅子へ着座・学習後", "url": "/world-model/stage1-sit.json"},
        {"title": "現在の住宅の物理形状で立位", "url": "/world-model/stage1-home.json"},
    ]
    if walking:
        walking = Path(walking)
        model = PPO.load(walking/"policy.zip", device="cpu")
        provenance = json.loads((walking/"provenance.json").read_text())
        if physical_approach:
            from .hierarchical import evaluate_hierarchical
            walk_result = evaluate_hierarchical(model, scene=provenance["scene"],
                               seeds=range(evaluation_seed_base+600, evaluation_seed_base+632), trace_path=output/"stage1-walking.json",
                               avatar_profile=avatar_profile,
                               checkpoint_sha256=hashlib.sha256((walking/"policy.zip").read_bytes()).hexdigest())
            if walk_result["modelBasedApproachCount"]:
                stage["traceOptions"].append({"title": "歩行・最後の接近を物理制御で補正する例",
                    "url": "/world-model/stage1-walking-approach-example.json"})
        else:
            walk_result = evaluate(model, task="walk", scene=provenance["scene"], reference=True,
                               reference_start=False, physical_footsteps=physical_footsteps,
                               seconds=provenance.get("maxEpisodeSeconds", 10), seeds=range(evaluation_seed_base+400, evaluation_seed_base+432),
                               audit_walk=True,
                               strict_gait_reward=provenance.get("strictGaitReward", False),
                               avatar_profile=avatar_profile,
                               trace_path=output/"stage1-walking.json")
        quality = audit_gait(json.loads((output/"stage1-walking.json").read_text()))
        walk_result["checkpointSha256"] = hashlib.sha256((walking/"policy.zip").read_bytes()).hexdigest()
        passed = (walk_result["walkingVerifiedRate"] >= .9 and walk_result["successRate"] >= .9
                  and walk_result.get("completionVerifiedRate", 1) >= .9)
        stage["gates"]["walking"] = "passed" if passed else "failed"
        manifest["gates"]["walking"] = stage["gates"]["walking"]
        stage["traceOptions"].append({"title": "歩行の検証記録"+("・通過" if passed else "・未通過"),
                                      "url": "/world-model/stage1-walking.json"})
        stage["metrics"]["walkingQuality"] = quality
        stage["metrics"]["walkingQualityPassRate"] = walk_result["walkingVerifiedRate"]
        stage["metrics"]["walkingEvaluation"] = {
            "count": walk_result["episodeCount"], "controller": walk_result["controller"],
            "successRate": walk_result["successRate"], "survivalRate": walk_result["survivalRate"],
            "completionVerifiedRate": walk_result.get("completionVerifiedRate"),
            "learnedOnlySuccessRate": walk_result.get("learnedOnlySuccessRate"),
            "learnedOnlyCompletionVerifiedRate": walk_result.get("learnedOnlyCompletionVerifiedRate"),
            "modelBasedApproachCount": walk_result.get("modelBasedApproachCount", 0),
            "scope": "2 m flat-floor goal; does not verify unseen-room navigation"}
        dump_json(output/"walking-heldout-evaluation.json", walk_result)
        dump_json(output/"walking-quality-gate.json", quality)
    all_passed = all(value == "passed" for value in stage["gates"].values())
    stage["proficiencyPassed"] = all_passed
    labels = {"standing": "立位", "walking": "歩行", "sitting": "近傍着座"}
    passed_labels = [labels[key] for key, value in stage["gates"].items() if value == "passed"]
    incomplete_labels = [labels[key] for key, value in stage["gates"].items() if value != "passed"]
    stage["status"] = ("基本動作を検証通過" if all_passed else
                       ("・".join(passed_labels)+"を検証済み／" if passed_labels else "")
                       +"・".join(incomplete_labels)+"は未通過")
    stage["description"] = "重力・接触・摩擦と64.7 kgの関節付き身体。立位、近傍の椅子への着座、歩行を個別に検証します。"
    if avatar_profile is not None:
        stage["description"] = "学長アバターの骨格・靴底・座面接触を実測して反映した身体。立位・歩行・近傍着座を再検証します。体重64.7 kgと関節の強さは仮定です。"
    manifest["trainingModels"] = {"standing": json.loads((standing/"provenance.json").read_text()),
                                  "sitting": json.loads((sitting/"provenance.json").read_text())}
    if walking:
        manifest["trainingModels"]["walking"] = provenance
    if wbc_trace:
        wbc = json.loads(Path(wbc_trace).read_text())
        if wbc.get("avatarProfileSha256") != avatar_profile_sha256(avatar_profile):
            raise ValueError("The model-based baseline belongs to a different avatar body")
        if avatar_profile is not None:
            contract_env = HumanoidRoomEnv(avatar_profile=avatar_profile)
            expected_body = contract_env.body_model_sha256
            contract_env.close()
            if wbc.get("bodyModelSha256") != expected_body:
                raise ValueError("The model-based baseline uses a different physical body contract")
        quality = audit_gait(wbc)
        if not quality["walkingVerified"]:
            raise ValueError("The model-based baseline did not pass its physical gait audit")
        dump_json(output/"stage1-wbc-walking.json", wbc)
        stage["traceOptions"].append({"title": "力学の基準制御・平床1 m歩行（学習なし）",
                                      "url": "/world-model/stage1-wbc-walking.json"})
        stage["metrics"]["modelBasedWalkingBaseline"] = {"gait": quality,
            "scope": "one nominal 1 m flat-floor case; analytic inverse dynamics, not a learned policy"}
    manifest["generatedAtUtc"] = datetime.now(timezone.utc).isoformat()
    manifest["evaluationSourceSha256"] = {
        name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ("training.py", "gait_audit.py", "publish.py", "hierarchical.py", "whole_body.py",
                     "avatar_profile.py", "body_contract.py")}
    for later in manifest["stages"][1:]:
        later["status"] = "物理環境の準備済み／第1段階の歩行検証待ち" if not all_passed else "物理環境の準備済み／次の学習課題"
        later["proficiencyPassed"] = False
    dump_json(output/"manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="frontend/public/world-model")
    parser.add_argument("--standing", default=".runtime/world-model/stage1-balance")
    parser.add_argument("--sitting", default=".runtime/world-model/stage1-seating")
    parser.add_argument("--walking")
    parser.add_argument("--physical-footsteps", action="store_true")
    parser.add_argument("--wbc-trace", help="Audited nominal model-based walking trace; does not set a PPO gate")
    parser.add_argument("--physical-approach", action="store_true", help="Audit learned walking with a separate stopped model-based approach on new seeds")
    parser.add_argument("--avatar-profile")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    result = publish(args.output, args.standing, args.sitting, args.walking, args.physical_footsteps,
                     args.wbc_trace, args.physical_approach, args.avatar_profile)
    print(json.dumps({"gates": result["gates"], "trainedSkills": list(result["trainingModels"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
