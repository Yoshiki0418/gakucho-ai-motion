import copy
import json
from pathlib import Path

import pytest

from .qualified_visual_training import require_qualified_manifest, compact_audit_frame, make_qualification_trace


def manifest():
    return {"split": "train", "heldoutUsedForLabels": False, "episodes": [
        {"sceneSeed": 93100, "accepted": True, "physicalControls": 23,
         "physicsSubsteps": 230, "quality": {"navigationVerified": True,
         "gates": {key: True for key in ("survived", "collisionFree", "goalReached", "stopped",
             "bothFeetSupport", "sustainedStop", "physicalRepeatedGait", "obstacleActuallyBlocksDirectRoute")}}},
        {"sceneSeed": 93101, "accepted": False, "physicalControls": 2,
         "physicsSubsteps": 20, "quality": {"navigationVerified": False,
         "gates": {"collisionFree": False}}}]}


def test_manifest_keeps_failed_attempts_but_accepts_only_all_gate_successes():
    data = manifest()
    episodes = require_qualified_manifest(data)
    assert len(episodes) == 2
    assert sum(e["accepted"] for e in episodes) == 1


@pytest.mark.parametrize("change", ["split", "heldout", "seed", "accept_failure", "gate", "counts", "empty"])
def test_unqualified_or_privileged_labels_are_rejected(change):
    data = manifest()
    if change == "split": data["split"] = "heldout"
    if change == "heldout": data["heldoutUsedForLabels"] = True
    if change == "seed": data["episodes"][0]["sceneSeed"] = 1180100
    if change == "accept_failure": data["episodes"][1]["accepted"] = True
    if change == "gate": data["episodes"][0]["quality"]["gates"]["sustainedStop"] = False
    if change == "counts": data["episodes"][0]["physicsSubsteps"] = 229
    if change == "empty": data["episodes"] = []
    with pytest.raises(ValueError): require_qualified_manifest(data)


def test_compact_actual_trace_preserves_independent_quality_exactly():
    path = Path(__file__).resolve().parents[1]/'.runtime/world-model/vision-navigation/cuda-depth-conservative-r4/after/stage2-visual-obstacle-1180101-verified.json'
    if not path.exists(): pytest.skip('saved actual successful trace unavailable')
    from .navigation_evaluation import audit_navigation
    trace = json.loads(path.read_text())
    before = audit_navigation(trace)
    compact = {key: value for key,value in trace.items() if key != 'frames'}
    compact['frames'] = [compact_audit_frame(frame) for frame in trace['frames']]
    assert audit_navigation(compact) == before
    from types import SimpleNamespace
    env = SimpleNamespace(unwrapped=SimpleNamespace(dt=trace['dt'],
        avatar_profile_sha256=trace['avatarProfileSha256'],body_model_sha256=trace['bodyModelSha256']),
        scene=trace['scene'],motor_checkpoint_sha256=trace['navigation']['motorCheckpointSha256'],
        snapshot=lambda: trace['frames'][0])
    qualified = make_qualification_trace(env, 93100, 103100)
    qualified['frames'] = compact['frames']
    checked = audit_navigation(qualified)
    assert checked['navigationVerified'] is True
    assert checked['gaitQuality']['gates']['walkTask'] is True
    assert checked['gates'] == before['gates']
    frame = compact_audit_frame(trace['frames'][0])
    frame['stats']['root_position'][0] += 1
    assert frame['stats']['root_position'] != trace['frames'][0]['stats']['root_position']
