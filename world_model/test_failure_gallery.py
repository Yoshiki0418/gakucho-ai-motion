"""Historical failures must not overwrite the accepted policy or its scored gates."""

from copy import deepcopy
import json

import pytest

from .avatar_profile import avatar_profile_sha256
from .failure_gallery import register_failure_examples


@pytest.fixture
def gallery(tmp_path):
    output = tmp_path / "public"
    source = tmp_path / "recordings"
    output.mkdir()
    source.mkdir()
    profile = {"physical": {"rootHeight": 1.}}
    profile_sha = avatar_profile_sha256(profile)
    manifest = {"version": 1, "avatarProfileSha256": profile_sha,
        "trainingModels": {"walking": {"bodyModelSha256": "body", "checkpointSha256": "accepted"}},
        "gates": {"walking": "passed"},
        "stages": [{"stage": 1, "status": "passed", "proficiencyPassed": True,
                    "gates": {"walking": "passed"}, "metrics": {"count": 32, "passed": 31},
                    "traceUrl": "/world-model/stage1-walking.json", "traceOptions": []}]}
    (output / "manifest.json").write_text(json.dumps(manifest))
    entry = {"filename": "stage1-walking-fall-1.json", "title": "Historical fall",
             "seed": 1, "checkpointSha256": "earlier", "fallTimeSeconds": .02, "settlingSeconds": .02}
    trace = {"version": 1, "geometryScope": "complete-world", "dt": .02,
        "avatarProfile": profile, "avatarProfileSha256": profile_sha, "bodyModelSha256": "body",
        "physicsRestBodies": [{"name": "pelvis"}],
        "failureExample": {**entry, "evaluationExcluded": True},
        "frames": [{"time": time, "bodies": [{"name": "pelvis"}], "geoms": [{"id": "pelvis"}],
                    "stats": {"fall": time >= .02}} for time in (0., .02, .04)]}
    (source / entry["filename"]).write_text(json.dumps(trace))
    index = {"version": 1, "avatarProfileSha256": profile_sha, "bodyModelSha256": "body", "examples": [entry]}
    index_path = source / "failure-examples.json"
    index_path.write_text(json.dumps(index))
    return output, index_path, manifest, trace, entry


def test_failures_are_idempotent_and_leave_scored_results_unchanged(gallery):
    output, index_path, manifest, trace, entry = gallery
    for _ in range(2):
        result = register_failure_examples(index_path, output)
        assert result["gates"] == manifest["gates"]
        assert result["trainingModels"] == manifest["trainingModels"]
        assert result["failureExamples"]["evaluationExcluded"] is True
        original_stage = deepcopy(manifest["stages"][0])
        actual_stage = deepcopy(result["stages"][0])
        del original_stage["traceOptions"], actual_stage["traceOptions"]
        assert actual_stage == original_stage
        assert len(result["stages"][0]["traceOptions"]) == 1
        assert json.loads((output / entry["filename"]).read_text()) == trace


@pytest.mark.parametrize("change, message", [
    (lambda trace: trace.update(bodyModelSha256="different"), "different avatar body"),
    (lambda trace: trace["failureExample"].update(evaluationExcluded=False), "must be excluded"),
    (lambda trace: trace["failureExample"].update(checkpointSha256="accepted"), "earlier checkpoint"),
    (lambda trace: trace["frames"][1]["stats"].update(fall=False), "declared physical fall"),
])
def test_invalid_examples_do_not_change_publication(gallery, change, message):
    output, index_path, manifest, trace, entry = gallery
    change(trace)
    (index_path.parent / entry["filename"]).write_text(json.dumps(trace))
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match=message):
        register_failure_examples(index_path, output)
    assert (output / "manifest.json").read_bytes() == before
    assert not (output / entry["filename"]).exists()
