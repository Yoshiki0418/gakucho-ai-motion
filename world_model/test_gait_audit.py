"""Sensor-math fixtures are synthetic; real saved traces are checked separately."""

import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .gait_audit import _sole_height, audit_gait


def _fixture(sequence=("left", "right", "left", "right")):
    frames = []
    windows = [(0.80 + i * .50, 1.04 + i * .50, foot) for i, foot in enumerate(sequence)]
    for time in np.arange(0, 3.021, .02):
        geoms = []
        stats = {"root_position": [0, -max(time - .5, 0) / 2.5, .93], "root_height": .93,
                 "upright": 1., "fall": False, "goal_distance": max(.10, 1.10 - max(time - .5, 0) / 2.5)}
        for side in ("left", "right"):
            air = any(start <= time <= end and foot == side for start, end, foot in windows)
            height = .06 if air else .003
            geoms.append({"id": "humanoid_" + side + "_foot", "type": "box",
                          "position": [.1 if side == "left" else -.1, 0, height + .035],
                          "quaternion": [1, 0, 0, 0], "size": [.055, .135, .035]})
            stats[side + "_foot_force"] = 0. if air else 300.
        frames.append({"time": float(time), "geoms": geoms, "stats": stats})
    return {"sizeConvention": "mujoco", "dt": .02, "task": "walk",
            "controller": "synthetic sensor fixture, not a physical evaluation", "frames": frames}


def test_tilted_box_bottom_matches_all_eight_actual_world_corners():
    for angles in ((.4, -.3, .8), (0, 0, 1.3), (.9, .2, -.4)):
        rotation = Rotation.from_euler("xyz", angles)
        q = rotation.as_quat()[[3, 0, 1, 2]]
        geom = {"type": "box", "position": [.3, -.5, 1.2],
                "quaternion": q.tolist(), "size": [.07, .12, .03]}
        corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * geom["size"]
        expected = (rotation.apply(corners) + geom["position"])[:, 2].min()
        assert _sole_height(geom) == pytest.approx(expected, abs=1e-12)
        geom["quaternion"] = (-q).tolist()
        assert _sole_height(geom) == pytest.approx(expected, abs=1e-12)


def test_synthetic_sensor_sequence_exercises_all_explicit_threshold_gates():
    result = audit_gait(_fixture())
    assert result["walkingVerified"]
    assert result["feet"]["left"]["swingCount"] == 2
    assert result["feet"]["right"]["swingCount"] == 2
    assert result["supportAlternations"] == 3
    assert result["forwardTravelMeters"] > .8
    assert result["feet"]["left"]["maxSoleClearanceMeters"] == pytest.approx(.06)
    json.dumps(result, allow_nan=False)


def _set_swing_windows(trace, windows):
    for frame in trace["frames"]:
        for geom, side in zip(frame["geoms"], ("left", "right")):
            air = any(start-1e-9 <= frame["time"] <= end+1e-9 and foot == side
                      for start, end, foot in windows)
            geom["position"][2] = .095 if air else .038
            frame["stats"][side+"_foot_force"] = 0. if air else 300.


def test_fast_alternating_contact_pulses_do_not_manufacture_repeated_steps():
    trace = _fixture()
    windows = [(.80+.14*i, .90+.14*i, "left" if i % 2 == 0 else "right")
               for i in range(8)]
    _set_swing_windows(trace, windows)
    result = audit_gait(trace)
    assert not result["walkingVerified"]
    assert result["supportAlternations"] == 1
    for side in ("left", "right"):
        foot = result["feet"][side]
        assert foot["rawSwingCount"] == 4
        assert foot["swingCount"] == 1
        assert len(foot["excludedSwings"]) == 3
        assert all(event["reason"] == "same_foot_start_interval_too_short"
                   for event in foot["excludedSwings"])
        assert all(event["startIntervalSeconds"] == pytest.approx(.28)
                   for event in foot["excludedSwings"])
    # Repeated flicker is compared with each preceding raw start, including
    # exclusions; it cannot regain a count every .4s in a longer chain.
    assert result["feet"]["left"]["excludedSwings"][-1]["start"] > 1.6


def test_same_foot_start_interval_boundary_and_override_are_explicit():
    trace = _fixture()
    _set_swing_windows(trace, [(.8, .9, "left"), (1., 1.1, "right"),
                              (1.2, 1.3, "left"), (1.4, 1.5, "right")])
    result = audit_gait(trace)
    assert result["thresholds"]["min_swing_start_interval_seconds"] == .4
    assert not result["walkingVerified"]
    assert audit_gait(trace, {"min_swing_start_interval_seconds": .39})["walkingVerified"]
    _set_swing_windows(trace, [(.8, .9, "left"), (1., 1.1, "right"),
                              (1.22, 1.32, "left"), (1.42, 1.52, "right")])
    assert audit_gait(trace)["walkingVerified"]


def test_short_glitch_does_not_suppress_a_later_complete_swing():
    trace = _fixture()
    _set_swing_windows(trace, [(.8, 1., "left"), (1.3, 1.5, "right"),
                              (1.7, 1.72, "left"), (1.8, 2., "left"),
                              (2.3, 2.5, "right")])
    result = audit_gait(trace)
    assert result["walkingVerified"]
    assert result["feet"]["left"]["rawCandidateCount"] == 3
    assert result["feet"]["left"]["rawSwingCount"] == 2
    assert result["feet"]["left"]["swingCount"] == 2
    assert result["feet"]["left"]["excludedSwings"][0]["reason"] == "duration_below_minimum"


def test_short_fragments_cannot_break_a_continuous_flicker_chain_into_steps():
    trace = _fixture()
    _set_swing_windows(trace, [(.8, .9, "left"), (1.04, 1.06, "left"),
                              (1.24, 1.26, "left"), (1.44, 1.54, "left")])
    result = audit_gait(trace)
    foot = result["feet"]["left"]
    assert foot["rawCandidateCount"] == 4
    assert foot["rawSwingCount"] == 2
    assert foot["swingCount"] == 1
    excluded = [event for event in foot["excludedSwings"]
                if event["reason"] == "same_foot_start_interval_too_short"]
    assert len(excluded) == 1
    assert excluded[0]["candidateClusterStart"] == pytest.approx(.8)
    assert excluded[0]["previousCandidateStart"] == pytest.approx(1.24)


def test_slow_control_sensor_sequence_is_independent_of_t2m_cycle_phase():
    trace = _fixture()
    trace["controller"] = "synthetic slow whole-body sensor fixture"
    trace["dt"] *= 3
    for frame in trace["frames"]:
        frame["time"] *= 3
        frame["stats"]["reference_phase"] = 0.
    result = audit_gait(trace)
    assert result["walkingVerified"]
    assert result["feet"]["left"]["swingCount"] == result["feet"]["right"]["swingCount"] == 2


def test_one_leg_steps_and_simultaneous_jumps_do_not_qualify_as_walking():
    result = audit_gait(_fixture(("left", "left", "left", "left")))
    assert not result["walkingVerified"]
    assert not result["gates"]["rightRepeatedSwing"]
    assert not result["gates"]["alternatingSupport"]
    trace = _fixture()
    for frame in trace["frames"]:
        if frame["stats"]["left_foot_force"] == 0 or frame["stats"]["right_foot_force"] == 0:
            for geom in frame["geoms"]:
                geom["position"][2] = .095
            frame["stats"]["left_foot_force"] = frame["stats"]["right_foot_force"] = 0.
    result = audit_gait(trace)
    assert not result["walkingVerified"]
    assert result["feet"]["left"]["swingCount"] == result["feet"]["right"]["swingCount"] == 0


def test_force_loading_and_initial_velocity_shortcuts_are_rejected():
    trace = _fixture()
    for frame in trace["frames"]:
        frame["stats"]["left_foot_force"] = 300.
        frame["stats"]["right_foot_force"] = 300.
    assert not audit_gait(trace)["walkingVerified"]
    trace = _fixture()
    for frame in trace["frames"]:
        frame["stats"]["root_position"][1] = -min(frame["time"] / .5, 1)
    result = audit_gait(trace)
    assert result["forwardTravelMeters"] == pytest.approx(0)
    assert not result["gates"]["forwardTravel"]


def test_floor_translation_preserves_sensor_clearance_and_gate_results():
    original = _fixture()
    shifted = copy.deepcopy(original)
    for frame in shifted["frames"]:
        for geom in frame["geoms"]:
            geom["position"][2] += 2.4
        frame["stats"]["root_position"][2] += 2.4
    a = audit_gait(original)
    b = audit_gait(shifted, {"floor_height_meters": 2.4})
    assert a["gates"] == b["gates"]
    assert a["feet"]["right"]["swingCount"] == b["feet"]["right"]["swingCount"]
    assert a["feet"]["left"]["maxSoleClearanceMeters"] == pytest.approx(b["feet"]["left"]["maxSoleClearanceMeters"])


@pytest.mark.parametrize("change,failed", [("goal", "goalReached"), ("fall", "survived"),
                                          ("height", "rootHeight"), ("upright", "upright"),
                                          ("sit", "walkTask")])
def test_successful_sensor_steps_still_require_task_survival_height_and_goal(change, failed):
    trace = _fixture()
    if change == "goal":
        trace["frames"][-1]["stats"]["goal_distance"] = .5
    elif change == "fall":
        trace["frames"][-1]["stats"]["fall"] = True
    elif change == "height":
        trace["frames"][-1]["stats"]["root_height"] = .5
    elif change == "upright":
        trace["frames"][-1]["stats"]["upright"] = .6
    else:
        trace["task"] = "sit"
    result = audit_gait(trace)
    assert not result["gates"][failed] and not result["walkingVerified"]


def test_missing_geometry_bad_quaternion_and_sparse_recording_are_not_accepted():
    trace = _fixture()
    trace["frames"][2]["geoms"].pop()
    with pytest.raises(ValueError, match="humanoid_right_foot"):
        audit_gait(trace)
    trace = _fixture()
    trace["frames"][0]["geoms"][0]["quaternion"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="zero"):
        audit_gait(trace)
    trace = _fixture()
    trace["frames"] = trace["frames"][::5]
    assert not audit_gait(trace)["gates"]["samplingComplete"]


def test_real_saved_incomplete_walk_and_seated_physics_traces_fail_walking_gate():
    root = Path(__file__).resolve().parents[1]
    paths = [root / ".runtime/world-model/stage1-reference-walk/rollout.json",
             Path("/tmp/gakucho-physical-reference-20261014/sit-pd-rollout.json")]
    available = [path for path in paths if path.exists()]
    if not available:
        pytest.skip("Machine-local recorded physics traces are not packaged test fixtures")
    for path in available:
        result = audit_gait(json.loads(path.read_text()))
        assert not result["walkingVerified"]
        if result["task"] == "sit":
            assert not result["gates"]["walkTask"]


def test_real_saved_two_meter_gait_survives_debouncing_and_fallen_flicker_is_reported():
    directory = Path("/tmp/gakucho-contact-gait-audit-20261014")
    paths = [directory/f"trace-{seed}.json" for seed in range(90000, 90004)]
    if not all(path.exists() for path in paths):
        pytest.skip("Machine-local recorded physics traces are not packaged test fixtures")
    for seed, path in zip(range(90000, 90004), paths):
        result = audit_gait(json.loads(path.read_text()))
        if seed != 90001:
            assert result["walkingVerified"]
        else:
            assert not result["walkingVerified"]
            right = result["feet"]["right"]
            assert right["rawSwingCount"] == 3
            assert right["swingCount"] == 2
            assert any(event["reason"] == "same_foot_start_interval_too_short"
                       and event["startIntervalSeconds"] == pytest.approx(.18)
                       for event in right["excludedSwings"])
