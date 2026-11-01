"""A scored arrival must not hide an earlier collision or a brief lost support."""
import json
from pathlib import Path

import numpy as np
import pytest

from .navigation_evaluation import audit_navigation


@pytest.fixture
def actual_arrival():
    path = Path(__file__).resolve().parents[1]/"frontend/public/world-model/stage1-walking.json"
    trace = json.loads(path.read_text())
    trace["navigation"] = {"mode": "turn"}
    for frame in trace["frames"]:
        stats = frame["stats"]
        forces = [stats["left_foot_force"], stats["right_foot_force"]]
        stats["navigation_floor_support"] = [force >= 20 for force in forces]
        stats["navigation_success_seconds"] = stats["success_seconds"]
        stats["collision_occurred"] = False
        stats["navigation"] = {"metrics": {"floor_support_forces_newtons": forces,
            "goal_floor_matches": True, "root_height_error_m": abs(stats["root_height"]-1.0063143303680484)}}
    assert np.linalg.norm(trace["frames"][-1]["stats"]["root_velocity"]) < .25
    return trace


def test_actual_successful_physics_trace_passes_independent_stop_and_gait(actual_arrival):
    result = audit_navigation(actual_arrival)
    assert result["navigationVerified"]
    assert result["gaitQuality"]["walkingVerified"]


def test_early_collision_cannot_be_erased_by_a_clean_final_arrival(actual_arrival):
    actual_arrival["frames"][30]["stats"]["collision_occurred"] = True
    assert actual_arrival["frames"][-1]["stats"]["collision_occurred"] is False
    result = audit_navigation(actual_arrival)
    assert not result["navigationVerified"]
    assert "collisionFree" in result["failedGates"]


def test_forged_success_counter_cannot_hide_a_lost_floor_support(actual_arrival):
    frame = actual_arrival["frames"][-10]
    frame["stats"]["navigation"]["metrics"]["floor_support_forces_newtons"][0] = 0.
    assert actual_arrival["frames"][-1]["stats"]["navigation_success_seconds"] > .5
    result = audit_navigation(actual_arrival)
    assert not result["navigationVerified"]
    assert "sustainedStop" in result["failedGates"]


def test_clear_route_cannot_be_reported_as_furniture_avoidance(actual_arrival):
    actual_arrival["navigation"]["mode"] = "obstacle"
    actual_arrival["scene"]["metadata"]["directRouteBlocked"] = True
    result = audit_navigation(actual_arrival)
    assert not result["navigationVerified"]
    assert "obstacleActuallyBlocksDirectRoute" in result["failedGates"]


def test_single_fast_sample_invalidates_the_claimed_half_second_stop(actual_arrival):
    actual_arrival["frames"][-15]["stats"]["root_velocity"] = [.4, 0., 0.]
    result = audit_navigation(actual_arrival)
    assert "sustainedStop" in result["failedGates"]
