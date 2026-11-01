"""The approach must start from supported, stopped, unfinished physical states."""
import numpy as np
import pytest
from gymnasium import spaces

from .hierarchical import evaluate_hierarchical, safe_stopped_handoff


def stopped_state():
    return {"root_velocity": [0., 0., 0.], "fall": False, "success": False,
            "goal_distance": .35, "upright": .99, "left_foot_force": 310., "right_foot_force": 325.}


def test_handoff_rejects_airborne_moving_fallen_or_already_complete_states():
    good = stopped_state()
    assert safe_stopped_handoff(good, .30)[0]
    assert not safe_stopped_handoff(good, .28)[0]
    for key, value in (("root_velocity", [.041, 0, 0]), ("fall", True), ("success", True),
                       ("goal_distance", .65), ("goal_distance", .20), ("upright", .90),
                       ("left_foot_force", 0.), ("right_foot_force", 0.)):
        assert not safe_stopped_handoff({**good, key: value}, 10)[0]


def test_sensor_checkpoint_cannot_be_silently_used_for_basic_walking():
    class WrongPolicy:
        observation_space = spaces.Box(-np.inf, np.inf, (110,), dtype=np.float32)
    with pytest.raises(ValueError, match="59-input"):
        evaluate_hierarchical(WrongPolicy(), scene=None, seeds=[42])
