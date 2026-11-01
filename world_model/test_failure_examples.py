"""Failure playback must preserve physics and disclose its evaluation boundary."""

import hashlib

import numpy as np
import pytest

from .environment import HumanoidRoomEnv
from .failure_examples import (
    continue_physical_step, floor_contacts, physics_frame,
    verify_checkpoint_sources, verify_historical_fall,
)


def test_releasing_completed_episode_guard_matches_an_uninterrupted_physical_step():
    stopped = HumanoidRoomEnv(randomize=True)
    uninterrupted = HumanoidRoomEnv(randomize=True)
    try:
        stopped.reset(seed=91823)
        uninterrupted.reset(seed=91823)
        action = np.linspace(-.2, .2, 20)
        for _ in range(8):
            stopped.step(action)
            uninterrupted.step(action)
        stopped._episode_done = True
        continue_physical_step(stopped, action)
        uninterrupted.step(action)
        # Exact equality verifies there was no reset, pose/velocity correction,
        # controller replacement, or change of physical integration timing.
        assert np.array_equal(stopped.data.qpos, uninterrupted.data.qpos)
        assert np.array_equal(stopped.data.qvel, uninterrupted.data.qvel)
        assert np.array_equal(stopped.data.ctrl, uninterrupted.data.ctrl)
        assert stopped.data.time == uninterrupted.data.time
    finally:
        stopped.close()
        uninterrupted.close()


def test_historical_failure_requires_identical_time_pose_velocity_and_reward():
    final = {"time": 2.32, "root_height": .8, "upright": .4, "fall": True,
             "root_position": [.2, -1.3, .8], "root_velocity": [1., -2., -.8]}
    expected = {"seconds": 2.32, "fell": True, "return": 900., "final": final}
    verify_historical_fall(expected, final, 900.)
    with pytest.raises(ValueError, match="historical evaluation"):
        verify_historical_fall(expected, {**final, "time": 2.34}, 900.)
    with pytest.raises(ValueError, match="root_velocity"):
        verify_historical_fall(expected, {**final, "root_velocity": [1., -2., -.7]}, 900.)
    with pytest.raises(ValueError, match="recorded fall"):
        verify_historical_fall(expected, {**final, "fall": False}, 900.)


def test_stale_checkpoint_or_simulation_sources_are_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fixture checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        verify_checkpoint_sources(checkpoint, {"checkpointSha256": "wrong"})
    with pytest.raises(ValueError, match="simulation source changed"):
        verify_checkpoint_sources(checkpoint, {"checkpointSha256": digest,
                                               "sourceSha256": {"environment.py": "wrong"}})
    with pytest.raises(ValueError, match="local module filenames"):
        verify_checkpoint_sources(checkpoint, {"checkpointSha256": digest,
                                               "sourceSha256": {"../environment.py": "wrong"}})


def test_floor_contact_telemetry_uses_actual_solver_force_and_rejects_external_push():
    env = HumanoidRoomEnv()
    try:
        action = np.zeros(20)
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
        contacts = floor_contacts(env)
        assert contacts
        assert sum(item["normalForceNewtons"] for item in contacts) > 400
        assert {item["body"] for item in contacts} <= {"left_foot", "right_foot"}
        frame = physics_frame(env, action, "post-fall-continuation")
        assert frame["controlsPhase"] == "post-fall-continuation"
        assert frame["physicsTelemetry"]["externalForceMaximumAbs"] == 0
        assert frame["physicsTelemetry"]["maximumTorqueLimitFraction"] <= 1
        env.apply_push([1., 0., 0.])
        with pytest.raises(ValueError, match="external force"):
            physics_frame(env, action, "evaluation")
    finally:
        env.close()
