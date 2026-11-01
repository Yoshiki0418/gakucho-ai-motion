"""Audit episodic physical worlds without claiming a trained skill."""

import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium import spaces

from .environment import HumanoidRoomEnv
from .generalization import (GeneralizationWrapper, assert_policy_compatible,
                             evaluate_generalization, make_generalization_env,
                             validate_scene_seed, verify_generalization)
from .avatar_profile import avatar_profile_sha256, load_avatar_profile
from .body_contract import bind_policy_body
from .scenes import HELD_OUT_SEED_BASE, generate_scene
from .sensors import SceneSensorWrapper, measure_scene


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_ten_episode_rebuilds_are_deterministic_110_and_change_layouts(stage):
    first = make_generalization_env(stage, base_seed=20261014, randomize=False, difficulty=(.25, .85))
    second = make_generalization_env(stage, base_seed=20261014, randomize=False, difficulty=(.25, .85))
    scene_seeds, hashes, ranges = [], [], []
    models = []
    try:
        for _ in range(10):
            observation, info = first.reset()
            other_observation, other_info = second.reset()
            np.testing.assert_array_equal(observation, other_observation)
            assert observation.shape == (110,)
            assert first.observation_space.contains(observation)
            assert np.isfinite(observation).all()
            assert info["generalization"] == other_info["generalization"]
            assert info["scene_sensors"] == other_info["scene_sensors"]
            assert isinstance(first.env, SceneSensorWrapper)
            assert first.unwrapped.observation_space.shape == (59,)
            assert np.all(first.unwrapped.model.opt.gravity == [0, 0, -9.81])
            assert not np.any(first.unwrapped.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            assert first.unwrapped.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
            assert 0 <= info["generalization"]["sceneSeed"] < HELD_OUT_SEED_BASE
            assert not info["generalization"]["proficiencyPassed"]
            assert info["generalization"]["learningStatus"] == "not_trained_or_validated"
            scene_seeds.append(info["generalization"]["sceneSeed"])
            hashes.append(info["generalization"]["scene_sha256"])
            ranges.append(observation[59:91].copy())
            models.append(first.unwrapped.model)
        assert len(set(scene_seeds)) == 10
        assert len(set(hashes)) == 10
        assert all(models[i] is not models[j] for i in range(10) for j in range(i))
        assert any(not np.array_equal(ranges[0], value) for value in ranges[1:])
        # Explicitly reseeding reproduces the same world/state even when it was
        # the immediately preceding episode, as Gym reset(seed=...) requires.
        one, one_info = first.reset(seed=184)
        two, two_info = first.reset(seed=184)
        np.testing.assert_array_equal(one, two)
        assert one_info["generalization"]["scene_sha256"] == two_info["generalization"]["scene_sha256"]
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("split,valid,invalid", [
    ("train", [0, 999999], [-1, 1000000, 2000000, True]),
    ("heldout", [1000000, 1999999], [-1, 0, 999999, 2000000, True]),
])
def test_scene_seed_domains_cannot_leak_and_bad_reset_preserves_current_physics(split, valid, invalid):
    env = make_generalization_env(2, split=split, randomize=False)
    try:
        for scene_seed in valid:
            observation, info = env.reset(options={"scene_seed": scene_seed})
            assert info["generalization"]["sceneSeed"] == scene_seed
            assert validate_scene_seed(scene_seed, split) == scene_seed
            assert observation.shape == (110,)
        current_model = env.unwrapped.model
        qpos, qvel, time = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), env.unwrapped.data.time
        for scene_seed in invalid:
            with pytest.raises(ValueError):
                env.reset(options={"scene_seed": scene_seed})
            with pytest.raises(ValueError):
                validate_scene_seed(scene_seed, split)
            assert env.unwrapped.model is current_model
            np.testing.assert_array_equal(env.unwrapped.data.qpos, qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, qvel)
            assert env.unwrapped.data.time == time
        with pytest.raises(ValueError):
            env.reset(options={"split": "train"})
        with pytest.raises(AttributeError):
            env.split = "heldout" if split == "train" else "train"
    finally:
        env.close()


def test_task_goal_bits_local_goal_and_heading_are_explicit_without_moving_spawn():
    env = make_generalization_env(2, task="mixed", randomize=False)
    try:
        for goal_id, task in [("across-room", "walk"), ("chair-seat", "sit"), ("spawn-stability", "stand")]:
            observation, info = env.reset(options={"scene_seed": 37, "physics_seed": 9, "goal_id": goal_id})
            meta = info["generalization"]
            assert meta["selectedTask"] == task
            assert meta["selectedGoalId"] == goal_id
            assert env.current_scene["goals"][0]["id"] == goal_id
            assert env.unwrapped.task == task
            expected_bits = [float(task == name) for name in ("stand", "walk", "sit")]
            np.testing.assert_array_equal(observation[53:56], expected_bits)
            physical_goal = np.asarray(env.selected_goal["position"], dtype=float)
            if task == "sit":
                physical_goal[2] += .09
            np.testing.assert_array_equal(env.unwrapped.goal, physical_goal)
            rotation = env.unwrapped.data.xmat[env.unwrapped.pelvis_body_id].reshape(3, 3)
            np.testing.assert_allclose(observation[50:53], rotation.T @ (physical_goal - env.unwrapped.data.qpos[:3]),
                                       atol=3e-7, rtol=0)
            source = generate_scene(2, seed=37, difficulty=.5)
            np.testing.assert_array_equal(env.current_scene["spawn"], source["spawn"])
            np.testing.assert_array_equal(env.unwrapped.data.qpos[:2], source["spawn"][:2])
            np.testing.assert_allclose(rotation @ [0, -1, 0],
                                       [math.sin(meta["spawnHeading"]), -math.cos(meta["spawnHeading"]), 0],
                                       atol=1e-12, rtol=0)
            assert meta["spawnHeading"] == meta["initialGoalHeading"]
            assert meta["goal"]["heading"] == meta["goalHeading"]
            assert not meta["goal_heading_observed"]
            if task == "sit":
                assert np.linalg.norm(env.unwrapped.data.qpos[:2] - physical_goal[:2]) > 1
                assert info["root_height"] > .9
                assert not meta["metrics"]["selected_seat_support_contact"]
        with pytest.raises(ValueError):
            env.reset(options={"scene_seed": 37, "goal_id": "return-to-entry"})
    finally:
        env.close()


def test_wrapper_sensor_and_metrics_do_not_apply_root_forces_or_change_contact_dynamics():
    wrapped = make_generalization_env(2, task="walk", randomize=False, max_episode_seconds=.3)
    observation, info = wrapped.reset(options={"scene_seed": 85, "physics_seed": 18, "goal_id": "across-room"})
    base = HumanoidRoomEnv(task="walk", scene=copy.deepcopy(wrapped.current_scene), randomize=False,
                           max_episode_seconds=.3)
    direct, _ = base.reset(seed=18)
    try:
        np.testing.assert_array_equal(observation[:59], direct)
        for action in np.random.default_rng(11).uniform(-.01, .01, (10, 20)):
            new, reward, done, truncated, info = wrapped.step(action)
            direct, direct_reward, direct_done, direct_truncated, direct_info = base.step(action)
            np.testing.assert_array_equal(new[:59], direct)
            np.testing.assert_array_equal(wrapped.unwrapped.data.qpos, base.data.qpos)
            np.testing.assert_array_equal(wrapped.unwrapped.data.qvel, base.data.qvel)
            assert reward == direct_reward
            assert (done, truncated) == (direct_done, direct_truncated)
            assert info["contact_count"] == direct_info["contact_count"]
            assert info["time"] == direct_info["time"]
            assert not np.any(wrapped.unwrapped.data.xfrc_applied)
            assert not np.any(wrapped.unwrapped.data.qfrc_applied)
            assert np.isfinite(new).all()
            np.testing.assert_allclose(info["scene_sensors"]["floor_heights_m"], 0, atol=1e-12)
            assert all(info["scene_sensors"]["floor_valid"])
        assert info["contact_count"] > 0
        assert info["left_foot_force"] + info["right_foot_force"] > 200
        before = wrapped.unwrapped.data.qpos.copy()
        forces = wrapped.unwrapped.data.xfrc_applied.copy()
        wrapped._add_info(info, advance=False)
        np.testing.assert_array_equal(wrapped.unwrapped.data.qpos, before)
        np.testing.assert_array_equal(wrapped.unwrapped.data.xfrc_applied, forces)
    finally:
        base.close()
        wrapped.close()


def test_lower_floor_xz_arrival_cannot_pass_a_second_floor_goal():
    env = make_generalization_env(4, difficulty=.85, split="heldout", randomize=False)
    try:
        _, info = env.reset(options={"scene_seed": 1000007, "goal_id": "second-floor-target"})
        base = env.unwrapped
        # A counterfactual observation below the actual second-floor slab proves
        # that the existing planar success flag cannot qualify a stair task.
        base.data.qpos[:2] = env.selected_goal["position"][:2]
        base.data.qpos[2] = .93
        mujoco.mj_forward(base.model, base.data)
        raw_info = base._info()
        raw_info["success"] = True
        raw_info["scene_sensors"] = measure_scene(base)
        qpos_before = base.data.qpos.copy()
        inspected = env._add_info(raw_info, advance=True)
        metrics = inspected["generalization"]["metrics"]
        assert metrics["horizontal_error_m"] == 0
        assert metrics["root_height_error_m"] > 2.5
        assert metrics["measured_surface_height_m"] == pytest.approx(0, abs=1e-12)
        assert not metrics["surface_level_matches"]
        assert metrics["base_success"]
        assert not metrics["instant_success"]
        assert not inspected["generalization"]["qualified_success"]
        assert not inspected["generalization"]["terrain_prior_supported"]
        np.testing.assert_array_equal(base.data.qpos, qpos_before)
        assert not np.any(base.data.xfrc_applied)
    finally:
        env.close()


class _ZeroPolicy:
    def __init__(self, observation_count):
        self.observation_space = spaces.Box(-np.inf, np.inf, (observation_count,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (20,), dtype=np.float32)
        self.predictions = 0

    def predict(self, observation, deterministic=True):
        assert observation.shape == self.observation_space.shape
        self.predictions += 1
        return np.zeros(20, np.float32), None


def test_59_checkpoints_rejected_before_prediction_and_heldout_failures_reported():
    incompatible = _ZeroPolicy(59)
    with pytest.raises(ValueError, match="110-input"):
        evaluate_generalization(incompatible, stages=[2], scene_seeds=[1000000], seconds=.04)
    assert incompatible.predictions == 0
    policy = _ZeroPolicy(110)
    report = evaluate_generalization(policy, stages=[2, 3, 4], scene_seeds=[1000000, 1000001],
                                     seconds=.04, randomize=False)
    assert policy.predictions == 12
    assert report["episode_count"] == 6
    assert report["observation_count"] == 110
    assert not report["proficiencyPassed"]
    assert not report["terrain_prior_supported"]
    for episode in report["episodes"]:
        assert episode["scene_seed"] >= HELD_OUT_SEED_BASE
        assert episode["max_external_force"] == 0
        assert not episode["qualified_success"]
        assert episode["final"]["metrics"]["horizontal_error_m"] > 1
        assert episode["physics_seconds"] == pytest.approx(.04, abs=1e-12)
    for stage in report["stages"].values():
        assert stage["qualified_success_rate"] == 0
        assert not stage["proficiencyPassed"]
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="heldout"):
        evaluate_generalization(policy, stages=[2], scene_seeds=[73], seconds=.04)


def test_explicit_footstep_prior_preserves_110_and_discloses_terrain_gap():
    env = make_generalization_env(4, difficulty=.25, motion_prior="footsteps", randomize=False,
                                  max_episode_seconds=.02)
    try:
        observation, info = env.reset(options={"scene_seed": 52})
        assert observation.shape == (110,)
        assert info["generalization"]["base_env_class"] == "FootstepHumanoidEnv"
        assert info["generalization"]["motion_prior"] == "footsteps"
        assert not info["generalization"]["terrain_prior_supported"]
        assert not env.unwrapped.reference_start
        assert env.unwrapped.reference.duration == 1.12
        env.step(np.zeros(20, np.float32))
        assert not np.any(env.unwrapped.data.xfrc_applied)
    finally:
        env.close()
    with pytest.raises(ValueError, match="walk-only"):
        make_generalization_env(2, task="sit", motion_prior="footsteps")


def test_invalid_factory_and_configuration_fail_explicitly():
    with pytest.raises(ValueError, match="59-input"):
        make_generalization_env(2, env_factory=lambda **kwargs: SceneSensorWrapper(HumanoidRoomEnv(**kwargs)))
    for kwargs in [{"stage": 1}, {"stage": 2.5}, {"split": "test"}, {"difficulty": (-.1, .5)},
                   {"env_kwargs": {"scene": {}}}, {"motion_prior": "footsteps", "env_factory": HumanoidRoomEnv}]:
        with pytest.raises(ValueError):
            GeneralizationWrapper(**kwargs)


@pytest.fixture
def gakucho_profile():
    return load_avatar_profile(Path(__file__).resolve().parents[1] /
                               "frontend/public/world-model/gakucho-avatar-profile.json")


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_avatar_dimensions_and_body_contract_survive_scene_rebuilds(stage, gakucho_profile):
    original = copy.deepcopy(gakucho_profile)
    env = make_generalization_env(stage, avatar_profile=gakucho_profile, task="stand", randomize=False)
    expected = HumanoidRoomEnv(avatar_profile=original)
    try:
        # The caller cannot change a future episode's physical body after init.
        gakucho_profile["physical"]["rootHeight"] += .02
        previous_model = None
        for scene_seed in (40, 41, 42):
            observation, info = env.reset(options={"scene_seed": scene_seed})
            base = env.unwrapped
            assert observation.shape == (110,)
            assert base.model is not previous_model
            previous_model = base.model
            assert base.avatar_profile_sha256 == avatar_profile_sha256(original)
            assert base.body_model_sha256 == expected.body_model_sha256
            assert info["generalization"]["body_model_sha256"] == expected.body_model_sha256
            assert info["generalization"]["avatar_profile_sha256"] == avatar_profile_sha256(original)
            for side in ("left", "right"):
                for section, child, key in (("legs", "shank", "thighLength"),
                                            ("legs", "foot", "shankLength")):
                    compiled = np.linalg.norm(base.model.body_pos[base.model.body(f"{side}_{child}").id])
                    assert compiled == pytest.approx(original["physical"][section][side][key], abs=1e-8)
                foot = base.model.geom(f"humanoid_{side}_foot").id
                np.testing.assert_allclose(base.model.geom_size[foot], original["physical"]["foot"][side]["halfSize"],
                                           atol=1e-10)
            source = generate_scene(stage, seed=scene_seed, difficulty=.5)
            np.testing.assert_array_equal(env.current_scene["spawn"], source["spawn"])
            goal = env.selected_goal
            assert goal["surfaceHeight"] == source["spawn"][2]
            assert goal["position"][2] == pytest.approx(source["spawn"][2] + base.standing_root_height)
            assert base.goal[2] == goal["position"][2]
            metrics = info["generalization"]["metrics"]
            assert metrics["goal_surface_height_m"] == source["spawn"][2]
            assert metrics["root_height_error_m"] == pytest.approx(0, abs=1e-12)
            assert not info["generalization"]["proficiencyPassed"]
            env.step(np.zeros(20, np.float32))
            assert not np.any(base.data.xfrc_applied) and not np.any(base.data.qfrc_applied)
    finally:
        expected.close()
        env.close()


def test_avatar_profile_env_kwargs_compatibility_and_conflicts(gakucho_profile):
    env = make_generalization_env(2, env_kwargs={"avatar_profile": gakucho_profile}, randomize=False)
    try:
        env.reset(options={"scene_seed": 5})
        assert env.unwrapped.avatar_profile_sha256 == avatar_profile_sha256(gakucho_profile)
    finally:
        env.close()
    with pytest.raises(ValueError, match="supplied once"):
        make_generalization_env(2, avatar_profile=gakucho_profile,
                                env_kwargs={"avatar_profile": gakucho_profile})

    def wrong_factory(**kwargs):
        kwargs.pop("avatar_profile")
        return HumanoidRoomEnv(**kwargs)

    with pytest.raises(ValueError, match="preserve.*profile"):
        make_generalization_env(2, avatar_profile=gakucho_profile, env_factory=wrong_factory)


def test_avatar_sitting_and_upstairs_metrics_use_actual_body_offsets(gakucho_profile):
    env = make_generalization_env(2, task="mixed", avatar_profile=gakucho_profile, randomize=False)
    try:
        _, info = env.reset(options={"scene_seed": 37, "goal_id": "chair-seat"})
        base = env.unwrapped
        expected_height = env.selected_goal["position"][2] + base.seat_pelvis_offset
        assert base.seat_pelvis_offset == pytest.approx(gakucho_profile["physical"]["pelvis"]["seatOffset"], abs=1e-8)
        assert base.goal[2] == pytest.approx(expected_height)
        assert info["generalization"]["metrics"]["target_root_height_m"] == pytest.approx(expected_height)
    finally:
        env.close()
    upstairs = make_generalization_env(4, difficulty=.85, split="heldout", randomize=False,
                                      avatar_profile=gakucho_profile)
    try:
        _, info = upstairs.reset(options={"scene_seed": 1000007, "goal_id": "second-floor-target"})
        base = upstairs.unwrapped
        expected_height = upstairs.selected_goal["position"][2] + base.standing_root_height
        assert info["generalization"]["metrics"]["target_root_height_m"] == pytest.approx(expected_height)
        # Counterfactual planar arrival downstairs still fails a real upper goal.
        base.data.qpos[:2] = upstairs.selected_goal["position"][:2]
        base.data.qpos[2] = base.standing_root_height
        mujoco.mj_forward(base.model, base.data)
        raw = base._info()
        raw["success"] = True
        raw["scene_sensors"] = measure_scene(base)
        before = base.data.qpos.copy()
        inspected = upstairs._add_info(raw, advance=True)
        metrics = inspected["generalization"]["metrics"]
        assert metrics["root_height_error_m"] > 2.5
        assert metrics["goal_surface_height_m"] == upstairs.selected_goal["position"][2]
        assert not metrics["surface_level_matches"]
        assert not metrics["instant_success"] and not inspected["generalization"]["qualified_success"]
        np.testing.assert_array_equal(base.data.qpos, before)
    finally:
        upstairs.close()


def test_110_policy_requires_matching_avatar_and_body_before_prediction(gakucho_profile):
    incompatible = _ZeroPolicy(59)
    with pytest.raises(ValueError, match="110-input"):
        evaluate_generalization(incompatible, stages=[2], scene_seeds=[1000000], seconds=.02,
                                avatar_profile=gakucho_profile)
    assert incompatible.predictions == 0
    other_body = _ZeroPolicy(110)
    with pytest.raises(ValueError, match="body profile"):
        evaluate_generalization(other_body, stages=[2], scene_seeds=[1000000], seconds=.02,
                                avatar_profile=gakucho_profile)
    assert other_body.predictions == 0
    calibrated = make_generalization_env(2, avatar_profile=gakucho_profile, randomize=False)
    try:
        matching = _ZeroPolicy(110)
        bind_policy_body(matching, calibrated.unwrapped)
        report = evaluate_generalization(matching, stages=[2], scene_seeds=[1000000], seconds=.02,
                                         randomize=False, avatar_profile=gakucho_profile)
        assert matching.predictions == 1
        assert report["avatarProfileSha256"] == avatar_profile_sha256(gakucho_profile)
        assert report["bodyModelSha256"] == calibrated.unwrapped.body_model_sha256
        assert not report["proficiencyPassed"]
        matching.body_model_sha256 = "another-physics-body"
        with pytest.raises(ValueError, match="physical body contract"):
            evaluate_generalization(matching, stages=[2], scene_seeds=[1000000], seconds=.02,
                                    avatar_profile=gakucho_profile)
        assert matching.predictions == 1
    finally:
        calibrated.close()


def test_profile_verification_reports_same_body_without_opening_skill_gates(gakucho_profile):
    report = verify_generalization(stages=(2, 3, 4), episodes=1, seconds=.02,
                                   avatar_profile=gakucho_profile)
    assert report["passed"] and report["episode_count"] == 3
    assert not report["proficiencyPassed"]
    assert {episode["avatarProfileSha256"] for episode in report["episodes"]} == {
        avatar_profile_sha256(gakucho_profile)}
    assert len({episode["bodyModelSha256"] for episode in report["episodes"]}) == 1
    assert all(episode["external_force"] == 0 for episode in report["episodes"])


def test_measured_avatar_also_rebuilds_with_explicit_footstep_prior(gakucho_profile):
    env = make_generalization_env(4, difficulty=.25, motion_prior="footsteps", randomize=False,
                                  avatar_profile=gakucho_profile, max_episode_seconds=.02)
    try:
        observation, info = env.reset(options={"scene_seed": 52})
        assert observation.shape == (110,)
        assert type(env.unwrapped).__name__ == "FootstepHumanoidEnv"
        assert env.unwrapped.avatar_profile_sha256 == avatar_profile_sha256(gakucho_profile)
        assert not env.unwrapped.reference_start
        assert not info["generalization"]["terrain_prior_supported"]
        assert not info["generalization"]["proficiencyPassed"]
        env.step(np.zeros(20, np.float32))
        assert not np.any(env.unwrapped.data.xfrc_applied)
    finally:
        env.close()
