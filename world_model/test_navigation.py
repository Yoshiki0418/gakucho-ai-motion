"""Verify real blocked paths, input contracts and full-substep collision audits."""
import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from .navigation import (EpisodeNavigationEnv, NAVIGATION_OBSERVATION_CONTRACT,
                         NAVIGATION_SUCCESS_BONUS, OBSTACLE_FAILURE_TERMINAL_COST, OBSTACLE_REWARD_FLOOR)
from .navigation_scenes import (MINIMUM_PASSAGE_WIDTH, NavigationPotential,
                                CURRICULUM_INTERSECTION_INSET,
                                generate_navigation_scene, segment_intersects_box)
from .scenes import box_corners, point_box_distance
from .sensors import measure_scene
from .steering import SteeringHumanoidEnv

PROFILE_PATH = Path(__file__).resolve().parents[1] / "frontend/public/world-model/gakucho-avatar-profile.json"


@pytest.mark.parametrize("difficulty", [0., .15, .6, 1.])
def test_procedural_obstacles_really_block_every_direct_route_and_have_body_clearance(difficulty):
    shapes, friction, wider_sides = set(), set(), set()
    for seed in range(30):
        scene = generate_navigation_scene(seed, difficulty=difficulty)
        furniture = [g for g in scene["geoms"] if g.get("semantic") == "furniture"]
        assert len(furniture) == 1
        obstacle = furniture[0]
        start, goal = scene["spawn"], scene["goals"][0]["position"]
        assert 2 <= math.dist(start, goal) <= 3.4
        assert segment_intersects_box(start, goal, obstacle)
        # The initial forward sensor ray must also see the box; a narrow
        # crossing only between the 16 sensor rays gives no avoidance cue.
        assert segment_intersects_box(start, [start[0], start[1] - 5.], obstacle)
        assert scene["metadata"]["obstacleBlockedDirectRoute"]
        assert scene["metadata"]["directRouteBlocked"]
        assert scene["metadata"]["minimumRequiredPassageWidthMeters"] == MINIMUM_PASSAGE_WIDTH
        bounds = scene["metadata"]["floorBounds"]
        corners = box_corners(obstacle)
        left = min(x for x, _ in corners) - bounds[0]
        right = bounds[2] - max(x for x, _ in corners)
        assert left >= MINIMUM_PASSAGE_WIDTH and right >= MINIMUM_PASSAGE_WIDTH
        assert left == pytest.approx(scene["metadata"]["leftPassageWidthMeters"])
        assert right == pytest.approx(scene["metadata"]["rightPassageWidthMeters"])
        assert point_box_distance(start, obstacle) >= .55
        assert point_box_distance(goal, obstacle) >= .55
        assert not scene["metadata"]["waypointController"]
        assert not scene["metadata"]["proficiencyPassed"]
        assert scene == generate_navigation_scene(seed, difficulty=difficulty)
        shapes.add(tuple(obstacle["size"]) + (obstacle["yaw"],))
        friction.add(scene["metadata"]["floorFriction"])
        wider_sides.add(left > right)
    assert len(shapes) == 30 and len(friction) == 30 and len(wider_sides) == 2


def test_turn_curriculum_has_no_furniture_and_only_its_real_final_goal():
    scenes = [generate_navigation_scene(seed, mode="turn", difficulty=1.) for seed in range(12)]
    assert all(not scene["metadata"]["obstacleBlockedDirectRoute"] for scene in scenes)
    assert all(len(scene["goals"]) == 1 for scene in scenes)
    assert all(not any(g.get("semantic") == "furniture" for g in scene["geoms"]) for scene in scenes)
    assert any(scene["goals"][0]["position"][0] < 0 for scene in scenes)
    assert any(scene["goals"][0]["position"][0] > 0 for scene in scenes)


def test_easy_obstacle_curriculum_extends_the_approach_and_offsets_both_sides():
    sides = set()
    for seed in range(30):
        scene = generate_navigation_scene(seed, difficulty=0.)
        metadata = scene["metadata"]
        assert 3.1 <= metadata["directRouteLengthMeters"] <= 3.4
        assert .55 <= metadata["furnitureRouteFraction"] <= .60
        assert metadata["curriculumBlend"] == 1.
        assert metadata["navigationCurriculum"] == "edge_intersection_extended_approach_v1"
        assert metadata["directLineIntersectionInsetMeters"] == pytest.approx(CURRICULUM_INTERSECTION_INSET)
        assert abs(metadata["furnitureLineNormalOffsetMeters"]) > .1
        assert not metadata["initialGoalAngleChanged"]
        assert abs(metadata["initialGoalAngleRadians"]) <= .025
        sides.add(metadata["furnitureLineNormalOffsetMeters"] > 0)
    assert sides == {False, True}
    hard = generate_navigation_scene(7, difficulty=1.)
    assert "navigationCurriculum" not in hard["metadata"]
    assert "furnitureLineNormalOffsetMeters" not in hard["metadata"]
    assert hard["metadata"]["directRouteLengthMeters"] < 3.


def test_large_rotated_furniture_keeps_clear_spawn_and_goal_approaches():
    # The previous high-difficulty seed placed this furniture 4mm too close
    # to the goal. Its static centre is now adjusted before physics exists.
    scene = generate_navigation_scene(367, difficulty=1.)
    obstacle = scene["geoms"][-1]
    start, goal = scene["spawn"], scene["goals"][0]["position"]
    assert obstacle["position"][:2] == [(start[axis] + goal[axis]) / 2 for axis in range(2)]
    assert point_box_distance(start, obstacle) == pytest.approx(.6244102917821844)
    assert point_box_distance(goal, obstacle) == pytest.approx(.6244102917821844)
    assert segment_intersects_box(start, goal, obstacle)
    assert scene["metadata"]["leftPassageWidthMeters"] >= MINIMUM_PASSAGE_WIDTH
    assert scene["metadata"]["rightPassageWidthMeters"] >= MINIMUM_PASSAGE_WIDTH


def test_rotated_rectangle_blocking_and_reward_distance_do_not_reward_penetration():
    obstacle = {"position": [0., 0., .5], "size": [.6, .4, 1.], "yaw": .35}
    scene = {"geoms": [{**obstacle, "entityId": "navigation-furniture"}],
             "goals": [{"position": [0., -1.2, 0.]}]}
    potential = NavigationPotential(scene)
    assert segment_intersects_box([0, 1.2], [0, -1.2], obstacle)
    assert not segment_intersects_box([1.3, 1.2], [1.3, -1.2], obstacle)
    assert potential.distance([0, 1.2]) > 2.4
    # Penetrating the inflated body envelope creates worse potential even
    # though the Euclidean goal distance becomes smaller.
    outside = [0., .85]
    inside = [0., .40]
    assert math.dist(inside, potential.goal) < math.dist(outside, potential.goal)
    assert potential.distance(inside) > potential.distance(outside)
    assert potential.distance([1.3, -1.2]) == pytest.approx(1.3)


@pytest.mark.parametrize("mode", ["turn", "obstacle"])
def test_rebuilds_are_deterministic_preserve_110_prefix_and_measured_body(mode):
    first = EpisodeNavigationEnv(mode=mode, base_seed=20261015, randomize=False,
                                 avatar_profile=PROFILE_PATH, difficulty_range=(0., .6))
    second = EpisodeNavigationEnv(mode=mode, base_seed=20261015, randomize=False,
                                  avatar_profile=PROFILE_PATH, difficulty_range=(0., .6))
    hashes, models = [], []
    try:
        for _ in range(4):
            observation, info = first.reset()
            other, other_info = second.reset()
            np.testing.assert_array_equal(observation, other)
            assert info == other_info
            assert observation.shape == (110,)
            assert first.observation_space.contains(observation)
            assert first.action_space.shape == (20,)
            assert first.unwrapped.observation_space.shape == (59,)
            assert first.avatar_profile_sha256 == first.unwrapped.avatar_profile_sha256
            assert first.body_model_sha256 == "c71ad1578106b2cf0754347070482371cf419ca6ec75bc9af1c22428c5bb339c"
            assert info["navigation"]["policy_input_contract"] == NAVIGATION_OBSERVATION_CONTRACT
            np.testing.assert_array_equal(first.unwrapped.goal, first.current_scene["goals"][0]["position"])
            np.testing.assert_array_equal(first.unwrapped.data.qpos[:2], first.current_scene["spawn"][:2])
            assert not np.any(first.unwrapped.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            hashes.append(info["navigation"]["scene_sha256"])
            models.append(first.unwrapped.model)
        assert len(set(hashes)) == 4
        assert all(models[i] is not models[j] for i in range(4) for j in range(i))
        one, one_info = first.reset(seed=98)
        two, two_info = first.reset(seed=98)
        np.testing.assert_array_equal(one, two)
        # The monotonic bookkeeping index deliberately counts both resets.
        one_info["navigation"].pop("episode_index")
        two_info["navigation"].pop("episode_index")
        assert one_info == two_info
    finally:
        first.close()
        second.close()


def test_sensor_reward_and_substep_audit_never_modify_physics_or_goal():
    wrapped = EpisodeNavigationEnv(randomize=False, avatar_profile=PROFILE_PATH,
                                   max_episode_seconds=.3, difficulty_range=(.15, .15))
    observation, _ = wrapped.reset(options={"scene_seed": 85, "physics_seed": 18})
    direct = SteeringHumanoidEnv(task="walk", scene=copy.deepcopy(wrapped.current_scene),
                                 randomize=False, avatar_profile=PROFILE_PATH,
                                 strict_gait_reward=True, max_episode_seconds=.3)
    direct_observation, _ = direct.reset(seed=18)
    try:
        np.testing.assert_array_equal(observation[:59], direct_observation)
        callback_count = 0
        actual_observer = wrapped.unwrapped._physics_contact_observer

        def observer(model, data):
            nonlocal callback_count
            callback_count += 1
            qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
            actual_observer(model, data)
            np.testing.assert_array_equal(data.qpos, qpos)
            np.testing.assert_array_equal(data.qvel, qvel)
            np.testing.assert_array_equal(data.ctrl, ctrl)

        wrapped.unwrapped._physics_contact_observer = observer
        goal = wrapped.unwrapped.goal.copy()
        for action in np.random.default_rng(11).uniform(-.01, .01, (10, 20)):
            observation, reward, done, truncated, info = wrapped.step(action)
            expected, _, direct_done, direct_truncated, _ = direct.step(action)
            np.testing.assert_array_equal(observation[:59], expected)
            np.testing.assert_array_equal(wrapped.unwrapped.data.qpos, direct.data.qpos)
            np.testing.assert_array_equal(wrapped.unwrapped.data.qvel, direct.data.qvel)
            np.testing.assert_array_equal(wrapped.unwrapped.goal, goal)
            assert (done, truncated) == (direct_done, direct_truncated)
            assert np.isfinite(reward) and np.isfinite(observation).all()
            assert not np.any(wrapped.unwrapped.data.xfrc_applied)
            assert not np.any(wrapped.unwrapped.data.qfrc_applied)
            assert not info["navigation"]["collision_occurred"]
        assert callback_count == 100
        before = wrapped.unwrapped.data.qpos.copy()
        snapshot = wrapped.snapshot()
        np.testing.assert_array_equal(wrapped.unwrapped.data.qpos, before)
        assert snapshot["scene"] == wrapped.current_scene
        assert len(snapshot["bodies"]) == 13
        assert snapshot["stats"]["navigation"]["collision_audit_timestep_seconds"] == .002
        assert snapshot["stats"]["navigation"]["minimum_floor_support_newtons_per_foot"] == 20.
        assert snapshot["stats"]["navigation_floor_support"] == info["navigation_floor_support"]
        json.dumps(snapshot, allow_nan=False)
    finally:
        direct.close()
        wrapped.close()


@pytest.mark.parametrize("split,valid,invalid", [
    ("train", [0, 999999], [-1, 1000000, 2000000, True]),
    ("heldout", [1100000, 1900031], [-1, 0, 999999, 2000000, True]),
])
def test_train_and_heldout_seeds_cannot_leak_and_invalid_reset_keeps_state(split, valid, invalid):
    env = EpisodeNavigationEnv(split=split, randomize=False, avatar_profile=PROFILE_PATH)
    try:
        for seed in valid:
            _, info = env.reset(options={"scene_seed": seed, "physics_seed": 37})
            assert info["navigation"]["sceneSeed"] == seed
        model = env.unwrapped.model
        qpos, qvel, time = env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy(), env.unwrapped.data.time
        for seed in invalid:
            with pytest.raises(ValueError):
                env.reset(options={"scene_seed": seed})
            assert env.unwrapped.model is model
            np.testing.assert_array_equal(env.unwrapped.data.qpos, qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, qvel)
            assert env.unwrapped.data.time == time
        with pytest.raises(ValueError):
            env.reset(options={"split": "train"})
        with pytest.raises(AttributeError):
            env.split = "train"
    finally:
        env.close()


def test_actual_transient_contact_cannot_disappear_from_qualified_success():
    env = EpisodeNavigationEnv(randomize=False, avatar_profile=PROFILE_PATH)
    _, _ = env.reset(options={"scene_seed": 37, "physics_seed": 9})
    base = env.unwrapped
    try:
        # A second actual MuJoCo state is the transient substep witness. The
        # main rollout state remains untouched, and has no furniture contact.
        transient = mujoco.MjData(base.model)
        transient.qpos[:] = base.data.qpos
        furniture = env._potential.obstacle
        transient.qpos[:2] = furniture["position"][:2]
        mujoco.mj_forward(base.model, transient)
        before = base.data.qpos.copy()
        env._observe_physics_contacts(base.model, transient)
        np.testing.assert_array_equal(base.data.qpos, before)
        assert env._collision_occurred
        assert env._collision_substeps == 1
        assert env._collision_seconds == .002
        assert env._collision_force_peak > 1
        assert env._collision_geom_ids == {"navigation-furniture"}
        # Counterfactual final arrival: real floor supports both shoes, but the
        # previous transient collision still disqualifies the whole episode.
        base.data.qpos[:2] = base.goal[:2]
        base.data.qpos[2] = base.standing_root_height - .003
        base.data.qvel[:] = 0.
        mujoco.mj_forward(base.model, base.data)
        info = base._info()
        info["scene_sensors"] = measure_scene(base)
        inspected = env._add_info(info, advance=False)
        assert inspected["navigation_floor_support"] == [True, True]
        assert inspected["navigation"]["metrics"]["goal_distance_m"] == 0
        assert inspected["navigation"]["metrics"]["root_speed_m_s"] == 0
        assert not inspected["navigation"]["metrics"]["instant_success"]
        assert not inspected["success"]
        assert inspected["collision_occurred"]
    finally:
        env.close()


def test_success_requires_half_second_of_stopped_real_floor_support_and_is_read_only():
    env = EpisodeNavigationEnv(mode="turn", randomize=False, avatar_profile=PROFILE_PATH)
    env.reset(options={"scene_seed": 19, "physics_seed": 9})
    base = env.unwrapped
    try:
        base.data.qpos[:2] = base.goal[:2]
        base.data.qpos[2] = base.standing_root_height - .003
        base.data.qvel[:] = 0.
        mujoco.mj_forward(base.model, base.data)
        info = base._info()
        info["scene_sensors"] = measure_scene(base)
        qpos, qvel = base.data.qpos.copy(), base.data.qvel.copy()
        for _ in range(24):
            result = env._add_info(info, advance=True)
            assert not result["success"]
        result = env._add_info(info, advance=True)
        assert result["success"]
        assert result["navigation_success_seconds"] == pytest.approx(.5)
        for _ in range(2):
            env._add_info(info, advance=False)
        assert env._qualified_success_seconds == pytest.approx(.5)
        np.testing.assert_array_equal(base.data.qpos, qpos)
        np.testing.assert_array_equal(base.data.qvel, qvel)
        base.data.qvel[:3] = [.3, 0., 0.]
        moving = base._info()
        moving["scene_sensors"] = measure_scene(base)
        assert not env._add_info(moving, advance=True)["success"]
        assert env._qualified_success_seconds == 0.
    finally:
        env.close()


def test_first_real_collision_ends_the_task_without_faking_a_physical_fall():
    env = EpisodeNavigationEnv(randomize=False, avatar_profile=PROFILE_PATH)
    env.reset(options={"scene_seed": 37, "physics_seed": 9})
    base = env.unwrapped
    direct = SteeringHumanoidEnv(task="walk", scene=copy.deepcopy(env.current_scene),
                                 randomize=False, avatar_profile=PROFILE_PATH,
                                 strict_gait_reward=True, max_episode_seconds=30.)
    direct.reset(seed=9)
    try:
        # Set up identical upright states whose shoes touch the furniture.
        # The ensuing contact force and state change come from actual mj_step.
        obstacle = env._potential.obstacle
        base.data.qpos[:2] = [obstacle["position"][0], obstacle["position"][1]
                              + obstacle["size"][1] / (2 * math.cos(obstacle["yaw"])) + .18]
        direct.data.qpos[:] = base.data.qpos
        direct.data.qvel[:] = base.data.qvel
        mujoco.mj_forward(base.model, base.data)
        mujoco.mj_forward(direct.model, direct.data)
        action = np.zeros(20, np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        _, _, physical_done, _, physical_info = direct.step(action)
        np.testing.assert_array_equal(base.data.qpos, direct.data.qpos)
        np.testing.assert_array_equal(base.data.qvel, direct.data.qvel)
        assert terminated and not truncated
        assert not physical_done and not physical_info["fall"] and not info["fall"]
        assert info["collision_occurred"]
        assert info["navigation"]["metrics"]["obstacle_contact_substeps"] > 0
        assert info["navigation"]["metrics"]["peak_collision_force_newtons"] > 1.
        assert not info["success"]
        terms = info["navigation"]["reward"]
        assert terms["terminal_reason"] == "collision"
        assert terms["terminal_cost"] == OBSTACLE_FAILURE_TERMINAL_COST
        assert reward == pytest.approx(terms["unclamped_reward"]
                                       + terms["reward_clamp_adjustment"] - terms["terminal_cost"])
        assert terms["unclamped_reward"] + terms["reward_clamp_adjustment"] >= OBSTACLE_REWARD_FLOOR
        before, time = base.data.qpos.copy(), base.data.time
        snapshot = env.snapshot()
        assert snapshot["stats"]["navigation"]["terminal_reason"] == "collision"
        assert snapshot["stats"]["navigation"]["reward"] == terms
        np.testing.assert_array_equal(base.data.qpos, before)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)
        np.testing.assert_array_equal(base.data.qpos, before)
        assert base.data.time == time
    finally:
        direct.close()
        env.close()


@pytest.mark.parametrize("mode", ["turn", "obstacle"])
def test_real_fall_cost_is_outside_the_floor_and_worse_than_negative_waiting_value(mode):
    env = EpisodeNavigationEnv(mode=mode, randomize=False, avatar_profile=PROFILE_PATH)
    env.reset(options={"scene_seed": 37, "physics_seed": 9})
    base = env.unwrapped
    try:
        mujoco.mju_axisAngle2Quat(base.data.qpos[3:7], np.asarray([0., 1., 0.]), 1.4)
        mujoco.mj_forward(base.model, base.data)
        _, reward, terminated, truncated, info = env.step(np.zeros(20, np.float32))
        assert terminated and not truncated and info["fall"]
        assert not info["collision_occurred"]
        terms = info["navigation"]["reward"]
        assert terms["terminal_reason"] == "fall"
        assert terms["terminal_cost"] == OBSTACLE_FAILURE_TERMINAL_COST
        assert terms["reward_clamp_adjustment"] > 0
        assert reward == pytest.approx(OBSTACLE_REWARD_FLOOR - OBSTACLE_FAILURE_TERMINAL_COST)
        gamma = terms["failure_bound_discount_factor"]
        worst_continuing_value = OBSTACLE_REWARD_FLOOR / (1 - gamma)
        assert gamma == .99 and worst_continuing_value == pytest.approx(-800.)
        assert reward < worst_continuing_value
        # Even the conservative maximum inherited posture/gait/progress credit
        # cannot offset this terminal cost relative to the continuing bound.
        maximum_nonterminal_credit = .5 + .5 + .4 + 4 + 16 + 24 + 20 - 6
        assert maximum_nonterminal_credit - OBSTACLE_FAILURE_TERMINAL_COST < worst_continuing_value
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["turn", "obstacle"])
def test_nonterminal_navigation_reward_has_a_finite_lower_bound_and_timeout_is_distinct(mode):
    env = EpisodeNavigationEnv(mode=mode, randomize=False, avatar_profile=PROFILE_PATH,
                               max_episode_seconds=.2)
    env.reset(options={"scene_seed": 37, "physics_seed": 9})
    try:
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(np.zeros(20, np.float32))
            assert reward >= OBSTACLE_REWARD_FLOOR
            assert not terminated
            assert info["navigation"]["reward"]["terminal_cost"] == 0
        assert truncated
        assert info["navigation"]["terminal_reason"] == "time_limit"
        assert not info["fall"] and not info["success"]
    finally:
        env.close()


def test_turn_task_preserves_low_level_footstep_speed_and_gait_reward_and_dynamics():
    env = EpisodeNavigationEnv(mode="turn", randomize=False, avatar_profile=PROFILE_PATH,
                               max_episode_seconds=.3)
    observation, _ = env.reset(options={"scene_seed": 37, "physics_seed": 9})
    base = SteeringHumanoidEnv(task="walk", scene=copy.deepcopy(env.current_scene),
                               randomize=False, avatar_profile=PROFILE_PATH,
                               strict_gait_reward=True, max_episode_seconds=.3)
    base_observation, _ = base.reset(seed=9)
    try:
        np.testing.assert_array_equal(observation[:59], base_observation)
        for action in np.random.default_rng(93).uniform(-.01, .01, size=(10, 20)):
            observation, reward, terminated, truncated, info = env.step(action)
            expected, inherited_reward, physical_done, physical_truncated, _ = base.step(action)
            np.testing.assert_array_equal(observation[:59], expected)
            np.testing.assert_array_equal(env.unwrapped.data.qpos, base.data.qpos)
            np.testing.assert_array_equal(env.unwrapped.data.qvel, base.data.qvel)
            assert (terminated, truncated) == (physical_done, physical_truncated)
            terms = info["navigation"]["reward"]
            assert terms["inherited_footstep_reward"] == inherited_reward
            assert terms["footstep_velocity_tracking_preserved"]
            assert terms["retained_goal_speed_tracking_reward"] > 0
            assert 0 <= terms["goal_speed_tracking_target_m_s"] <= .45
            assert terms["removed_straight_goal_progress"] == 0
            assert terms["removed_straight_goal_velocity"] == 0
            assert terms["removed_straight_goal_near"] == 0
            assert terms["geometry_potential_progress"] == 0
            assert terms["proximity_penalty"] == 0
            assert terms["time_penalty"] == 0
            assert terms["terminal_cost"] == 0 and terms["success_bonus"] == 0
            assert reward == max(OBSTACLE_REWARD_FLOOR, inherited_reward)
        assert env.snapshot()["stats"]["navigation"]["reward"] == terms
    finally:
        base.close()
        env.close()


@pytest.mark.parametrize("mode", ["turn", "obstacle"])
def test_stopped_supported_success_bonus_beats_near_goal_continuing_reward(mode):
    env = EpisodeNavigationEnv(mode=mode, randomize=False, avatar_profile=PROFILE_PATH)
    env.reset(options={"scene_seed": 37, "physics_seed": 9})
    base = env.unwrapped
    try:
        base.data.qpos[:2] = base.goal[:2]
        base.data.qpos[2] = base.standing_root_height - .003
        base.data.qvel[:] = 0.
        mujoco.mj_forward(base.model, base.data)
        # Represent the preceding 24 qualifying supported control frames;
        # this last real physics frame must complete the half-second hold.
        env._qualified_success_seconds = .48
        _, reward, terminated, truncated, info = env.step(np.zeros(20, np.float32))
        assert terminated and not truncated
        assert info["success"] and not info["fall"] and not info["collision_occurred"]
        assert info["navigation_floor_support"] == [True, True]
        terms = info["navigation"]["reward"]
        assert terms["terminal_reason"] == "success"
        assert terms["success_bonus"] == NAVIGATION_SUCCESS_BONUS == 1500
        assert terms["terminal_cost"] == 0
        assert reward >= NAVIGATION_SUCCESS_BONUS + OBSTACLE_REWARD_FLOOR
        # With stopped feet the strict-gait terms fade at the actual goal.
        # Remaining upright/height/imitation/speed/near/clearance credit is
        # bounded by 12.4, whose gamma=.99 continuing value is below 1500.
        stopped_goal_reward_upper_bound = .5 + .5 + .4 + 4 + 3 + 4
        assert NAVIGATION_SUCCESS_BONUS > stopped_goal_reward_upper_bound / (1 - .99)
        snapshot = env.snapshot()
        assert snapshot["stats"]["navigation"]["terminal_reason"] == "success"
        assert snapshot["stats"]["navigation"]["reward"] == terms
    finally:
        env.close()
