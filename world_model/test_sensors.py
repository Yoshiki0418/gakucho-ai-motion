"""Exact ray perception, whole-rig exclusion, and unchanged physical dynamics."""
from __future__ import annotations

import copy
import math
import unittest

import mujoco
import numpy as np

from .environment import HumanoidRoomEnv
from .scenes import convert_room_scene, generate_scene
from .sensors import SceneSensorWrapper, measure_scene


def _scene(extra=(), spawn=(0, 0, 0), heading=0):
    return {"version": 1, "stage": 1, "seed": 0, "name": "sensor-test",
            "geoms": [{"id": "floor", "type": "box", "position": [0, 0, -0.04],
                       "size": [10, 10, 0.08], "semantic": "floor"}, *extra],
            "spawn": list(spawn), "spawnHeading": heading, "goals": [], "metadata": {}}


def _box(identifier, centre, size, yaw=0):
    return {"id": identifier, "type": "box", "position": centre,
            "size": size, "yaw": yaw, "semantic": "furniture"}


class SensorTests(unittest.TestCase):
    def test_rotated_furniture_intersects_actual_obb_not_aabb(self):
        yaw = math.pi / 4
        scene = _scene([_box("desk", [0, -2, 0.5], [1, 0.4, 1], yaw)])
        env = HumanoidRoomEnv(scene=scene, randomize=False)
        try:
            measurement = measure_scene(env)
            actual = 2 - 0.2 / math.cos(yaw)
            aabb = 2 - (0.5 * math.sin(yaw) + 0.2 * math.cos(yaw))
            for layer in measurement["horizontal_ranges_m"]:
                self.assertAlmostEqual(layer[0], actual, places=8)
                self.assertGreater(layer[0], aabb + 0.2)
            geom = env.model.geom("desk").id
            self.assertEqual(measurement["horizontal_hit_geom_ids"][0][0], geom)
        finally:
            env.close()

    def test_ranges_rotate_with_actual_pelvis_yaw(self):
        scene = _scene([_box("east-wall", [2, 0, 1.4], [0.2, 8, 2.8])], heading=math.pi / 2)
        env = HumanoidRoomEnv(scene=scene, randomize=False)
        try:
            measurement = measure_scene(env)
            self.assertAlmostEqual(measurement["body_basis"]["yaw_radians"], math.pi / 2)
            np.testing.assert_allclose(measurement["body_basis"]["forward_xy"], [1, 0], atol=1e-8)
            self.assertAlmostEqual(measurement["horizontal_ranges_m"][0][0], 1.9, places=8)
        finally:
            env.close()

    def test_randomized_room_front_wall_is_observed(self):
        distances = []
        for seed in range(8):
            scene = generate_scene(2, seed)
            scene["geoms"] = [g for g in scene["geoms"] if g["semantic"] in ("floor", "wall")]
            env = HumanoidRoomEnv(scene=scene, randomize=False)
            try:
                measured = measure_scene(env)["horizontal_ranges_m"][0][0]
                depth_half = scene["metadata"]["floorBounds"][3]
                self.assertAlmostEqual(measured, depth_half, places=8)
                distances.append(measured)
            finally:
                env.close()
        self.assertGreater(max(distances) - min(distances), 1.0)

    def test_hole_has_missing_flags_and_no_assumed_floor(self):
        ring = lambda a: [{"x": x, "z": z} for x, z in [(-a, -a), (a, -a), (a, a), (-a, a)]]
        scene = convert_room_scene({"version": 1, "name": "hole", "floor": {"outline": ring(3), "holes": [ring(0.6)]},
                                    "obstacles": [], "destinations": [], "spawn": {"x": 1.5, "z": 0}})
        env = HumanoidRoomEnv(scene=scene, randomize=False)
        try:
            # Select a physical state above the missing floor; measurement must not repair it.
            env.data.qpos[:2] = 0
            mujoco.mj_forward(env.model, env.data)
            before = env.data.qpos.copy()
            measurement = measure_scene(env)
            self.assertEqual(measurement["floor_heights_m"], [None] * 9)
            self.assertEqual(measurement["floor_valid"], [False] * 9)
            self.assertEqual(measurement["missing_floor"], [True] * 9)
            self.assertEqual(measurement["features"][32:41], [-1.0] * 9)
            self.assertEqual(measurement["features"][41:50], [0.0] * 9)
            self.assertEqual(measurement["horizontal_reference"], "nominal_body_height_missing_surface")
            np.testing.assert_array_equal(env.data.qpos, before)
            self.assertFalse(np.any(env.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        finally:
            env.close()

    def test_floor_map_sees_real_step_in_front(self):
        step = _box("step", [0, -0.6, 0.1], [2, 0.8, 0.2])
        env = HumanoidRoomEnv(scene=_scene([step]), randomize=False)
        try:
            measurement = measure_scene(env)
            np.testing.assert_allclose(measurement["floor_heights_m"][:6], [0] * 6, atol=1e-8)
            np.testing.assert_allclose(measurement["floor_heights_m"][6:], [0.2] * 3, atol=1e-8)
            self.assertEqual(measurement["floor_valid"], [True] * 9)
            self.assertEqual(measurement["floor_hit_geom_ids"][7], env.model.geom("step").id)
            self.assertGreater(measurement["features"][39], measurement["features"][36])
        finally:
            env.close()

    def test_downward_ray_origin_lifts_out_of_actual_solid_only(self):
        step = _box("high-step", [0, 0, 0.5], [2, 2, 1])
        env = HumanoidRoomEnv(scene=_scene([step]), randomize=False)
        try:
            # The root is deliberately selected inside this failed-contact state.
            env.data.qpos[2] = 0.3
            mujoco.mj_forward(env.model, env.data)
            before = env.data.qpos.copy()
            measurement = measure_scene(env)
            self.assertTrue(all(measurement["floor_origin_lifted"]))
            self.assertTrue(all(p[2] > 1 for p in measurement["floor_origins_m"]))
            np.testing.assert_allclose(measurement["floor_heights_m"], [1] * 9, atol=1e-8)
            np.testing.assert_array_equal(env.data.qpos, before)
        finally:
            env.close()

    def test_entire_own_rig_is_excluded(self):
        env = HumanoidRoomEnv(scene=_scene(), randomize=False)
        try:
            measurement = measure_scene(env)
            self.assertEqual(measurement["excluded_own_geom_count"], env.model.ngeom - 1)
            np.testing.assert_array_equal(measurement["horizontal_ranges_m"], np.full((2, 16), 5.0))
            self.assertEqual(measurement["headroom_m"], 3.0)
            self.assertTrue(all(g == env.model.geom("floor").id for g in measurement["floor_hit_geom_ids"]))
            # An unfiltered ray from knee height hits a shin; the scene-only ray does not.
            geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(env.model, env.data, np.array([0., 0., 0.4]), np.array([1., 0., 0.]),
                                     np.ones(6, dtype=np.uint8), True, -1, geom)
            self.assertTrue(0 <= distance < 0.2)
            self.assertTrue(env.model.geom(int(geom[0])).name.startswith("humanoid_"))
            self.assertEqual(measurement["horizontal_ranges_m"][1][4], 5.0)
        finally:
            env.close()

    def test_headroom_uses_actual_ceiling_surface(self):
        env = HumanoidRoomEnv(scene=_scene([_box("ceiling", [0, 0, 2.4], [2, 2, 0.2])]), randomize=False)
        try:
            measurement = measure_scene(env)
            expected = 2.3 - measurement["headroom_origin_m"][2]
            self.assertAlmostEqual(measurement["headroom_m"], expected, places=8)
            self.assertEqual(measurement["headroom_hit_geom_id"], env.model.geom("ceiling").id)
            self.assertAlmostEqual(measurement["features"][-1], expected / 3)
        finally:
            env.close()

    def test_measurement_does_not_change_state_or_forces(self):
        env = HumanoidRoomEnv(scene=generate_scene(2, 3), randomize=False)
        try:
            env.step(np.zeros(env.action_space.shape, np.float32))
            fields = ["qpos", "qvel", "qacc", "ctrl", "qfrc_applied", "xfrc_applied", "qfrc_bias",
                      "qfrc_passive", "qfrc_constraint", "qfrc_actuator"]
            saved = {field: getattr(env.data, field).copy() for field in fields}
            model_fields = ["geom_contype", "geom_conaffinity", "geom_size", "geom_pos", "body_mass", "body_inertia"]
            model_saved = {field: getattr(env.model, field).copy() for field in model_fields}
            sim_time = env.data.time
            first = measure_scene(env)
            self.assertEqual(first, measure_scene(env))
            for field, value in saved.items():
                np.testing.assert_array_equal(getattr(env.data, field), value, err_msg=field)
            for field, value in model_saved.items():
                np.testing.assert_array_equal(getattr(env.model, field), value, err_msg=field)
            self.assertEqual(env.data.time, sim_time)
        finally:
            env.close()

    def test_wrapper_prefix_bounds_and_physics_match_unwrapped_env(self):
        scene = generate_scene(1)
        raw = HumanoidRoomEnv(scene=scene, randomize=False, max_episode_seconds=2)
        wrapped = SceneSensorWrapper(HumanoidRoomEnv(scene=copy.deepcopy(scene), randomize=False, max_episode_seconds=2))
        try:
            old, _ = raw.reset(seed=14)
            new, info = wrapped.reset(seed=14)
            self.assertEqual(old.shape, (59,))
            self.assertEqual(new.shape, (110,))
            np.testing.assert_array_equal(new[:59], old)
            self.assertEqual(info["scene_sensors"]["feature_count"], 51)
            self.assertTrue(wrapped.observation_space.contains(new))
            rng = np.random.default_rng(9)
            for _ in range(15):
                action = rng.uniform(-0.08, 0.08, raw.action_space.shape).astype(np.float32)
                old, reward_a, done_a, trunc_a, _ = raw.step(action)
                new, reward_b, done_b, trunc_b, info = wrapped.step(action)
                np.testing.assert_array_equal(new[:59], old)
                np.testing.assert_array_equal(raw.data.qpos, wrapped.unwrapped.data.qpos)
                self.assertEqual((reward_a, done_a, trunc_a), (reward_b, done_b, trunc_b))
                self.assertTrue(np.isfinite(new).all())
                self.assertTrue(wrapped.observation_space.contains(new))
                self.assertTrue(np.all(new[59:91] >= 0) and np.all(new[59:91] <= 1))
                self.assertTrue(np.all(new[91:100] >= -1) and np.all(new[91:100] <= 1))
                self.assertTrue(np.all(new[100:] >= 0) and np.all(new[100:] <= 1))
        finally:
            raw.close()
            wrapped.close()


if __name__ == "__main__":
    unittest.main()
