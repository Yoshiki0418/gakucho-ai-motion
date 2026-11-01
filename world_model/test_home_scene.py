"""Verify physical home proxies, measured seats, open leg space and reset contacts."""
from __future__ import annotations

import copy
import math
import unittest

import mujoco
import numpy as np

from .environment import HumanoidRoomEnv
from .home_scene import HOME_CHAIR_SEAT_HEIGHT, create_home_scene, create_seated_task_scene
from .test_scenes import _drop


def _human_furniture_penetrations(env):
    result = []
    mujoco.mj_forward(env.model, env.data)
    for contact in env.data.contact:
        a, b = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                for i in (contact.geom1, contact.geom2)]
        human_a, human_b = a.startswith("humanoid_"), b.startswith("humanoid_")
        if human_a == human_b:
            continue
        static = b if human_a else a
        if static not in ("floor", "home-floor") and contact.dist < -1e-5:
            result.append((a, b, float(contact.dist)))
    return result


class HomeSceneTests(unittest.TestCase):
    def test_major_furniture_and_up_axis_match_displayed_home(self):
        scene = create_home_scene()
        self.assertEqual(scene, create_home_scene())
        self.assertTrue(scene["metadata"]["contactProxies"])
        self.assertTrue(scene["metadata"]["approximations"])
        by_id = {g["id"]: g for g in scene["geoms"]}
        floor = by_id["home-floor"]
        self.assertEqual(floor["size"][:2], [10.12, 10.12])
        self.assertEqual(floor["position"][2] + floor["size"][2] / 2, 0)
        island = by_id["kitchen-island/counter"]
        self.assertEqual(island["position"], [-3.04, 1.28, 0.925])
        self.assertEqual(island["size"], [1.24, 2.23, 0.065])
        self.assertAlmostEqual(island["position"][2] + island["size"][2] / 2, 0.9575)
        self.assertEqual(by_id["refrigerator"]["position"], [-4.61, 4.45, 1.175])
        self.assertEqual(by_id["dining-table/top"]["position"], [-3.25, -2.42, 0.765])
        chaise = by_id["sofa/chaise-base"]
        self.assertAlmostEqual(chaise["position"][0], 3.19)
        self.assertAlmostEqual(chaise["position"][1], 3.32)
        self.assertEqual(chaise["yaw"], math.pi / 2)

    def test_all_four_seats_keep_measured_visible_centres(self):
        scene = create_home_scene()
        expected = {
            "dining-chair-wall-back": [-4.196504009806554, -1.782983924161362, HOME_CHAIR_SEAT_HEIGHT],
            "dining-chair-wall-front": [-4.196504009806554, -3.0429839241613617, HOME_CHAIR_SEAT_HEIGHT],
            "dining-chair-aisle-back": [-2.303495990193447, -1.7970160758386382, HOME_CHAIR_SEAT_HEIGHT],
            "dining-chair-aisle-front": [-2.303495990193447, -3.0570160758386384, HOME_CHAIR_SEAT_HEIGHT],
        }
        seats = [g for g in scene["goals"] if g["kind"] == "sit"]
        self.assertEqual(len(seats), 4)
        for goal in seats:
            np.testing.assert_allclose(goal["position"], expected[goal["entityId"]], atol=1e-12)
            geom = next(g for g in scene["geoms"] if g["id"] == f"{goal['entityId']}/seat")
            np.testing.assert_allclose(geom["position"][:2], goal["position"][:2], atol=1e-12)
            self.assertAlmostEqual(geom["position"][2] + geom["size"][2] / 2, HOME_CHAIR_SEAT_HEIGHT)
            p = goal["position"]
            _, data, contacts = _drop(scene, [p[0], p[1], p[2] + 0.10])
            self.assertAlmostEqual(data.qpos[2], p[2] + 0.025, delta=0.001)
            self.assertTrue(any(g["id"] == geom["id"] for g in contacts))

    def test_table_and_chair_leg_space_is_physically_open(self):
        scene = create_home_scene()
        # Under the tabletop reaches the floor rather than an artificial filled envelope.
        _, under, contacts = _drop(scene, [-3.25, -2.42, 0.15])
        self.assertAlmostEqual(under.qpos[2], 0.025, delta=0.001)
        self.assertTrue(all(g["id"] == "home-floor" for g in contacts))
        _, over, contacts = _drop(scene, [-3.25, -2.42, 0.90])
        self.assertAlmostEqual(over.qpos[2], 0.825, delta=0.001)
        self.assertTrue(any(g["id"] == "dining-table/top" for g in contacts))
        for goal in scene["goals"]:
            if goal["kind"] == "sit":
                x, y, _ = goal["position"]
                _, under, contacts = _drop(scene, [x, y, 0.15])
                self.assertAlmostEqual(under.qpos[2], 0.025, delta=0.001)
                self.assertTrue(all(g["id"] == "home-floor" for g in contacts))

    def test_home_spawn_and_three_standing_destinations_do_not_interpenetrate(self):
        scene = create_home_scene()
        for position in [scene["spawn"], *[g["position"] for g in scene["goals"] if g["kind"] == "walk"]]:
            reset_scene = copy.deepcopy(scene)
            reset_scene["spawn"] = position
            env = HumanoidRoomEnv(scene=reset_scene, randomize=False, max_episode_seconds=1)
            try:
                env.reset(seed=31)
                self.assertFalse(_human_furniture_penetrations(env), (position, _human_furniture_penetrations(env)))
                for _ in range(50):
                    _, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape, np.float32))
                    if terminated:
                        break
                self.assertFalse(terminated, position)
                self.assertTrue(np.isfinite(env.data.qpos).all())
                self.assertGreater(env.data.ncon, 0)
                self.assertFalse(_human_furniture_penetrations(env), (position, _human_furniture_penetrations(env)))
            finally:
                env.close()

    def test_near_chair_has_collision_free_initial_humanoid_and_real_seat(self):
        for distance in (0.28, 0.30, 0.35, 0.45):
            scene = create_seated_task_scene(distance)
            np.testing.assert_allclose(scene["goals"][0]["position"], [0, distance, HOME_CHAIR_SEAT_HEIGHT])
            self.assertFalse(scene["metadata"]["navigationPrerequisite"])
            env = HumanoidRoomEnv(task="sit", scene=scene, randomize=False)
            try:
                env.reset(seed=0)
                self.assertFalse(_human_furniture_penetrations(env), (distance, _human_furniture_penetrations(env)))
                self.assertEqual(env.model.nmocap, 0)
                self.assertEqual(env.model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
                np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -9.81])
                self.assertAlmostEqual(env.goal[2], HOME_CHAIR_SEAT_HEIGHT + 0.09)
                self.assertTrue(env._seat_geom_ids)
                # Geometry remains fixed while the unpowered body responds to gravity.
                static_before = env.model.geom_pos.copy()
                for _ in range(150):
                    env.data.ctrl[:] = 0
                    mujoco.mj_step(env.model, env.data)
                np.testing.assert_array_equal(env.model.geom_pos, static_before)
            finally:
                env.close()
        with self.assertRaises(ValueError):
            create_seated_task_scene(0.10)


if __name__ == "__main__":
    unittest.main()
