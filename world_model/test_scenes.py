"""Geometry and contact tests; these do not score or claim a learned policy."""
from __future__ import annotations

import json
import math
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scenes import (HELD_OUT_SEED_BASE, box_corners, convert_room_scene,
                     generate_scene, point_box_distance, y_up_to_z_up,
                     z_up_to_y_up)


def _room(outline, holes=(), obstacles=(), destinations=(), spawn=None):
    result = {"version": 1, "name": "contact test", "units": "m", "upAxis": "Y",
              "floor": {"outline": [{"x": x, "z": z} for x, z in outline],
                        "holes": [[{"x": x, "z": z} for x, z in h] for h in holes]},
              "obstacles": list(obstacles), "destinations": list(destinations)}
    if spawn:
        result["spawn"] = {"x": spawn[0], "z": spawn[1]}
    return result


def _rectangle(x0, z0, x1, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def _fmt(values):
    return " ".join(format(v, ".16g") for v in values)


def _contact_model(scene, point, radius=0.025):
    """Independent XML construction to ask the actual engine about floor holes."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    names = {}
    for i, geom in enumerate(scene["geoms"]):
        name = f"static-{i}"
        names[name] = geom
        args = {"name": name, "pos": _fmt(geom["position"]),
                "quat": _fmt([math.cos(geom.get("yaw", 0) / 2), 0, 0, math.sin(geom.get("yaw", 0) / 2)]),
                "friction": "0.9 0.005 0.0001"}
        if geom["type"] == "box":
            args.update(type="box", size=_fmt(np.array(geom["size"]) / 2))
        else:
            mesh = f"mesh-{i}"
            ET.SubElement(asset, "mesh", name=mesh,
                          vertex=_fmt([v for p in geom["vertices"] for v in p]),
                          face=" ".join(str(v) for p in geom["faces"] for v in p))
            args.update(type="mesh", mesh=mesh)
        ET.SubElement(world, "geom", **args)
    body = ET.SubElement(world, "body", name="probe-body", pos=_fmt(point))
    ET.SubElement(body, "freejoint", name="probe-root")
    ET.SubElement(body, "geom", name="probe", type="sphere", size=str(radius), mass="0.1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data, names


def _drop(scene, point, seconds=0.6):
    model, data, names = _contact_model(scene, point)
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    contacts = []
    for contact in data.contact:
        for index in (contact.geom1, contact.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            if name in names:
                contacts.append(names[name])
    return model, data, contacts


def _triangle_contains(p, triangle):
    signs = []
    for i, a in enumerate(triangle):
        b = triangle[(i + 1) % 3]
        signs.append((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
    return min(signs) >= -1e-8 or max(signs) <= 1e-8


def _covered(point, floor_geoms):
    for geom in floor_geoms:
        if geom["type"] == "box":
            if point_box_distance([*point, 0], geom) < 1e-8:
                return True
        else:
            x, y, _ = geom["position"]
            tri = [(x + v[0], y + v[1]) for v in geom["vertices"][:3]]
            if _triangle_contains(point, tri):
                return True
    return False


def _floor_area(scene):
    area = 0.0
    for geom in scene["geoms"]:
        if geom["semantic"] != "floor":
            continue
        if geom["type"] == "box":
            area += geom["size"][0] * geom["size"][1]
        else:
            a, b, c = geom["vertices"][:3]
            area += abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    return area


def _point_clear(point, geoms, radius=0.30):
    # Standing body envelope: floor/support contacts are allowed, tall furniture is not.
    for geom in geoms:
        if geom["type"] != "box" or geom["semantic"] == "floor":
            continue
        bottom = geom["position"][2] - geom["size"][2] / 2
        top = geom["position"][2] + geom["size"][2] / 2
        if top > point[2] + 0.08 and bottom < point[2] + 1.8 and point_box_distance(point, geom) < radius:
            return False
    return True


class SceneTests(unittest.TestCase):
    def test_deterministic_varied_generators_and_heldout_seeds(self):
        layouts = set()
        for stage in range(1, 5):
            for seed in list(range(20)) + [HELD_OUT_SEED_BASE + i for i in range(5)]:
                scene = generate_scene(stage, seed)
                self.assertEqual(scene, generate_scene(stage, seed))
                self.assertFalse(scene["metadata"]["infinitePlane"])
                self.assertEqual(scene["metadata"]["split"], "heldout" if seed >= HELD_OUT_SEED_BASE else "train")
                self.assertTrue(_point_clear(scene["spawn"], scene["geoms"]), (stage, seed, "spawn"))
                self.assertEqual(len(scene["geoms"]), len({g["id"] for g in scene["geoms"]}))
                for goal in scene["goals"]:
                    if goal["kind"] == "walk":
                        self.assertTrue(_point_clear(goal["position"], scene["geoms"]), (stage, seed, goal["id"]))
                if stage == 2:
                    layouts.add(json.dumps(scene["metadata"]["furnitureEnvelopes"], sort_keys=True))
                # Compile every generated physical world in the real engine.
                model, _, _ = _contact_model(scene, [*scene["spawn"][:2], scene["spawn"][2] + 0.3])
                self.assertFalse(np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
        self.assertEqual(len(layouts), 25)

    def test_stage_two_passage_and_no_overlap(self):
        for seed in range(30):
            scene = generate_scene(2, seed, 0.9)
            entities = scene["metadata"]["furnitureEnvelopes"]
            passage = scene["metadata"]["guaranteedPassage"]["width"]
            for x in np.linspace(scene["spawn"][0], scene["goals"][0]["position"][0], 121):
                for y in (-passage / 2 + 0.33, 0, passage / 2 - 0.33):
                    self.assertTrue(_point_clear([x, y, 0], scene["geoms"], 0.30), (seed, x, y))
            for i, a in enumerate(entities):
                for b in entities[i + 1:]:
                    # Independent separating-axis overlap check on world vertices.
                    separated = False
                    ac, bc = box_corners(a), box_corners(b)
                    for polygon in (ac, bc):
                        for j, p in enumerate(polygon):
                            q = polygon[(j + 1) % len(polygon)]
                            axis = (q[1] - p[1], p[0] - q[0])
                            ap = [x * axis[0] + y * axis[1] for x, y in ac]
                            bp = [x * axis[0] + y * axis[1] for x, y in bc]
                            separated |= max(ap) < min(bp) or max(bp) < min(ap)
                    self.assertTrue(separated, (seed, a["id"], b["id"]))

    def test_mult_rooms_connected_through_actual_door_gaps(self):
        for seed in range(20):
            scene = generate_scene(3, seed, 0.9)
            spawn = scene["spawn"]
            for goal, portal in zip(scene["goals"][:4], scene["metadata"]["portals"]):
                target = goal["position"]
                # Actual geometry provides a route along the corridor and through each door.
                route = [spawn, [portal["position"][0], 0, 0], portal["position"], target]
                for a, b in zip(route, route[1:]):
                    for t in np.linspace(0, 1, 61):
                        point = np.array(a) * (1 - t) + np.array(b) * t
                        self.assertTrue(_point_clear(point, scene["geoms"]), (seed, goal["id"], point.tolist()))

    def test_stage_four_has_real_support_at_all_three_levels(self):
        for difficulty, expected in [(0.1, "low-step"), (0.5, "short-stairs"), (0.9, "stairs-and-second-floor")]:
            scene = generate_scene(4, 7, difficulty)
            self.assertEqual(scene["metadata"]["terrain"], expected)
            goal = scene["goals"][0]["position"]
            model, data, contacts = _drop(scene, [goal[0], goal[1], goal[2] + 0.12])
            self.assertAlmostEqual(data.qpos[2], goal[2] + 0.025, delta=0.001)
            self.assertTrue(contacts)
            if difficulty > 2 / 3:
                slab = next(g for g in scene["geoms"] if g["id"] == "second-floor")
                self.assertAlmostEqual(slab["size"][2], 0.20)
                self.assertGreater(slab["position"][2] - slab["size"][2] / 2, 0.5)
                self.assertEqual(len(scene["metadata"]["treadCentres"]), 16)
                self.assertTrue(2.5 <= scene["metadata"]["upperFloorHeight"] <= 3.0)
                self.assertGreaterEqual(slab["position"][2] - slab["size"][2] / 2, 2.3)
                # The lower room below the slab is real open space and supports a falling body.
                _, below, contacts = _drop(scene, [goal[0], goal[1], 0.15])
                self.assertAlmostEqual(below.qpos[2], 0.025, delta=0.001)
                self.assertTrue(all(g["semantic"] == "floor" for g in contacts))

    def test_random_spawns_and_goals_have_actual_support_contacts(self):
        for stage in range(1, 5):
            for seed in range(20):
                difficulty = 0.9 if stage == 4 else 0.5
                scene = generate_scene(stage, seed, difficulty)
                for point, kind in [(scene["spawn"], "spawn"),
                                    *[(g["position"], g["kind"]) for g in scene["goals"]]]:
                    _, data, contacts = _drop(scene, [point[0], point[1], point[2] + 0.15])
                    self.assertAlmostEqual(data.qpos[2], point[2] + 0.025, delta=0.001,
                                           msg=str((stage, seed, kind, point)))
                    self.assertTrue(contacts, (stage, seed, kind, point))
                    if kind == "sit":
                        self.assertTrue(any(g["semantic"] == "chair-seat" for g in contacts))

    def test_lossless_axis_floor_and_hole_contacts(self):
        outer, hole = _rectangle(-4, -3, 4, 3), _rectangle(-1, -1, 1, 1)
        scene = convert_room_scene(_room(outer, [hole], spawn=(-3, 0)))
        self.assertEqual(scene["metadata"]["floorDecomposition"], "axis-aligned-boxes")
        self.assertAlmostEqual(_floor_area(scene), 44)
        floor = [g for g in scene["geoms"] if g["semantic"] == "floor"]
        for x in np.linspace(-4.1, 4.1, 49):
            for z in np.linspace(-3.1, 3.1, 39):
                expected = -4 <= x <= 4 and -3 <= z <= 3 and not (-1 < x < 1 and -1 < z < 1)
                self.assertEqual(_covered((x, -z), floor), expected, (x, z))
        _, supported, contacts = _drop(scene, [-3, 0, 0.15])
        self.assertAlmostEqual(supported.qpos[2], 0.025, delta=0.001)
        self.assertTrue(contacts)
        _, falling, contacts = _drop(scene, [0, 0, 0.15])
        self.assertLess(falling.qpos[2], -1.0)
        self.assertFalse(contacts)
        _, outside, contacts = _drop(scene, [4.4, 0, 0.15])
        self.assertLess(outside.qpos[2], -1.0)
        self.assertFalse(contacts)

    def test_rotated_polygon_hole_uses_convex_prisms_not_hulls(self):
        c, s = math.cos(0.43), math.sin(0.43)
        transform = lambda p: (3 + c * p[0] - s * p[1], -2 + s * p[0] + c * p[1])
        outer = [transform(p) for p in _rectangle(-3, -2, 3, 2)]
        hole = [transform(p) for p in _rectangle(-0.6, -0.5, 0.6, 0.5)]
        spawn = transform((-2, 0))
        scene = convert_room_scene(_room(outer, [hole], spawn=spawn))
        self.assertEqual(scene["metadata"]["floorDecomposition"], "convex-triangular-prisms")
        self.assertTrue(all(g["type"] == "mesh" for g in scene["geoms"]))
        self.assertAlmostEqual(_floor_area(scene), 24 - 1.2)
        floors = scene["geoms"]
        for x in np.linspace(-3.1, 3.1, 37):
            for z in np.linspace(-2.1, 2.1, 29):
                p = transform((x, z))
                expected = -3 <= x <= 3 and -2 <= z <= 2 and not (-0.6 < x < 0.6 and -0.5 < z < 0.5)
                self.assertEqual(_covered((p[0], -p[1]), floors), expected)
        supported = y_up_to_z_up([spawn[0], 0.15, spawn[1]])
        _, data, contacts = _drop(scene, supported)
        self.assertAlmostEqual(data.qpos[2], 0.025, delta=0.001)
        self.assertTrue(contacts)
        centre = transform((0, 0))
        _, data, contacts = _drop(scene, y_up_to_z_up([centre[0], 0.15, centre[1]]))
        self.assertLess(data.qpos[2], -1.0)
        self.assertFalse(contacts)

    def test_concave_floor_preserves_missing_corner(self):
        outline = [(-3, -3), (3, -3), (3, 0), (0, 0), (0, 3), (-3, 3)]
        scene = convert_room_scene(_room(outline, spawn=(-2, 2)))
        self.assertAlmostEqual(_floor_area(scene), 27)
        _, data, contacts = _drop(scene, [2, -2, 0.15])
        self.assertLess(data.qpos[2], -1.0)
        self.assertFalse(contacts)

    def test_existing_chair_geometry_and_coordinate_mapping(self):
        yaw = 0.57
        obstacle = {"id": "chair", "label": "椅子", "position": [1, 0, 1.5],
                    "size": [0.52, 0.95, 0.5], "yaw": yaw, "chair": {"seatHeight": 0.46}}
        destination = {"id": "window", "label": "窓", "position": {"x": -1.5, "z": 1},
                       "heading": yaw, "aliases": ["窓辺"]}
        scene = convert_room_scene(_room(_rectangle(-3, -3, 3, 3), obstacles=[obstacle],
                                         destinations=[destination], spawn=(0, 0)))
        self.assertEqual(scene["spawn"], [0, 0, 0])
        self.assertEqual(z_up_to_y_up(y_up_to_z_up([1.2, 0.3, -2.1])), [1.2, 0.3, -2.1])
        seat = next(g for g in scene["geoms"] if g["semantic"] == "chair-seat")
        self.assertEqual(seat["position"][:2], [1, -1.5])
        self.assertAlmostEqual(seat["position"][2] + seat["size"][2] / 2, 0.46)
        self.assertEqual(seat["yaw"], yaw)
        goal = next(g for g in scene["goals"] if g["id"] == "window")
        self.assertEqual(goal["position"], [-1.5, -1, 0])
        self.assertEqual(goal["aliases"], ["窓辺"])
        # Below the seat remains open: a falling sphere reaches the actual floor.
        _, data, contacts = _drop(scene, [1, -1.5, 0.15])
        self.assertAlmostEqual(data.qpos[2], 0.025, delta=0.001)
        self.assertTrue(all(g["semantic"] == "floor" for g in contacts))
        # Above it contacts the seat, with its exact physical top height.
        _, data, contacts = _drop(scene, [1, -1.5, 0.55])
        self.assertAlmostEqual(data.qpos[2], 0.485, delta=0.001)
        self.assertTrue(any(g["id"] == "chair/seat" for g in contacts))

    def test_imported_spawn_heading_orients_actual_physical_body(self):
        from .environment import HumanoidRoomEnv

        room = _room(_rectangle(-3, -3, 3, 3), spawn=(1.1, -0.7))
        for heading in [0., 0.57, -1.4, math.pi / 2]:
            room["spawn"]["heading"] = heading
            scene = convert_room_scene(room)
            self.assertEqual(scene["spawnHeading"], heading)
            self.assertEqual(scene["metadata"]["spawnHeading"], heading)
            env = HumanoidRoomEnv(scene=scene, randomize=False)
            try:
                np.testing.assert_allclose(env.data.qpos[:2], [1.1, 0.7], atol=1e-12, rtol=0)
                rotation = env.data.xmat[env.pelvis_body_id].reshape(3, 3)
                # Y-up frontend forward +Z becomes Z-up forward -Y; this asks
                # actual MuJoCo FK about the resulting world body direction.
                np.testing.assert_allclose(rotation @ [0, -1, 0],
                                           [math.sin(heading), -math.cos(heading), 0], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(scene["spawn"], [1.1, 0.7, 0])
            finally:
                env.close()

    def test_invalid_floor_and_spawn_are_rejected(self):
        bad = [
            _room([(0, 0), (2, 2), (0, 2), (2, 0)]),
            _room(_rectangle(-2, -2, 2, 2), [_rectangle(1, 1, 3, 3)]),
            _room(_rectangle(-2, -2, 2, 2), [_rectangle(-1, -1, 1, 1)], spawn=(0, 0)),
            {**_room(_rectangle(-2, -2, 2, 2)), "floorHeight": 1},
            {**_room(_rectangle(-2, -2, 2, 2)), "units": "cm"},
        ]
        for room in bad:
            with self.assertRaises(ValueError):
                convert_room_scene(room)
        for args in [(0, 0, 0.5), (5, 0, 0.5), (1, -1, 0.5), (1, 0, 2)]:
            with self.assertRaises(ValueError):
                generate_scene(*args)


if __name__ == "__main__":
    unittest.main()
