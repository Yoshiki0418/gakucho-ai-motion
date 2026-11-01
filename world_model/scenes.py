"""Finite physical scenes in metres, Z-up, with forward along -Y.

The generators describe geometry and tasks; they never move an actor or claim a
trained policy. ``spawn`` is a point on the supporting floor, not a pelvis pose.
Boxes use full extents. Meshes are individual closed convex triangular prisms
with geom-local vertices, so MuJoCo's convex collision handling preserves holes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

EPS = 1e-8
FLOOR_THICKNESS = 0.08
ACTOR_RADIUS = 0.32
HELD_OUT_SEED_BASE = 1_000_000
Scene = dict[str, Any]


def y_up_to_z_up(position: Iterable[float]) -> list[float]:
    x, y, z = position
    return [float(x), -float(z), float(y)]


def z_up_to_y_up(position: Iterable[float]) -> list[float]:
    x, y, z = position
    return [float(x), float(z), -float(y)]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return [_number(v, name) for v in value]


def _box(identifier: str, position: Iterable[float], size: Iterable[float],
         yaw: float = 0.0, semantic: str = "furniture", **extra: Any) -> dict[str, Any]:
    return {"id": identifier, "type": "box", "position": list(position),
            "size": list(size), "yaw": yaw, "semantic": semantic, **extra}


def box_corners(geom: dict[str, Any]) -> list[tuple[float, float]]:
    """World XY corners of an upright box, including its rotation."""
    x, y, _ = geom["position"]
    width, depth, _ = geom["size"]
    c, s = math.cos(geom.get("yaw", 0)), math.sin(geom.get("yaw", 0))
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in [(-width / 2, -depth / 2), (width / 2, -depth / 2),
                           (width / 2, depth / 2), (-width / 2, depth / 2)]]


def point_box_distance(point: Iterable[float], geom: dict[str, Any]) -> float:
    """Distance in XY to the actual oriented rectangle, rather than its AABB."""
    px, py, *_ = point
    x, y, _ = geom["position"]
    c, s = math.cos(geom.get("yaw", 0)), math.sin(geom.get("yaw", 0))
    dx, dy = px - x, py - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(lx) - geom["size"][0] / 2, 0),
                      max(abs(ly) - geom["size"][1] / 2, 0))


def _boxes_separated(a: dict[str, Any], b: dict[str, Any], margin: float = 0.12) -> bool:
    ca, cb = box_corners(a), box_corners(b)
    for angle in (a.get("yaw", 0), b.get("yaw", 0)):
        for ax, ay in ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))):
            pa = [x * ax + y * ay for x, y in ca]
            pb = [x * ax + y * ay for x, y in cb]
            if min(pb) - max(pa) >= margin or min(pa) - max(pb) >= margin:
                return True
    return False


def _bounds_clear(geom: dict[str, Any], width: float, depth: float, margin: float = 0.42) -> bool:
    return all(abs(x) <= width / 2 - margin and abs(y) <= depth / 2 - margin
               for x, y in box_corners(geom))


def _chair(identifier: str, origin: Iterable[float], size: Iterable[float],
           seat_height: float, yaw: float = 0.0, **extra: Any) -> list[dict[str, Any]]:
    """Same seat/leg/back boxes as frontend roomScene.ts; no solid chair AABB."""
    ox, oy, oz = origin
    width, depth, height = size
    thickness = min(0.055, seat_height * 0.2, height * 0.12)
    leg = min(0.045, width * 0.13, depth * 0.13)
    leg_height = seat_height - thickness
    c, s = math.cos(yaw), math.sin(yaw)
    result: list[dict[str, Any]] = []

    def add(part: str, local: list[float], dimensions: list[float], semantic: str) -> None:
        x, y, z = local
        result.append(_box(f"{identifier}/{part}", [ox + c * x - s * y, oy + s * x + c * y, oz + z],
                           dimensions, yaw, semantic, entityId=identifier, **extra))

    add("seat", [0, 0, seat_height - thickness / 2], [width, depth, thickness], "chair-seat")
    for side in (-1, 1):
        for end in (-1, 1):
            add(f"leg-{side}-{end}", [side * (width - leg) / 2, end * (depth - leg) / 2, leg_height / 2],
                [leg, leg, leg_height], "chair-leg")
    back_height = height - seat_height
    if back_height > EPS:
        back_thickness = min(depth * 0.12, 0.055)
        add("back", [0, (depth - back_thickness) / 2, seat_height + back_height / 2],
            [width, back_thickness, back_height], "chair-back")
    return result


def _room_shell(width: float, depth: float, height: float = 2.8) -> list[dict[str, Any]]:
    wall = 0.10
    return [
        _box("floor", [0, 0, -FLOOR_THICKNESS / 2], [width, depth, FLOOR_THICKNESS], semantic="floor"),
        _box("wall-west", [-width / 2 - wall / 2, 0, height / 2], [wall, depth + 2 * wall, height], semantic="wall"),
        _box("wall-east", [width / 2 + wall / 2, 0, height / 2], [wall, depth + 2 * wall, height], semantic="wall"),
        _box("wall-north", [0, depth / 2 + wall / 2, height / 2], [width, wall, height], semantic="wall"),
        _box("wall-south", [0, -depth / 2 - wall / 2, height / 2], [width, wall, height], semantic="wall"),
    ]


def _scene(stage: int, seed: int, name: str, geoms: list[dict[str, Any]], spawn: list[float],
           goals: list[dict[str, Any]], difficulty: float, **metadata: Any) -> Scene:
    result = {"version": 1, "stage": stage, "seed": seed, "name": name,
              "geoms": geoms, "spawn": spawn, "goals": goals,
              "metadata": {"units": "m", "upAxis": "Z", "forwardAxis": "-Y",
                           "spawnReference": "supporting_floor", "difficulty": difficulty,
                           "geometryOnly": True, "infinitePlane": False,
                           "split": "heldout" if seed >= HELD_OUT_SEED_BASE else "train", **metadata}}
    return validate_scene(result)


def _stage_one(seed: int, difficulty: float, include_chair: bool = True) -> Scene:
    geoms = _room_shell(6, 6)
    # Contact proxy uses the measured visible home-chair height and seat level.
    # It preserves open leg space, unlike the navigation-only chair envelope.
    chair_pos = [-1.65, -1.55, 0]
    seat_height = 0.443916156991599
    goals = [{"id": "walk-target", "position": [0, -1.6, 0], "kind": "walk"}]
    if include_chair:
        geoms += _chair("home-chair", chair_pos, [0.428832, 0.536166, 0.95], seat_height,
                        provenance="measured-home-chair-contact-proxy")
        goals += [
        {"id": "chair-seat", "position": [*chair_pos[:2], seat_height], "kind": "sit", "entityId": "home-chair"},
        {"id": "chair-front", "position": [-1.65, -2.30, 0], "kind": "walk"},
        ]
    return _scene(1, seed, "物理基盤・有限床と椅子" if include_chair else "物理基盤・有限床", geoms, [0, 1.6, 0],
                  goals, difficulty, floorBounds=[-3, -3, 3, 3], chairSeatHeight=seat_height if include_chair else None,
       referenceAvatarHeight=1.8, ankleClearanceIsVisualOnly=0.075)


def _stage_two(seed: int, difficulty: float) -> Scene:
    rng = random.Random(seed)
    width, depth = rng.uniform(7, 10), rng.uniform(6.5, 10)
    passage = rng.uniform(1.0, 1.6) - difficulty * 0.12
    geoms = _room_shell(width, depth)
    geoms[0]["friction"] = [rng.uniform(0.8, 1.15), 0.005, 0.0001]
    spawn, target = [-width * 0.36, 0, 0], [width * 0.36, 0, 0]
    entities: list[dict[str, Any]] = []
    reserved = [spawn, target]
    goals = [{"id": "across-room", "position": target, "kind": "walk"},
             {"id": "return-to-entry", "position": spawn, "kind": "walk"}]
    # Place the chair first and reserve its physical front approach for later furniture.
    chair_pos = [-width * 0.24, -depth * 0.29, 0]
    seat_height = rng.uniform(0.42, 0.50)
    chair_size = [rng.uniform(0.46, 0.55), rng.uniform(0.46, 0.57), rng.uniform(0.88, 1.02)]
    for _ in range(100):
        yaw = rng.uniform(-math.pi, math.pi)
        envelope = _box("chair-envelope", [chair_pos[0], chair_pos[1], chair_size[2] / 2], chair_size, yaw)
        front = [chair_pos[0] + math.sin(yaw) * 0.82, chair_pos[1] - math.cos(yaw) * 0.82, 0]
        if _bounds_clear(envelope, width, depth) and abs(front[0]) < width / 2 - 0.45 and abs(front[1]) < depth / 2 - 0.45:
            break
    else:
        raise ValueError("could not place the randomized chair")
    entities.append(envelope)
    reserved.append(front)
    geoms += _chair("chair", chair_pos, chair_size, seat_height, yaw)
    goals += [{"id": "chair-front", "position": front, "kind": "walk"},
              {"id": "chair-seat", "position": [*chair_pos[:2], seat_height], "kind": "sit", "entityId": "chair"}]
    desired = 4 + int(difficulty * 4)
    for index in range(desired):
        for _ in range(1500):
            height = rng.uniform(0.42, 1.45)
            size = [rng.uniform(0.45, 1.45), rng.uniform(0.45, 1.35), height]
            side = -1 if rng.random() < 0.5 else 1
            position = [rng.uniform(-width / 2 + 0.6, width / 2 - 0.6),
                        side * rng.uniform(passage / 2 + 0.5, depth / 2 - 0.6), height / 2]
            candidate = _box(f"furniture-{index}", position, size, rng.uniform(-math.pi, math.pi))
            if not _bounds_clear(candidate, width, depth):
                continue
            if any(abs(y) < passage / 2 + 0.10 for _, y in box_corners(candidate)):
                continue
            if any(point_box_distance(p, candidate) < 0.45 for p in reserved):
                continue
            if not all(_boxes_separated(candidate, other) for other in entities):
                continue
            entities.append(candidate)
            geoms.append(candidate)
            break
        else:
            raise ValueError("could not construct a nonoverlapping furniture layout")
    return _scene(2, seed, "配置・寸法・向きを変えた部屋", geoms, spawn, goals, difficulty,
                  floorBounds=[-width / 2, -depth / 2, width / 2, depth / 2],
                  guaranteedPassage={"axis": "X", "centreY": 0, "width": passage},
                  furnitureEnvelopes=entities, heldoutSeedBase=HELD_OUT_SEED_BASE)


def _wall_segments_at_y(identifier: str, y: float, width: float, door_centres: list[float],
                        door_width: float, height: float) -> list[dict[str, Any]]:
    intervals = sorted((c - door_width / 2, c + door_width / 2) for c in door_centres)
    cursor, result = -width / 2, []
    for i, (left, right) in enumerate(intervals + [(width / 2, width / 2)]):
        if left - cursor > EPS:
            result.append(_box(f"{identifier}-{i}", [(cursor + left) / 2, y, height / 2],
                               [left - cursor, 0.12, height], semantic="partition"))
        cursor = right
    return result


def _stage_three(seed: int, difficulty: float) -> Scene:
    rng = random.Random(seed)
    width, depth = rng.uniform(13, 15), rng.uniform(9, 11)
    hall_half, door_width = 1.0, rng.uniform(1.15, 1.60) - difficulty * 0.10
    quarter = width / 4
    room_depth = depth / 2 - hall_half
    geoms = _room_shell(width, depth)
    for side in (-1, 1):
        geoms += _wall_segments_at_y(f"hall-wall-{side}", side * hall_half, width,
                                     [-quarter, quarter], door_width, 2.6)
        geoms.append(_box(f"room-divider-{side}", [0, side * (hall_half + room_depth / 2), 1.3],
                          [0.12, room_depth, 2.6], semantic="partition"))
    goals, rooms, portals = [], [], []
    names = ["work-room", "kitchen", "bed-room", "living-room"]
    for index, (sx, sy) in enumerate(((-1, 1), (1, 1), (-1, -1), (1, -1))):
        x, y = sx * quarter, sy * (hall_half + room_depth * 0.54)
        name = names[index]
        goals.append({"id": name, "position": [x, y, 0], "kind": "walk"})
        portals.append({"id": f"door-{name}", "position": [x, sy * hall_half, 0], "width": door_width})
        rooms.append({"id": name, "bounds": [(-width / 2 if sx < 0 else 0), (hall_half if sy > 0 else -depth / 2),
                                               (0 if sx < 0 else width / 2), (depth / 2 if sy > 0 else -hall_half)]})
        height = rng.uniform(0.5, 1.2)
        geoms.append(_box(f"fixture-{name}", [sx * (width / 2 - 0.9), sy * (depth / 2 - 0.9), height / 2],
                          [0.85, 0.80, height], rng.uniform(-0.15, 0.15)))
    chair_pos = [-quarter - 0.8, -depth / 2 + 1.2, 0]
    geoms += _chair("bedroom-chair", chair_pos, [0.50, 0.50, 0.94], 0.46, math.pi / 2)
    goals.append({"id": "bedroom-chair-seat", "position": [*chair_pos[:2], 0.46], "kind": "sit", "entityId": "bedroom-chair"})
    return _scene(3, seed, "廊下と扉でつながる四つの部屋", geoms, [-width / 2 + 1.1, 0, 0], goals,
                  difficulty, floorBounds=[-width / 2, -depth / 2, width / 2, depth / 2],
                  rooms=rooms, portals=portals, corridorWidth=2 * hall_half,
                  taskSequence=[g["id"] for g in goals], connectedWithoutTeleport=True)


def _stage_four(seed: int, difficulty: float) -> Scene:
    rng = random.Random(seed)
    if difficulty < 1 / 3:
        height = 0.06 + difficulty * 0.18
        geoms = _room_shell(6, 8)
        geoms.append(_box("low-step", [0, -1.5, height / 2], [2.6, 1.8, height], semantic="step"))
        return _scene(4, seed, "低い段差への移動", geoms, [0, 2.2, 0],
                      [{"id": "raised-target", "position": [0, -1.5, height], "kind": "step"}], difficulty,
                      terrain="low-step", rise=height, steps=1, floorBounds=[-3, -4, 3, 4])
    short = difficulty < 2 / 3
    width, depth = (6, 8) if short else (8, 16)
    count = 3 if short else 16
    rise = rng.uniform(0.095, 0.115) if short else rng.uniform(0.16, 0.18)
    run = 0.38 if short else rng.uniform(0.28, 0.30)
    stair_width = 1.8
    start_y = 1.0 if short else 4.3
    total_height = count * rise
    geoms = _room_shell(width, depth, max(2.8, total_height + 2.4))
    treads = []
    for i in range(count):
        top = (i + 1) * rise
        y = start_y - (i + 0.5) * run
        geoms.append(_box(f"step-{i + 1}", [0, y, top / 2], [stair_width, run, top], semantic="stair"))
        treads.append({"position": [0, y, top], "index": i + 1})
    end_y = start_y - count * run
    landing_depth = 1.6
    landing_y = end_y - landing_depth / 2
    geoms.append(_box("landing", [0, landing_y, total_height / 2],
                      [3.2, landing_depth, total_height], semantic="landing"))
    goals = [{"id": "stair-landing", "position": [0, landing_y, total_height], "kind": "stairs"}]
    if not short:
        upper_end = end_y - landing_depth
        upper_start = -depth / 2
        upper_depth = upper_end - upper_start
        upper_y = (upper_end + upper_start) / 2
        # A slab with air below it is a true upper floor, not a tall filled room.
        geoms.append(_box("second-floor", [0, upper_y, total_height - 0.10],
                          [width, upper_depth, 0.20], semantic="upper-floor"))
        goals.insert(0, {"id": "second-floor-target", "position": [0, upper_y, total_height], "kind": "stairs"})
        for side in (-1, 1):
            geoms.append(_box(f"landing-rail-{side}", [side * 1.64, landing_y, total_height + 0.5],
                              [0.08, landing_depth, 1.0], semantic="railing"))
        chair_pos = [1.5, upper_y, total_height]
        geoms += _chair("upper-chair", chair_pos, [0.50, 0.50, 0.94], 0.45)
        goals.append({"id": "upper-chair-seat", "position": [*chair_pos[:2], total_height + 0.45],
                      "kind": "sit", "entityId": "upper-chair"})
    return _scene(4, seed, "短い階段" if short else "階段から二階の床へ", geoms,
                  [0, 2.7 if short else 5.6, 0], goals, difficulty,
                  terrain="short-stairs" if short else "stairs-and-second-floor", rise=rise,
                  run=run, steps=count, upperFloorHeight=total_height, treadCentres=treads,
                  floorBounds=[-width / 2, -depth / 2, width / 2, depth / 2])


def generate_scene(stage: int, seed: int = 0, difficulty: float = 0.5, *, include_chair: bool = True) -> Scene:
    """Deterministic four-stage geometry; held-out seeds start at 1,000,000."""
    if isinstance(stage, bool) or stage not in (1, 2, 3, 4):
        raise ValueError("stage must be one of 1, 2, 3, 4")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    difficulty = _number(difficulty, "difficulty")
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be between 0 and 1")
    if not isinstance(include_chair, bool):
        raise ValueError("include_chair must be a boolean")
    return _stage_one(seed, difficulty, include_chair) if stage == 1 else (_stage_two, _stage_three, _stage_four)[stage - 2](seed, difficulty)


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_edge_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denominator)) if denominator else 0
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _edges(ring: list[tuple[float, float]]) -> Iterable[tuple[tuple[float, float], tuple[float, float]]]:
    return zip(ring, ring[1:] + ring[:1])


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ab_c * ab_d < -EPS * EPS and cd_a * cd_b < -EPS * EPS:
        return True
    return min(_point_edge_distance(a, c, d), _point_edge_distance(b, c, d),
               _point_edge_distance(c, a, b), _point_edge_distance(d, a, b)) <= EPS


def _inside(p: tuple[float, float], ring: list[tuple[float, float]], boundary: bool = True) -> bool:
    result = False
    for a, b in _edges(ring):
        if _point_edge_distance(p, a, b) <= EPS:
            return boundary
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]:
            result = not result
    return result


def _area(ring: list[tuple[float, float]]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges(ring)) / 2


def _ring(value: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 128:
        raise ValueError(f"{name} must contain 3..128 points")
    result = [(_number(p.get("x"), name), -_number(p.get("z"), name))
              for p in value if isinstance(p, dict)]
    if len(result) != len(value):
        raise ValueError(f"{name} contains an invalid point")
    if math.dist(result[0], result[-1]) <= EPS:
        result.pop()
    if len(result) < 3 or abs(_area(result)) <= EPS:
        raise ValueError(f"{name} has no area")
    edges = list(_edges(result))
    for i, (a, b) in enumerate(edges):
        if math.dist(a, b) <= EPS:
            raise ValueError(f"{name} has a zero-length edge")
        previous = result[(i - 1) % len(result)]
        if abs(_cross(previous, a, b)) <= EPS and (a[0] - previous[0]) * (b[0] - a[0]) + (a[1] - previous[1]) * (b[1] - a[1]) < 0:
            raise ValueError(f"{name} has an overlapping folded edge")
        for j, (c, d) in enumerate(edges[i + 1:], i + 1):
            if j == i + 1 or (i == 0 and j == len(edges) - 1):
                continue
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"{name} self-intersects")
    return result


def _floor_contains(p: tuple[float, float], outline: list[tuple[float, float]],
                    holes: list[list[tuple[float, float]]]) -> bool:
    return _inside(p, outline) and not any(_inside(p, hole) for hole in holes)


def _floor_clear(p: tuple[float, float], outline: list[tuple[float, float]],
                 holes: list[list[tuple[float, float]]], radius: float) -> bool:
    return _floor_contains(p, outline, holes) and all(_point_edge_distance(p, a, b) >= radius - EPS
                                                   for ring in [outline, *holes] for a, b in _edges(ring))


def _axis_floor_boxes(outline: list[tuple[float, float]], holes: list[list[tuple[float, float]]]) -> list[dict[str, Any]] | None:
    rings = [outline, *holes]
    if any(abs(a[0] - b[0]) > EPS and abs(a[1] - b[1]) > EPS for ring in rings for a, b in _edges(ring)):
        return None
    xs = sorted({p[0] for ring in rings for p in ring})
    ys = sorted({p[1] for ring in rings for p in ring})
    active: dict[tuple[float, float], list[float]] = {}
    rectangles: list[list[float]] = []
    for y0, y1 in zip(ys, ys[1:]):
        row: list[tuple[float, float]] = []
        run_start = None
        for i, (x0, x1) in enumerate(zip(xs, xs[1:])):
            inside = _floor_contains(((x0 + x1) / 2, (y0 + y1) / 2), outline, holes)
            if inside and run_start is None:
                run_start = x0
            if run_start is not None and (not inside or i == len(xs) - 2):
                row.append((run_start, x0 if not inside else x1))
                run_start = None
        next_active = {}
        for interval in row:
            rectangle = active.pop(interval, [interval[0], y0, interval[1], y0])
            rectangle[3] = y1
            next_active[interval] = rectangle
        rectangles.extend(active.values())
        active = next_active
    rectangles.extend(active.values())
    rectangles.sort()
    return [_box(f"floor-{i}", [(x0 + x1) / 2, (y0 + y1) / 2, -FLOOR_THICKNESS / 2],
                 [x1 - x0, y1 - y0, FLOOR_THICKNESS], semantic="floor")
            for i, (x0, y0, x1, y1) in enumerate(rectangles)]


def _floor_prisms(outline: list[tuple[float, float]], holes: list[list[tuple[float, float]]]) -> list[dict[str, Any]]:
    try:
        import mapbox_earcut
        import numpy as np
    except ImportError as error:
        raise RuntimeError("arbitrary polygon floors require mapbox-earcut and numpy in the physics venv") from error
    points = [p for ring in [outline, *holes] for p in ring]
    ends = []
    for ring in [outline, *holes]:
        ends.append((ends[-1] if ends else 0) + len(ring))
    indices = mapbox_earcut.triangulate_float64(np.array(points, dtype=np.float64), np.array(ends, dtype=np.uint32))
    result, area = [], 0.0
    for index in range(0, len(indices), 3):
        tri = [points[int(i)] for i in indices[index:index + 3]]
        signed = _cross(*tri) / 2
        if abs(signed) <= EPS:
            continue
        if signed < 0:
            tri[1], tri[2] = tri[2], tri[1]
        cx, cy = sum(p[0] for p in tri) / 3, sum(p[1] for p in tri) / 3
        if not _floor_contains((cx, cy), outline, holes):
            raise ValueError("floor triangulation produced a triangle outside the walkable floor")
        area += abs(signed)
        vertices = [[x - cx, y - cy, z] for z in (-FLOOR_THICKNESS / 2, FLOOR_THICKNESS / 2) for x, y in tri]
        result.append({"id": f"floor-prism-{len(result)}", "type": "mesh", "semantic": "floor",
                       "position": [cx, cy, -FLOOR_THICKNESS / 2], "yaw": 0.0,
                       "vertices": vertices,
                       "faces": [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                                 [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]})
    expected = abs(_area(outline)) - sum(abs(_area(hole)) for hole in holes)
    if not result or abs(area - expected) > max(1e-7, expected * 1e-8):
        raise ValueError("floor triangulation failed to preserve its area and holes")
    return result


def convert_room_scene(definition: dict[str, Any]) -> Scene:
    """Losslessly convert the supported flat frontend room JSON to contact geometry.

    No infinite plane or floor AABB is inserted below missing floor regions.
    Original chair pieces exactly match the common JSON renderer. Display-only
    labels/aliases are retained in metadata and goals.
    """
    if not isinstance(definition, dict) or definition.get("version") != 1:
        raise ValueError("room version must be 1")
    if definition.get("units", "m") != "m" or definition.get("upAxis", "Y") not in ("Y", "Y-up"):
        raise ValueError("room must use metres and Y-up")
    if definition.get("axes", "Y") not in ("Y", "Y-up") or definition.get("floorHeight", 0) != 0:
        raise ValueError("room must have a flat floor at Y=0")
    floor = definition.get("floor", {})
    outline = _ring(floor.get("outline"), "floor.outline")
    holes = [_ring(h, "floor.holes") for h in floor.get("holes", [])]
    if len(holes) > 16:
        raise ValueError("room contains too many holes")
    for i, hole in enumerate(holes):
        if not all(_inside(p, outline, False) for p in hole) or any(_segments_intersect(a, b, c, d)
                for a, b in _edges(outline) for c, d in _edges(hole)):
            raise ValueError("floor hole touches or leaves the outer ring")
        for prior in holes[:i]:
            if _inside(hole[0], prior) or _inside(prior[0], hole) or any(_segments_intersect(a, b, c, d)
                    for a, b in _edges(hole) for c, d in _edges(prior)):
                raise ValueError("floor holes overlap")
    obstacles = definition.get("obstacles", [])
    destinations = definition.get("destinations", [])
    if not isinstance(obstacles, list) or not isinstance(destinations, list):
        raise ValueError("obstacles and destinations must be arrays")
    if sum(len(ring) for ring in [outline, *holes]) + 4 * len(obstacles) > 240:
        raise ValueError("room exceeds the 240 corner complexity budget")
    geoms = _axis_floor_boxes(outline, holes)
    decomposition = "axis-aligned-boxes" if geoms is not None else "convex-triangular-prisms"
    if geoms is None:
        geoms = _floor_prisms(outline, holes)
    goals, chair_goals, envelopes, ids = [], [], [], set()
    for item in obstacles:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("room entity IDs must be nonempty and unique")
        ids.add(identifier)
        position = y_up_to_z_up(_vector(item.get("position"), f"{identifier}.position"))
        raw_size = _vector(item.get("size"), f"{identifier}.size")
        if any(v <= 0 for v in raw_size):
            raise ValueError("obstacle sizes must be positive")
        size = [raw_size[0], raw_size[2], raw_size[1]]
        yaw = _number(item.get("yaw", 0), "yaw")
        chair = item.get("chair")
        if chair is not None:
            seat_height = _number(chair.get("seatHeight"), "seatHeight")
            if not 0 < seat_height <= size[2] or abs(position[2]) > 1e-7:
                raise ValueError("chair must have a valid seat height and floor origin")
            geoms += _chair(identifier, position, size, seat_height, yaw, label=item.get("label", identifier))
            envelopes.append(_box(identifier, [position[0], position[1], position[2] + size[2] / 2], size, yaw))
            chair_goals.append({"id": f"chair:{identifier}", "position": [position[0], position[1], position[2] + seat_height],
                          "kind": "sit", "entityId": identifier, "heading": yaw, "label": item.get("label", identifier)})
        else:
            box = _box(identifier, position, size, yaw, label=item.get("label", identifier))
            geoms.append(box)
            envelopes.append(box)
    for item in destinations:
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("destination IDs must be unique")
        ids.add(identifier)
        p = item.get("position", {})
        point = [_number(p.get("x"), "destination.x"), -_number(p.get("z"), "destination.z"), 0.0]
        if not _floor_contains(tuple(point[:2]), outline, holes):
            raise ValueError("destination leaves the floor")
        goals.append({"id": identifier, "position": point, "kind": "walk",
                      "heading": _number(item.get("heading", 0), "heading"),
                      "label": item.get("label", identifier), "aliases": list(item.get("aliases", []))})
    explicit = definition.get("spawn")
    if explicit is not None:
        spawn = [_number(explicit.get("x"), "spawn.x"), -_number(explicit.get("z"), "spawn.z"), 0.0]
        heading = _number(explicit.get("heading", 0), "spawn.heading")
        if not _floor_clear(tuple(spawn[:2]), outline, holes, 0.275) or any(point_box_distance(spawn, box) < 0.275 - EPS for box in envelopes):
            raise ValueError("spawn overlaps furniture or the floor boundary")
    else:
        xs, ys = [p[0] for p in outline], [p[1] for p in outline]
        centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        candidates = [centre] + [(min(xs) + (i + 0.5) / 64 * (max(xs) - min(xs)),
                                  min(ys) + (j + 0.5) / 64 * (max(ys) - min(ys))) for i in range(64) for j in range(64)]
        candidates.sort(key=lambda p: math.dist(p, centre))
        chosen = next((p for p in candidates if _floor_clear(p, outline, holes, ACTOR_RADIUS)
                       and all(point_box_distance(p, box) >= ACTOR_RADIUS for box in envelopes)), None)
        if chosen is None:
            raise ValueError("room has no safe spawn")
        spawn, heading = [*chosen, 0.0], 0.0
    scene = _scene(1, 0, str(definition.get("name", "Imported room")), geoms, spawn, goals + chair_goals, 0.0,
                  source="frontend-room-scene-v1", sourceDefinition=copy.deepcopy(definition),
                  conversion={"position": "(x,y,z) -> (x,-z,y)", "yaw": "Y-yaw -> Z-yaw, same sign"},
                  spawnHeading=heading, floorOutline=[list(p) for p in outline],
                  floorHoles=[[list(p) for p in h] for h in holes], floorDecomposition=decomposition,
                  floorArea=abs(_area(outline)) - sum(abs(_area(hole)) for hole in holes))
    scene["spawnHeading"] = heading
    return scene


def validate_scene(scene: Scene) -> Scene:
    if scene.get("version") != 1 or scene.get("stage") not in (1, 2, 3, 4):
        raise ValueError("invalid physics scene version/stage")
    ids: set[str] = set()
    for geom in scene.get("geoms", []):
        if not isinstance(geom.get("id"), str) or not geom["id"] or geom["id"] in ids:
            raise ValueError("geom IDs must be nonempty and unique")
        ids.add(geom["id"])
        _vector(geom.get("position"), "geom.position")
        _number(geom.get("yaw", 0), "geom.yaw")
        if geom.get("type", "box") == "box":
            if any(v <= 0 for v in _vector(geom.get("size"), "geom.size")):
                raise ValueError("box extents must be positive")
        elif geom["type"] == "mesh":
            vertices = geom.get("vertices", [])
            faces = geom.get("faces", [])
            if len(vertices) < 4 or len(faces) < 4:
                raise ValueError("collision meshes must be closed convex volumes")
            for vertex in vertices:
                _vector(vertex, "mesh.vertex")
            if any(len(face) != 3 or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(vertices)
                                        for i in face) for face in faces):
                raise ValueError("invalid mesh face")
        else:
            raise ValueError("unknown collision geom type")
        if "friction" in geom and any(v < 0 for v in _vector(geom["friction"], "friction")):
            raise ValueError("friction must be nonnegative")
    if not ids:
        raise ValueError("a physical scene needs finite collision geometry")
    _vector(scene.get("spawn"), "spawn")
    for goal in scene.get("goals", []):
        _vector(goal.get("position"), "goal.position")
        if not isinstance(goal.get("id"), str) or not goal["id"]:
            raise ValueError("goals require an ID")
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--source", type=Path, help="Convert an existing frontend room JSON")
    parser.add_argument("--output", "--out", type=Path)
    args = parser.parse_args()
    scene = convert_room_scene(json.loads(args.source.read_text())) if args.source else generate_scene(args.stage, args.seed, args.difficulty)
    text = json.dumps(scene, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
