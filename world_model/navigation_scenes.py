"""Small phase-two worlds with a genuinely blocked spawn-to-goal route.

Only static geometry and the final destination are generated here. The
visibility graph below supplies a disclosed reward potential, never a policy
observation, waypoint, root position, or controller command.
"""
from __future__ import annotations

import copy
import heapq
import math
import random

from .scenes import ACTOR_RADIUS, _box, _room_shell, box_corners, point_box_distance, validate_scene

MINIMUM_PASSAGE_WIDTH = 2 * ACTOR_RADIUS + .20
REWARD_BODY_MARGIN = ACTOR_RADIUS + .06
CURRICULUM_INTERSECTION_INSET = .10


def segment_intersects_box(start, end, geom, *, margin=0., interior=False):
    """Exact planar segment/rotated-rectangle test, including yaw.

    ``interior`` permits a line along the rectangle boundary. That variant is
    used by the reward visibility graph; the default verifies blocked scenes.
    """
    c, s = math.cos(geom.get("yaw", 0.)), math.sin(geom.get("yaw", 0.))
    x, y = geom["position"][:2]
    ax, ay = start[0] - x, start[1] - y
    bx, by = end[0] - x, end[1] - y
    a = (c * ax + s * ay, -s * ax + c * ay)
    b = (c * bx + s * by, -s * bx + c * by)
    epsilon = -1e-8 if interior else 1e-10
    half = (geom["size"][0] / 2 + margin + epsilon,
            geom["size"][1] / 2 + margin + epsilon)
    low, high = 0., 1.
    for axis in range(2):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if abs(a[axis]) > half[axis]:
                return False
        else:
            first, last = (-half[axis] - a[axis]) / delta, (half[axis] - a[axis]) / delta
            low, high = max(low, min(first, last)), min(high, max(first, last))
            if low > high:
                return False
    return True


def generate_navigation_scene(seed, *, mode="obstacle", difficulty=.5):
    """One physical box, variable clearances and friction; no predetermined path."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("navigation scene seed must be a nonnegative integer")
    if mode not in ("turn", "obstacle"):
        raise ValueError("mode must be turn or obstacle")
    if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)) or not math.isfinite(difficulty):
        raise ValueError("difficulty must be finite")
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must lie between zero and one")
    rng = random.Random(seed)
    distance = rng.uniform(2.0 + .20 * difficulty, 2.12 + .68 * difficulty)
    original_distance = distance
    curriculum_blend = 1. - difficulty if mode == "obstacle" else 0.
    curriculum_metadata = None
    if curriculum_blend > 0.:
        fraction_of_range = (distance - (2.0 + .20 * difficulty)) / (.12 + .48 * difficulty)
        extended_distance = 3.1 + .3 * fraction_of_range
        distance = difficulty * distance + curriculum_blend * extended_distance
    angle_limit = (.04 + .32 * difficulty) if mode == "turn" else (.025 + .09 * difficulty)
    goal_angle = rng.uniform(-angle_limit, angle_limit)
    spawn = [0., distance / 2, 0.]
    goal = [distance * math.sin(goal_angle), spawn[1] - distance * math.cos(goal_angle), 0.]
    obstacle = None
    if mode == "obstacle":
        fraction = rng.uniform(.45, .55)
        if curriculum_blend > 0.:
            extended_fraction = .55 + .5 * (fraction - .45)
            fraction = difficulty * fraction + curriculum_blend * extended_fraction
        height = rng.uniform(.98, 1.24)
        obstacle = _box("navigation-furniture", [
            spawn[0] + fraction * (goal[0] - spawn[0]),
            spawn[1] + fraction * (goal[1] - spawn[1]), height / 2],
            [rng.uniform(.48 + .20 * difficulty, .64 + .46 * difficulty),
             rng.uniform(.30 + .14 * difficulty, .42 + .30 * difficulty), height],
            rng.uniform(-.08 - .40 * difficulty, .08 + .40 * difficulty),
            "furniture", entityId="navigation-furniture", label="回避する家具")
        line_normal = [-(goal[1] - spawn[1]) / distance, (goal[0] - spawn[0]) / distance]
        normal_offset = 0.
        if curriculum_blend > 0.:
            yaw = obstacle["yaw"]
            projected_half_width = (abs(line_normal[0] * math.cos(yaw) + line_normal[1] * math.sin(yaw))
                                    * obstacle["size"][0] / 2
                                    + abs(-line_normal[0] * math.sin(yaw) + line_normal[1] * math.cos(yaw))
                                    * obstacle["size"][1] / 2)
            offset_sign = -1. if rng.random() < .5 else 1.
            # A 10cm projected inset keeps the furniture in the 16-ray sensor
            # fan at reset. A 2cm edge crossing often lies between its rays.
            normal_offset = offset_sign * max(projected_half_width - CURRICULUM_INTERSECTION_INSET, 0.) * curriculum_blend
            for axis in range(2):
                obstacle["position"][axis] += normal_offset * line_normal[axis]
            curriculum_metadata = {
                "navigationCurriculum": "edge_intersection_extended_approach_v1",
                "curriculumBlend": curriculum_blend,
                "originalDirectRouteLengthMeters": original_distance,
                "furnitureRouteFraction": fraction,
                "furnitureLineNormalOffsetMeters": normal_offset,
                "furnitureProjectedHalfWidthMeters": projected_half_width,
                "directLineIntersectionInsetMeters": projected_half_width - abs(normal_offset),
                "minimumCurriculumIntersectionInsetMeters": CURRICULUM_INTERSECTION_INSET,
                "initialGoalAngleChanged": False,
                "description": "low difficulty extends the approach and offsets the blocking furniture towards a route edge",
            }
        # At the largest dimensions a near-endpoint fraction can remove the
        # 55cm spawn/goal approach allowance. Re-centre that static furniture
        # before constructing the room; do not alter endpoints, size, yaw,
        # random draws, or any already initialized physical actor.
        if min(point_box_distance(spawn, obstacle), point_box_distance(goal, obstacle)) < .55:
            obstacle["position"][:2] = [(spawn[axis] + goal[axis]) / 2 for axis in range(2)]
            if curriculum_metadata is not None:
                for axis in range(2):
                    obstacle["position"][axis] += normal_offset * line_normal[axis]
                curriculum_metadata["furnitureRouteFraction"] = .5
        corners = box_corners(obstacle)
        left, right = min(x for x, _ in corners), max(x for x, _ in corners)
        # A left/right asymmetric room is observable in the rays. Both actual
        # lateral gaps exceed a 64cm body envelope plus 20cm safety allowance.
        narrow = rng.uniform(.96, 1.12)
        broad = rng.uniform(1.28, 1.70)
        gaps = (narrow, broad) if rng.random() < .5 else (broad, narrow)
        min_x, max_x = left - gaps[0], right + gaps[1]
    else:
        gaps = None
        min_x, max_x = -1.8, 1.8
    min_y, max_y = min(goal[1], spawn[1]) - 1.15, max(goal[1], spawn[1]) + 1.15
    centre = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    geoms = _room_shell(max_x - min_x, max_y - min_y)
    for geom in geoms:
        geom["position"][0] += centre[0]
        geom["position"][1] += centre[1]
    friction = rng.uniform(.82, 1.16)
    geoms[0]["friction"] = [friction, .005, .0001]
    if obstacle is not None:
        geoms.append(obstacle)
        if not segment_intersects_box(spawn, goal, obstacle):
            raise RuntimeError("navigation direct route is unexpectedly clear")
        if min(gaps) < MINIMUM_PASSAGE_WIDTH:
            raise RuntimeError("navigation scene has insufficient body clearance")
        if min(point_box_distance(spawn, obstacle), point_box_distance(goal, obstacle)) < .55:
            raise RuntimeError("navigation spawn or destination overlaps furniture approach")
    scene = validate_scene({
        "version": 1, "stage": 2, "seed": seed,
        "name": "家具1個を避ける物理歩行" if mode == "obstacle" else "家具なし・方向を変える物理歩行",
        "spawn": spawn, "spawnHeading": 0.,
        "goals": [{"id": "navigation-goal", "kind": "walk", "position": goal,
                   "positionReference": "supporting_surface", "label": "最終目的地"}],
        "geoms": geoms,
        "metadata": {"units": "m", "upAxis": "Z", "forwardAxis": "-Y",
                     "spawnReference": "supporting_floor", "difficulty": float(difficulty),
                     "geometryOnly": True, "infinitePlane": False,
                     "split": "train" if seed < 1_000_000 else "heldout",
                     "floorBounds": [min_x, min_y, max_x, max_y],
                     "floorFriction": friction, "navigationMode": mode,
                     "obstacleBlockedDirectRoute": obstacle is not None,
                     "directRouteBlocked": obstacle is not None,
                     "directRouteLengthMeters": distance,
                     "initialGoalAngleRadians": goal_angle,
                     "obstacleCount": int(obstacle is not None),
                     "leftPassageWidthMeters": gaps[0] if gaps else None,
                     "rightPassageWidthMeters": gaps[1] if gaps else None,
                     "minimumRequiredPassageWidthMeters": MINIMUM_PASSAGE_WIDTH,
                     "rewardBodyMarginMeters": REWARD_BODY_MARGIN,
                     "waypointController": False, "rootStepControl": "MuJoCo dynamics only",
                     "proficiencyPassed": False,
                     "learningStatus": "not_trained_or_validated"}})
    if curriculum_metadata is not None:
        scene["metadata"].update(curriculum_metadata)
    return scene


class NavigationPotential:
    """Collision-free distance used only for reward shaping.

    A single expanded furniture rectangle has four corners. Visibility edges
    and Dijkstra give the distance to the *actual* final goal. Entering its body
    envelope adds a penetration penalty, so pushing into/through the box cannot
    collect shorter-distance credit. There are no commands or selected routes.
    """

    def __init__(self, scene):
        self.goal = tuple(scene["goals"][0]["position"][:2])
        self.obstacle = next((copy.deepcopy(g) for g in scene["geoms"]
                              if g.get("entityId") == "navigation-furniture"), None)
        self.corners = []
        if self.obstacle is not None:
            expanded = copy.deepcopy(self.obstacle)
            expanded["size"][0] += 2 * REWARD_BODY_MARGIN
            expanded["size"][1] += 2 * REWARD_BODY_MARGIN
            self.corners = box_corners(expanded)

    def distance(self, point):
        start = tuple(map(float, point[:2]))
        direct = math.dist(start, self.goal)
        if self.obstacle is None:
            return direct
        obstacle = self.obstacle
        c, s = math.cos(obstacle["yaw"]), math.sin(obstacle["yaw"])
        dx, dy = start[0] - obstacle["position"][0], start[1] - obstacle["position"][1]
        local = (c * dx + s * dy, -s * dx + c * dy)
        half = (obstacle["size"][0] / 2 + REWARD_BODY_MARGIN,
                obstacle["size"][1] / 2 + REWARD_BODY_MARGIN)
        if abs(local[0]) < half[0] - 1e-8 and abs(local[1]) < half[1] - 1e-8:
            penetration = min(half[0] - abs(local[0]), half[1] - abs(local[1]))
            return direct + 2. + 8. * penetration
        if not segment_intersects_box(start, self.goal, obstacle, margin=REWARD_BODY_MARGIN, interior=True):
            return direct
        nodes = [start, self.goal, *self.corners]
        graph = [[] for _ in nodes]
        for first in range(len(nodes)):
            for second in range(first):
                if not segment_intersects_box(nodes[first], nodes[second], obstacle,
                                              margin=REWARD_BODY_MARGIN, interior=True):
                    length = math.dist(nodes[first], nodes[second])
                    graph[first].append((second, length))
                    graph[second].append((first, length))
        queue, distances = [(0., 0)], [0.] + [math.inf] * (len(nodes) - 1)
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            if node == 1:
                return distance
            for neighbour, length in graph[node]:
                candidate = distance + length
                if candidate < distances[neighbour]:
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        raise RuntimeError("navigation reward visibility graph unexpectedly disconnected")
