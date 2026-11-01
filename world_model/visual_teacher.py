"""Offline geometry labels for training only, never called by a visual actor.

The oracle knows actual scene geometry. Its labels are disclosed supervision,
not an inference sensor or a guarantee that a physical rollout will succeed.
"""
from __future__ import annotations

import copy
import heapq
import math

import numpy as np

from .generalization import validate_scene_seed
from .navigation_scenes import segment_intersects_box
from .scenes import box_corners


def oracle_route(scene, margin=.5):
    """A shortest collision-free polygonal route, with actual floor margins."""
    start = np.asarray(scene["spawn"][:2], dtype=np.float64)
    goal = np.asarray(scene["goals"][0]["position"][:2], dtype=np.float64)
    obstacle = next((g for g in scene["geoms"] if g.get("entityId") == "navigation-furniture"), None)
    if obstacle is None:
        return [start, goal]
    inflated = copy.deepcopy(obstacle)
    inflated["size"][0] += 2*margin
    inflated["size"][1] += 2*margin
    floor = next(g for g in scene["geoms"] if g.get("semantic") == "floor")
    low = np.asarray(floor["position"][:2])-np.asarray(floor["size"][:2])/2+.4
    high = np.asarray(floor["position"][:2])+np.asarray(floor["size"][:2])/2-.4
    corners = [np.asarray(p) for p in box_corners(inflated)
               if np.all(np.asarray(p) >= low) and np.all(np.asarray(p) <= high)]
    nodes = [start, goal, *corners]
    graph = [[] for _ in nodes]
    for first in range(len(nodes)):
        for second in range(first):
            if not segment_intersects_box(nodes[first], nodes[second], obstacle,
                                          margin=margin, interior=True):
                distance = float(np.linalg.norm(nodes[first]-nodes[second]))
                graph[first].append((second, distance))
                graph[second].append((first, distance))
    distances, parents, queue = [0.]+[math.inf]*(len(nodes)-1), {}, [(0., 0)]
    while queue:
        cost, index = heapq.heappop(queue)
        if cost > distances[index]:
            continue
        if index == 1:
            indices = [1]
            while indices[-1] != 0:
                indices.append(parents[indices[-1]])
            return [nodes[i] for i in reversed(indices)]
        for neighbour, distance in graph[index]:
            candidate = cost+distance
            if candidate < distances[neighbour]:
                distances[neighbour], parents[neighbour] = candidate, index
                heapq.heappush(queue, (candidate, neighbour))
    raise ValueError("teacher cannot find an inflated-body route inside the actual room")


def route_target(route, position, lookahead=.55, previous_progress=0.):
    lengths = [float(np.linalg.norm(b-a)) for a, b in zip(route, route[1:])]
    cumulative = np.concatenate([[0.], np.cumsum(lengths)])
    candidates = []
    for index, (start, end) in enumerate(zip(route, route[1:])):
        delta = end-start
        fraction = float(np.clip(np.dot(position-start, delta)/max(np.dot(delta, delta), 1e-12), 0, 1))
        projected = start+fraction*delta
        candidates.append((float(np.linalg.norm(position-projected)), cumulative[index]+fraction*lengths[index]))
    progress = max(previous_progress, min(candidates)[1])
    target_distance = min(cumulative[-1], progress+lookahead)
    for index, length in enumerate(lengths):
        if target_distance <= cumulative[index+1]+1e-12:
            fraction = (target_distance-cumulative[index])/max(length, 1e-12)
            return route[index]+fraction*(route[index+1]-route[index]), progress
    return route[-1], progress


class TrainingGeometryTeacher:
    """A disclosed label generator using train scenes only."""

    def __init__(self, env, *, margin=.5, carrot=.55, walking_action=-.6):
        if env.navigation_env.split != "train":
            raise ValueError("privileged visual teacher must not use held-out rooms")
        validate_scene_seed(env.navigation_env._last_scene_seed, "train")
        self.env, self.carrot, self.walking_action = env, carrot, walking_action
        self.route, self.progress = oracle_route(env.scene, margin), 0.

    def action(self):
        body = self.env.unwrapped
        point, self.progress = route_target(self.route, body.data.qpos[:2], self.carrot, self.progress)
        rotation = body.data.xmat[body.pelvis_body_id].reshape(3, 3)
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        delta = point-body.data.qpos[:2]
        bearing = math.atan2(delta[0], -delta[1])
        error = math.atan2(math.sin(bearing-heading), math.cos(bearing-heading))
        return np.asarray([np.clip(error/.45, -1, 1), self.walking_action], dtype=np.float32)
