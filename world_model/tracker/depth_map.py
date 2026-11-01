"""G5: plan the route on a map built from the head depth camera, not from the scene.

The avatar stands at the spawn and takes one first-person depth image
(``world_model.depth_perception``: 64x48 pinhole rays, 90 degree FOV, 5 m,
15 degree down pitch, own body excluded). Every hit between 5 cm and 2 m above
the floor marks its floor cell as occupied; cells within ``margin`` of an
occupied cell are blocked (the privileged planner inflates the furniture box by
the same margin). Unseen space counts as free, so only what the camera saw can
bend the route. A* on 5 cm cells, then line-of-sight shortcuts give a polyline
like the privileged visibility route. Only the spawn and the goal position
are given; no scene geometry (furniture, walls, room size) is read.
"""
from __future__ import annotations

import heapq
import math
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

CELL = .05            # m
OBSTACLE_Z = (.05, 2.)  # m above the floor; floor hits and the ceiling are not obstacles
PAD = 3.              # m of map around the spawn and the goal


class Grid:
    def __init__(self, start, goal, pad=PAD, cell=CELL):
        corners = np.stack([start, goal])
        self.low = corners.min(axis=0) - pad
        self.cell = cell
        shape = np.ceil((corners.max(axis=0) + pad - self.low) / cell).astype(int)
        self.occupied = np.zeros(tuple(shape), bool)

    def index(self, xy):
        return np.floor((np.asarray(xy) - self.low) / self.cell).astype(int)

    def centre(self, ij):
        return self.low + (np.asarray(ij) + .5) * self.cell

    def mark(self, points_xy):
        ij = self.index(points_xy)
        inside = np.all((ij >= 0) & (ij < self.occupied.shape), axis=1)
        self.occupied[ij[inside, 0], ij[inside, 1]] = True

    def blocked(self, margin):
        if not self.occupied.any():
            return np.zeros_like(self.occupied)
        return distance_transform_edt(~self.occupied) * self.cell <= margin


def observe(model, data, grid, config=None):
    """Mark what the head depth camera sees now. Returns the number of obstacle hits."""
    from ..depth_perception import DepthCameraConfig, _camera_pose, _cast, _filtered_model, _pixel_rays
    config = config or DepthCameraConfig()
    _, origin, basis, self_root = _camera_pose(SimpleNamespace(model=model, data=data), config)
    sensor_model, query_data, _ = _filtered_model(model, self_root)
    mujoco.mj_copyData(query_data, model, data)
    rays = np.ascontiguousarray(_pixel_rays(config) @ basis)
    ranges, geom_ids = _cast(sensor_model, query_data, origin, rays, config.far_meters, "multi_ray")
    valid = (geom_ids >= 0) & (ranges >= 0) & (ranges <= config.far_meters)
    points = origin + rays[valid] * ranges[valid, None]
    obstacle = points[(points[:, 2] > OBSTACLE_Z[0]) & (points[:, 2] < OBSTACLE_Z[1])]
    grid.mark(obstacle[:, :2])
    return int(len(obstacle))


def _free_segment(blocked, a, b):
    steps = int(np.ceil(np.abs(np.asarray(b) - np.asarray(a)).max())) + 1
    line = np.rint(np.linspace(a, b, steps)).astype(int)
    return not blocked[line[:, 0], line[:, 1]].any()


def plan(grid, start, goal, margin, clear=2):
    """Shortest 8-connected route avoiding blocked cells, shortcut by line of sight.

    ``clear`` cells around the start and the goal stay free: the avatar and the
    goal stand in free space even if a hit lands near them (home goals stand
    next to furniture, closer than the margin).
    """
    blocked = grid.blocked(margin)
    s, g = tuple(grid.index(start)), tuple(grid.index(goal))
    for cell in (s, g):
        blocked[max(cell[0] - clear, 0):cell[0] + clear + 1, max(cell[1] - clear, 0):cell[1] + clear + 1] = False
    moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    best, parent, queue = {s: 0.}, {}, [(0., s)]
    while queue:
        _, cell = heapq.heappop(queue)
        if cell == g:
            break
        for dx, dy, step in moves:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < blocked.shape[0] and 0 <= nxt[1] < blocked.shape[1]) or blocked[nxt]:
                continue
            cost = best[cell] + step
            if cost < best.get(nxt, math.inf):
                best[nxt], parent[nxt] = cost, cell
                heuristic = math.hypot(nxt[0] - g[0], nxt[1] - g[1])
                heapq.heappush(queue, (cost + heuristic, nxt))
    if g not in best:
        raise ValueError("no route on the depth map")
    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    keep, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _free_segment(blocked, cells[i], cells[j]):
            j -= 1
        keep.append(cells[j])
        i = j
    route = [np.asarray(start, float)] + [grid.centre(c) for c in keep[1:-1]] + [np.asarray(goal, float)]
    return route, blocked
