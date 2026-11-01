"""Compound contact proxies for the displayed home and a near-chair curriculum.

Dimensions/placements come from frontend homeEnvironment.ts. The imported CC0
chair's measured visible seat position is retained, while its contact geometry
uses explicit seat, back and feet boxes rather than a filled navigation AABB.
These are declared physical proxies, not a copy of every decorative render mesh.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

from .scenes import FLOOR_THICKNESS, _box, _room_shell, validate_scene, y_up_to_z_up

HOME_CHAIR_SEAT_HEIGHT = 0.443916156991599
HOME_CHAIR_HEIGHT = 0.95
HOME_CHAIR_WIDTH = 0.428832
HOME_CHAIR_DEPTH = 0.536166
HOME_CHAIR_SEAT_LOCAL_X = 0.007016075838638
HOME_CHAIR_SEAT_LOCAL_FORWARD = 0.083495990193447


def _rotate_xy(local: Iterable[float], origin: Iterable[float], yaw: float) -> list[float]:
    x, y, z = local
    ox, oy, oz = origin
    c, s = math.cos(yaw), math.sin(yaw)
    return [ox + c * x - s * y, oy + s * x + c * y, oz + z]


def _add_y_box(geoms: list[dict[str, Any]], identifier: str, position: Iterable[float],
               size: Iterable[float], semantic: str = "furniture", yaw: float = 0,
               entity: str | None = None) -> None:
    width, height, depth = size
    geoms.append(_box(identifier, y_up_to_z_up(position), [width, depth, height], yaw, semantic,
                      entityId=entity or identifier))


def _add_group_box(geoms: list[dict[str, Any]], identifier: str, local: Iterable[float],
                   size: Iterable[float], origin: Iterable[float], yaw: float,
                   semantic: str = "furniture", entity: str = "sofa") -> None:
    x, y, z = local
    ox, oy, oz = origin
    c, s = math.cos(yaw), math.sin(yaw)
    _add_y_box(geoms, identifier, [ox + c * x + s * z, oy + y, oz - s * x + c * z],
               size, semantic, yaw, entity)


def _measured_chair(identifier: str, origin: list[float], yaw: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """origin is the measured model's floor anchor in physical Z-up coordinates."""
    geoms = []
    seat_local = [HOME_CHAIR_SEAT_LOCAL_X, -HOME_CHAIR_SEAT_LOCAL_FORWARD, HOME_CHAIR_SEAT_HEIGHT]
    seat_point = _rotate_xy(seat_local, origin, yaw)
    thickness = 0.05

    def add(part, local, size, semantic):
        geoms.append(_box(f"{identifier}/{part}", _rotate_xy(local, origin, yaw), size, yaw,
                          semantic, entityId=identifier, provenance="measured-home-chair-compound-proxy"))

    # Seat front extent is inside the measured full frame bounds, with its true centre bias.
    add("seat", [*seat_local[:2], HOME_CHAIR_SEAT_HEIGHT - thickness / 2],
        [0.41, 0.36, thickness], "chair-seat")
    back_thickness = 0.05
    back_height = HOME_CHAIR_HEIGHT - HOME_CHAIR_SEAT_HEIGHT
    add("back", [0, (HOME_CHAIR_DEPTH - back_thickness) / 2,
                 HOME_CHAIR_SEAT_HEIGHT + back_height / 2],
        [HOME_CHAIR_WIDTH, back_thickness, back_height], "chair-back")
    leg_size = 0.04
    leg_height = HOME_CHAIR_SEAT_HEIGHT - thickness
    for side in (-1, 1):
        for end in (-1, 1):
            add(f"leg-{side}-{end}", [side * (HOME_CHAIR_WIDTH - leg_size) / 2,
                                     end * (HOME_CHAIR_DEPTH - leg_size) / 2, leg_height / 2],
                [leg_size, leg_size, leg_height], "chair-leg")
    goal = {"id": f"seat:{identifier}", "position": seat_point, "kind": "sit", "entityId": identifier,
            "heading": yaw, "label": identifier}
    return geoms, goal


def create_home_scene() -> dict[str, Any]:
    """Current 10.12m home: major furniture, open table/chair leg spaces, four seats."""
    geoms = [_box("home-floor", [0, 0, -FLOOR_THICKNESS / 2],
                  [10.12, 10.12, FLOOR_THICKNESS], semantic="floor")]
    # Rendering uses a cutaway and zero-thickness planes; physical bounds use finite slabs.
    for identifier, position, size in [
        ("home-wall-left", [-5, 1.4, 0], [0.10, 2.8, 10.12]),
        ("home-wall-right", [5, 1.4, 0], [0.10, 2.8, 10.12]),
        ("home-wall-back", [0, 1.4, -5], [10.12, 2.8, 0.10]),
        ("home-wall-cutaway-front", [0, 1.4, 5], [10.12, 2.8, 0.10]),
    ]:
        _add_y_box(geoms, identifier, position, size, "wall")
    for identifier, position, size in [
        ("kitchen/toe-kick", [-4.66, 0.055, -1.62], [0.67, 0.11, 3.38]),
        ("kitchen/lower-run", [-4.64, 0.5, -1.62], [0.67, 0.78, 3.42]),
        ("kitchen/side-counter", [-4.59, 0.92, -1.62], [0.79, 0.055, 3.5]),
        ("kitchen/upper-cabinets", [-4.78, 2.12, -1.6], [0.4, 0.7, 2.48]),
        ("kitchen/back-cabinet", [-3.03, 0.44, -4.63], [2.35, 0.88, 0.66]),
        ("kitchen/back-counter", [-3.03, 0.92, -4.59], [2.43, 0.055, 0.77]),
        ("kitchen/back-splash", [-3.03, 1.16, -4.984], [2.43, 0.42, 0.023]),
        ("refrigerator", [-4.61, 1.175, -4.45], [0.74, 2.35, 0.81]),
        ("kitchen-island/base", [-3.04, 0.45, -1.28], [1.12, 0.86, 2.1]),
        ("kitchen-island/toe-kick", [-3.04, 0.05, -1.28], [0.96, 0.10, 1.92]),
        ("kitchen-island/counter", [-3.04, 0.925, -1.28], [1.24, 0.065, 2.23]),
        ("media-console", [4.78, 0.235, -1.97], [0.39, 0.37, 2.15]),
    ]:
        _add_y_box(geoms, identifier, position, size)
    # Sectional: retain separate base, chaise, cushions, arms and thin supporting feet.
    sofa_origin, sofa_yaw = [2.45, 0, -2.55], math.pi / 2
    for identifier, position, size, semantic in [
        ("sofa/base", [0, 0.25, 0], [2.55, 0.23, 0.91], "sofa-base"),
        ("sofa/back", [0, 0.6, -0.37], [2.49, 0.57, 0.20], "sofa-back"),
        ("sofa/chaise-base", [0.77, 0.27, 0.74], [0.77, 0.29, 1.48], "sofa-base"),
        ("sofa/chaise-cushion", [0.77, 0.46, 0.77], [0.71, 0.17, 1.43], "sofa-cushion"),
    ]:
        _add_group_box(geoms, identifier, position, size, sofa_origin, sofa_yaw, semantic)
    for side in (-1, 1):
        _add_group_box(geoms, f"sofa/arm-{side}", [side * 1.175, 0.48, 0], [0.20, 0.41, 0.94],
                       sofa_origin, sofa_yaw, "sofa-arm")
        for end, z in enumerate((-0.30, 0.31)):
            _add_group_box(geoms, f"sofa/foot-{side}-{end}", [side * 1.07, 0.07, z],
                           [0.066, 0.14, 0.066], sofa_origin, sofa_yaw, "furniture-foot")
    for i in range(3):
        _add_group_box(geoms, f"sofa/seat-cushion-{i}", [-0.75 + i * 0.75, 0.43, 0.04],
                       [0.71, 0.17, 0.70], sofa_origin, sofa_yaw, "sofa-cushion")
        _add_group_box(geoms, f"sofa/back-cushion-{i}", [-0.75 + i * 0.75, 0.70, -0.245],
                       [0.72, 0.43, 0.15], sofa_origin, sofa_yaw, "sofa-cushion")
    # Cylinder/rounded top uses a declared box envelope; the pedestal remains separate.
    _add_y_box(geoms, "coffee-table/top", [3.75, 0.355, -2], [1.04, 0.048, 0.8216], "tabletop", entity="coffee-table")
    _add_y_box(geoms, "coffee-table/pedestal", [3.75, 0.1625, -2], [0.46, 0.325, 0.46], "table-leg", entity="coffee-table")
    dining_origin = [-3.25, 0, 2.42]
    _add_group_box(geoms, "dining-table/top", [0, 0.765, 0], [1.46, 0.07, 2.15], dining_origin, 0,
                   "tabletop", "dining-table")
    _add_group_box(geoms, "dining-table/apron", [0, 0.68, 0], [1.22, 0.12, 1.91], dining_origin, 0,
                   "table-apron", "dining-table")
    for side in (-1, 1):
        for end in (-1, 1):
            _add_group_box(geoms, f"dining-table/leg-{side}-{end}", [side * 0.57, 0.335, end * 0.88],
                           [0.094, 0.67, 0.094], dining_origin, 0, "table-leg", "dining-table")
    goals = [
        {"id": "kitchen", "position": [-2.073, 1.28, 0], "kind": "walk", "heading": -math.pi / 2,
         "label": "キッチン（アイランド横）"},
        {"id": "dining", "position": [-2.173, -2.42, 0], "kind": "walk", "heading": -math.pi / 2,
         "label": "ダイニング（食卓横）"},
        {"id": "living", "position": [3.588, 0.928, 0], "kind": "walk", "heading": 2.289150974607008,
         "label": "リビング（ソファ前）"},
    ]
    chair_anchors = []
    for side in (-1, 1):
        for end in (-1, 1):
            identifier = f"dining-chair-{'wall' if side < 0 else 'aisle'}-{'back' if end < 0 else 'front'}"
            anchor_y_up = [-3.25 + side * 1.03, 0, 2.42 + end * 0.63]
            anchor = y_up_to_z_up(anchor_y_up)
            yaw = math.pi / 2 if side < 0 else -math.pi / 2
            pieces, goal = _measured_chair(identifier, anchor, yaw)
            geoms += pieces
            goals.append(goal)
            chair_anchors.append({"id": identifier, "anchorYUp": anchor_y_up, "yaw": yaw,
                                  "seatZUp": goal["position"]})
    # Collision-relevant floor items; leaves, lampshade fabric and books are display-only.
    for identifier, position, size in [
        ("living-plant/pot", [4.48, 0.16, -4.24], [0.40, 0.32, 0.40]),
        ("dining-plant/pot", [-4.52, 0.15, 4.08], [0.32, 0.30, 0.32]),
        ("floor-lamp/base", [2.04, 0.013, -4.22], [0.32, 0.026, 0.32]),
        ("floor-lamp/stem", [2.04, 0.76, -4.22], [0.022, 1.49, 0.022]),
        ("floor-lamp/shade", [2.04, 1.535, -4.22], [0.48, 0.31, 0.48]),
    ]:
        _add_y_box(geoms, identifier, position, size)
    return validate_scene({
        "version": 1, "stage": 1, "seed": 0, "name": "現在の住宅・物理接触モデル",
        "geoms": geoms, "spawn": [0, 0, 0], "goals": goals,
        "metadata": {"units": "m", "upAxis": "Z", "forwardAxis": "-Y", "spawnReference": "supporting_floor",
                     "sourceFile": "frontend/components/organisms/BvhViewer/homeEnvironment.ts",
                     "sourceNavigation": "roomNavigation.ts measured visible GLTF seat centres",
                     "floorSize": [10.12, 10.12], "floorBounds": [-5.06, -5.06, 5.06, 5.06],
                     "chairSeatHeight": HOME_CHAIR_SEAT_HEIGHT, "chairAnchors": chair_anchors,
                     "geometryOnly": True, "infinitePlane": False, "contactProxies": True,
                     "approximations": ["rounded furniture surfaces use oriented box contact proxies",
                                        "cylindrical feet and coffee table use separate box envelopes",
                                        "imported chairs retain measured seat centres with compound seat/back/feet",
                                        "cutaway zero-thickness walls become four finite boundary slabs",
                                        "decorative leaves, books, curtains, lights and screen images are omitted"]}})


def create_seated_task_scene(seat_distance: float = 0.30) -> dict[str, Any]:
    """A fixed chair behind a collision-free standing start; no navigation prerequisite."""
    if not isinstance(seat_distance, (float, int)) or not math.isfinite(seat_distance) or not 0.28 <= seat_distance <= 0.50:
        raise ValueError("seat_distance must be between 0.28 and 0.50 metres")
    # Solve the measured floor anchor from the desired seat centre, facing -Y.
    anchor = [-HOME_CHAIR_SEAT_LOCAL_X, seat_distance + HOME_CHAIR_SEAT_LOCAL_FORWARD, 0]
    pieces, goal = _measured_chair("training-chair", anchor, 0)
    return validate_scene({
        "version": 1, "stage": 1, "seed": 0, "name": "着座技能・後方の固定椅子",
        "geoms": _room_shell(6, 6) + pieces, "spawn": [0, 0, 0],
        "goals": [goal, {"id": "standing-front", "position": [0, -1, 0], "kind": "walk"}],
        "metadata": {"units": "m", "upAxis": "Z", "forwardAxis": "-Y", "spawnReference": "supporting_floor",
                     "geometryOnly": True, "infinitePlane": False, "contactProxies": True,
                     "navigationPrerequisite": False, "seatDistance": seat_distance,
                     "chairSeatHeight": HOME_CHAIR_SEAT_HEIGHT,
                     "targetSeatPosition": [0, seat_distance, HOME_CHAIR_SEAT_HEIGHT]}})
