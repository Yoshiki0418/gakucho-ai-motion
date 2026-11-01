"""G7: the visible two-storey house (about 53 tsubo) built around the production LDK.

1F keeps the LDK room of the conversation screen (``home_scene.create_home_scene``: 10 x 10 m, garden
windows on the north (+Y)), furnished in a typical Japanese layout (kitchen and dining north-west, living
facing the TV on the east wall, a tatami corner south-west),
and adds in front of it (south, -Y) the entrance, the hall, a straight staircase, a toilet, the
washroom and the bathroom. 2F (floor at 3.04 m = 16 risers x 0.19 m, treads 0.25 m) covers the
south block and the south 2 m of the LDK: master bedroom with a walk-in closet, two children's
rooms, a toilet and the hall with a study corner. Doors are openings (no door leaves yet).

One description serves physics (box contact proxies for ``navigation_eval.scene_elements``),
rendering (``objects``: class, pose and size of every fixture and piece of furniture) and the
G7 evaluation (``rooms``: name, floor and rectangles; ``objects``: ground-truth class and room).
Metres, Z up, the LDK centre at the origin, front door on the south facade.

    python -m world_model.house2f --output .runtime/world-model/house2f
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .home_scene import _measured_chair, create_home_scene
from .scenes import _box, validate_scene

TSUBO = 3.30579          # m2
WALL = 0.10              # m, wall thickness (as the LDK walls)
CEILING_1F = 2.80        # m, the LDK ceiling
RISE, RISERS, TREAD = 0.19, 16, 0.25
FLOOR_2F = round(RISE * RISERS, 4)   # 3.04 m: top of the 2F floor
WALL_2F = 2.40           # m, 2F room height
DOOR_HEIGHT = 2.0
GENKAN_DROP = 0.12       # m, the entrance floor (doma) is one low step below the hall
FRONT, BACK_2F = -7.8, -3.0          # y of the south facade and of the 2F north wall
STAIR_X0, STAIR_Y = -0.6, (-7.8, -6.8)   # first riser and the stair's wall lines
STAIR_X1 = STAIR_X0 + TREAD * (RISERS - 1)   # 3.15: where the top riser meets the 2F floor

# Rooms as rectangles [x0, y0, x1, y1] (wall centre lines). Every square metre of both floors
# belongs to exactly one entry (checked in ``check``).
ROOMS = [
    ("ldk", "LDK（リビング・ダイニング・キッチン）", 1, [[-5, -5, 5, 5]]),
    ("entrance", "玄関", 1, [[3.15, FRONT, 5, -7.0]]),
    ("hall-1f", "1階ホール・廊下", 1, [[-3.3, -6.1, 5, -5], [-0.6, -6.8, 3.15, -6.1], [-1.6, FRONT, -0.6, -6.1],
                                     [3.15, -7.0, 5, -6.1]]),
    ("stairs", "階段", 1, [[STAIR_X0, FRONT, STAIR_X1, -6.8]]),
    ("toilet-1f", "1階トイレ", 1, [[-5, -6.1, -3.3, -5]]),
    ("washroom", "洗面脱衣所", 1, [[-3.3, FRONT, -1.6, -6.1]]),
    ("bathroom", "浴室", 1, [[-5, FRONT, -3.3, -6.1]]),
    ("hall-2f", "2階ホール", 2, [[-2.5, -6.8, 4.15, -5.9], [3.15, FRONT, 4.15, -6.8]]),
    ("study-corner", "スタディコーナー", 2, [[-2.5, FRONT, -0.6, -6.8]]),
    ("stair-well", "階段（吹き抜け）", 2, [[STAIR_X0, FRONT, STAIR_X1, -6.8]]),
    ("master-bedroom", "主寝室", 2, [[-5, -5.9, -1, BACK_2F]]),
    ("walk-in-closet", "ウォークインクローゼット", 2, [[-5, FRONT, -2.5, -5.9]]),
    ("kids-room-1", "子ども部屋1", 2, [[-1, -5.9, 2, BACK_2F]]),
    ("kids-room-2", "子ども部屋2", 2, [[2, -5.9, 5, BACK_2F]]),
    ("toilet-2f", "2階トイレ", 2, [[4.15, FRONT, 5, -5.9]]),
]
# Places to walk to (ground truth for "go to the kitchen"): a free spot inside the room.
PLACES = {
    "living": ([2.9, -1.2], "ldk"), "kitchen": ([-3.975, 2.4], "ldk"), "dining": ([-0.95, 1.05], "ldk"),
    "tatami": ([-2.7, -3.6], "ldk"),
    "entrance": ([4.1, -6.55], "hall-1f"), "hall-1f": ([1.2, -5.55], "hall-1f"),
    "toilet-1f": ([-3.8, -5.55], "toilet-1f"), "washroom": ([-2.45, -6.6], "washroom"),
    "bathroom": ([-3.95, -6.6], "bathroom"),
    "hall-2f": ([1.0, -6.35], "hall-2f"), "study-corner": ([-1.55, -6.36], "study-corner"),
    "master-bedroom": ([-1.7, -4.0], "master-bedroom"), "walk-in-closet": ([-3.5, -6.85], "walk-in-closet"),
    "kids-room-1": ([0.75, -4.9], "kids-room-1"), "kids-room-2": ([3.3, -4.9], "kids-room-2"),
    "toilet-2f": ([4.575, -6.3], "toilet-2f"),
}


def _wall(geoms, identifier, a, b, z0, height, doors=(), door_height=DOOR_HEIGHT, semantic="wall"):
    """A straight wall on the centre line a-b (axis aligned) with door openings [(from, to), ...]
    measured along it; a lintel closes each opening above ``door_height``."""
    (x0, y0), (x1, y1) = a, b
    along_x = abs(y1 - y0) < 1e-9
    start, end = (min(x0, x1), max(x0, x1)) if along_x else (min(y0, y1), max(y0, y1))
    line = y0 if along_x else x0
    cuts = sorted(doors)

    def piece(name, s, e, bottom, top):
        if e - s < 1e-6 or top - bottom < 1e-6:
            return
        centre = [(s + e) / 2, line] if along_x else [line, (s + e) / 2]
        size = [e - s, WALL] if along_x else [WALL, e - s]
        geoms.append(_box(name, [*centre, (bottom + top) / 2], [*size, top - bottom], semantic=semantic))
    cursor = start
    for i, (s, e) in enumerate(cuts):
        piece(f"{identifier}-{i}", cursor, s, z0, z0 + height)
        piece(f"{identifier}-lintel-{i}", s, e, z0 + door_height, z0 + height)
        cursor = e
    piece(f"{identifier}-{len(cuts)}", cursor, end, z0, z0 + height)


def _rect_box(identifier, rect, bottom, top, semantic):
    x0, y0, x1, y1 = rect
    return _box(identifier, [(x0 + x1) / 2, (y0 + y1) / 2, (bottom + top) / 2], [x1 - x0, y1 - y0, top - bottom],
                semantic=semantic)


def create_house() -> dict[str, Any]:
    home = create_home_scene()
    # only the room: its floor and the west, east and north (garden window) walls
    geoms = [g for g in home["geoms"] if g["id"] in ("home-floor", "home-wall-left", "home-wall-right", "home-wall-back")]
    objects: list[dict[str, Any]] = []

    def thing(identifier, cls, label, room, floor, centre, size, yaw=0., base=0., collide=True, coco=None, **extra):
        """A recognisable object: size = [width, depth, height] in its own frame, its front faces its
        local -Y (yaw 0: south, pi/2: east, pi: north, -pi/2: west); ``base`` = bottom above that floor."""
        z0 = (0. if floor == 1 else FLOOR_2F) + base
        entry = {"id": identifier, "class": cls, "label": label, "room": room, "floor": floor,
                 "position": [float(centre[0]), float(centre[1]), z0 + size[2] / 2], "size": list(size),
                 "yaw": yaw, "coco": coco, "collision": collide, **extra}
        objects.append(entry)
        if collide:
            geoms.append(_box(identifier, entry["position"], size, yaw, semantic=cls, entityId=identifier))

    # ---- floors: the LDK floor comes with the home; the south block, the lowered doma, the 2F slab
    geoms.append(_rect_box("floor-1f-south", [-5.06, FRONT - 0.05, STAIR_X1, -5.06], -0.08, 0., "floor"))
    geoms.append(_rect_box("floor-1f-entrance-hall", [STAIR_X1, -7.0, 5.06, -5.06], -0.08, 0., "floor"))
    geoms.append(_rect_box("floor-doma", [STAIR_X1, FRONT - 0.05, 5.06, -7.0], -GENKAN_DROP - 0.08, -GENKAN_DROP, "floor"))
    slab = (CEILING_1F, FLOOR_2F)
    for name, rect in [("west", [-5.06, FRONT - 0.05, STAIR_X0, BACK_2F + 0.05]),
                       ("east", [STAIR_X1, FRONT - 0.05, 5.06, BACK_2F + 0.05]),
                       ("middle", [STAIR_X0, STAIR_Y[1], STAIR_X1, BACK_2F + 0.05])]:
        geoms.append(_rect_box(f"floor-2f-{name}", rect, *slab, "floor"))

    # ---- 1F walls (the LDK's west, east and north walls come with the home)
    low = -GENKAN_DROP
    _wall(geoms, "wall-ldk-south", (-4.95, -5), (4.95, -5), 0., CEILING_1F, doors=[(0.0, 0.9)])
    _wall(geoms, "wall-1f-front", (-5.06, FRONT), (5.06, FRONT), low, CEILING_1F - low, doors=[(3.6, 4.5)],
          door_height=DOOR_HEIGHT - low)
    _wall(geoms, "wall-1f-west", (-5, FRONT), (-5, -5.06), 0., CEILING_1F)
    _wall(geoms, "wall-1f-east", (5, FRONT), (5, -5.06), low, CEILING_1F - low)
    _wall(geoms, "wall-wet-north", (-4.95, -6.1), (-1.6, -6.1), 0., CEILING_1F, doors=[(-2.8, -2.1)])
    _wall(geoms, "wall-wet-split", (-3.3, FRONT + 0.05), (-3.3, -5.05), 0., CEILING_1F,
          doors=[(-6.95, -6.25), (-5.95, -5.2)])
    _wall(geoms, "wall-washroom-east", (-1.6, FRONT + 0.05), (-1.6, -6.15), 0., CEILING_1F)
    _wall(geoms, "wall-stair-1f", (STAIR_X0, -6.8), (STAIR_X1, -6.8), 0., CEILING_1F)

    # ---- the straight staircase: 15 solid treads, the 16th riser reaches the 2F floor at STAIR_X1
    for k in range(1, RISERS):
        x0 = STAIR_X0 + TREAD * (k - 1)
        geoms.append(_box(f"stair-tread-{k:02d}", [x0 + TREAD / 2, (STAIR_Y[0] + STAIR_Y[1]) / 2, RISE * k / 2],
                          [TREAD, STAIR_Y[1] - STAIR_Y[0] - WALL, RISE * k], semantic="stair", entityId="staircase"))

    # ---- 2F walls and the railing round the stair well
    z2 = FLOOR_2F
    _wall(geoms, "wall-2f-front", (-5.06, FRONT), (5.06, FRONT), z2, WALL_2F)
    _wall(geoms, "wall-2f-back", (-5.06, BACK_2F), (5.06, BACK_2F), z2, WALL_2F)
    _wall(geoms, "wall-2f-west", (-5, FRONT + 0.05), (-5, BACK_2F - 0.05), z2, WALL_2F)
    _wall(geoms, "wall-2f-east", (5, FRONT + 0.05), (5, BACK_2F - 0.05), z2, WALL_2F)
    _wall(geoms, "wall-2f-rooms", (-4.95, -5.9), (4.95, -5.9), z2, WALL_2F,
          doors=[(-4.3, -3.6), (-2.3, -1.5), (0.1, 0.9), (3.1, 3.9)])
    _wall(geoms, "wall-2f-master-kids", (-1, -5.85), (-1, BACK_2F - 0.05), z2, WALL_2F)
    _wall(geoms, "wall-2f-kids", (2, -5.85), (2, BACK_2F - 0.05), z2, WALL_2F)
    _wall(geoms, "wall-2f-closet", (-2.5, FRONT + 0.05), (-2.5, -5.95), z2, WALL_2F)
    _wall(geoms, "wall-2f-toilet", (4.15, FRONT + 0.05), (4.15, -5.95), z2, WALL_2F, doors=[(-6.7, -6.0)])
    _wall(geoms, "railing-well-side", (STAIR_X0, -6.8), (STAIR_X1, -6.8), z2, 0.9, semantic="railing")
    _wall(geoms, "railing-well-end", (STAIR_X0, FRONT + 0.05), (STAIR_X0, -6.85), z2, 0.9, semantic="railing")

    # ---- LDK: the room, its floor and outer walls are the production home's; the furniture is a typical
    # Japanese family layout: a face-to-face kitchen in the north-west (cupboard and appliances on the wall,
    # the counter with sink and IH facing the dining), the dining table beside it, the living facing the TV on
    # the east wall, a flush tatami corner in the south-west. Dining chairs are the home's measured CC0 chair.
    pi = math.pi
    thing("cupboard", "cupboard", "カップボード（家電収納）", "ldk", 1, [-4.725, 2.4], [2.4, 0.45, 0.85], pi / 2)
    thing("wall-cabinet", "wall cabinet", "吊り戸棚", "ldk", 1, [-4.775, 2.4], [2.4, 0.35, 0.7], pi / 2, base=1.45)
    thing("microwave", "microwave", "電子レンジ", "ldk", 1, [-4.76, 3.1], [0.5, 0.38, 0.3], pi / 2, base=0.85,
          collide=False, coco="microwave")
    thing("rice-cooker", "rice cooker", "炊飯器", "ldk", 1, [-4.76, 2.45], [0.27, 0.32, 0.22], pi / 2, base=0.85, collide=False)
    thing("toaster", "toaster", "トースター", "ldk", 1, [-4.78, 1.95], [0.3, 0.2, 0.2], pi / 2, base=0.85, collide=False,
          coco="toaster")
    thing("kettle", "kettle", "電気ケトル", "ldk", 1, [-4.78, 1.5], [0.2, 0.18, 0.22], pi / 2, base=0.85, collide=False)
    thing("refrigerator", "refrigerator", "冷蔵庫", "ldk", 1, [-4.58, 3.9925], [0.685, 0.7, 1.82], pi / 2, coco="refrigerator")
    thing("trash-bins", "trash bin", "ゴミ箱", "ldk", 1, [-4.72, 4.65], [0.6, 0.38, 0.62], pi / 2)
    thing("tall-cupboard", "cupboard", "食器棚（ガラス扉）", "ldk", 1, [-3.35, 4.725], [1.2, 0.45, 1.9], 0., variant="glass")
    thing("kitchen-counter", "kitchen counter", "対面キッチンのカウンター", "ldk", 1, [-3.0, 2.475], [2.55, 0.9, 1.05], -pi / 2)
    thing("kitchen-sink", "sink", "キッチンのシンク", "ldk", 1, [-3.15, 3.05], [0.75, 0.45, 0.2], -pi / 2, base=0.65,
          collide=False, coco="sink", variant="kitchen")
    thing("cooktop", "stove", "IHコンロ", "ldk", 1, [-3.15, 1.85], [0.6, 0.5, 0.02], -pi / 2, base=0.85, collide=False,
          coco="oven")
    thing("pot", "pot", "鍋", "ldk", 1, [-3.2, 1.72], [0.24, 0.24, 0.14], base=0.87, collide=False)
    thing("range-hood", "range hood", "レンジフード", "ldk", 1, [-3.15, 1.85], [0.75, 0.55, 0.3], -pi / 2, base=1.95)
    thing("dish-rack", "dish rack", "水切りかご", "ldk", 1, [-3.15, 3.58], [0.33, 0.3, 0.15], -pi / 2, base=0.85, collide=False)
    for i, y in enumerate((1.85, 3.05)):
        thing(f"pendant-kitchen-{i}", "pendant light", "ペンダントライト", "ldk", 1, [-2.9, y], [0.22, 0.22, 0.3], base=1.95,
              collide=False)
    thing("dining-table", "dining table", "ダイニングテーブル（4人掛け）", "ldk", 1, [-0.95, 2.45], [0.85, 1.6, 0.72],
          coco="dining table")
    seats = []
    for identifier, x, y, yaw in [("dining-chair-west-south", -1.675, 2.05, pi / 2), ("dining-chair-west-north", -1.675, 2.85, pi / 2),
                                  ("dining-chair-east-south", -0.225, 2.05, -pi / 2), ("dining-chair-east-north", -0.225, 2.85, -pi / 2)]:
        pieces, seat = _measured_chair(identifier, [x, y, 0.], yaw)
        geoms.extend(pieces)
        seats.append(seat)
        thing(identifier, "chair", "ダイニングチェア", "ldk", 1, [x, y], [0.43, 0.54, 0.95], yaw, collide=False, coco="chair")
    thing("dining-vase", "vase", "花瓶", "ldk", 1, [-0.95, 2.3], [0.12, 0.12, 0.3], base=0.72, collide=False, coco="vase")
    thing("fruit-bowl", "bowl", "果物の器", "ldk", 1, [-0.95, 2.85], [0.3, 0.3, 0.1], base=0.72, collide=False, coco="bowl")
    for i, y in enumerate((2.1, 2.8)):
        thing(f"pendant-dining-{i}", "pendant light", "ペンダントライト", "ldk", 1, [-0.95, y], [0.3, 0.3, 0.35], base=1.75,
              collide=False)
    thing("tv-board", "tv board", "テレビボード", "ldk", 1, [4.74, 0.3], [1.8, 0.42, 0.42], -pi / 2)
    thing("tv", "tv", "テレビ", "ldk", 1, [4.76, 0.3], [1.23, 0.25, 0.77], -pi / 2, base=0.42, collide=False, coco="tv")
    thing("sofa", "sofa", "ソファ（3人掛け）", "ldk", 1, [1.75, 0.3], [2.1, 0.9, 0.82], pi / 2, coco="couch")
    thing("coffee-table", "coffee table", "ローテーブル", "ldk", 1, [3.05, 0.3], [1.0, 0.5, 0.38], pi / 2)
    thing("rug", "rug", "ラグ", "ldk", 1, [2.9, 0.3], [2.4, 1.9, 0.012], pi / 2, collide=False)
    thing("armchair", "armchair", "1人掛けソファ", "ldk", 1, [3.0, 2.1], [0.78, 0.78, 0.8], 0., coco="chair")
    thing("side-table", "side table", "サイドテーブル", "ldk", 1, [1.75, 1.65], [0.45, 0.45, 0.55])
    thing("living-lamp", "table lamp", "スタンドライト", "ldk", 1, [1.75, 1.65], [0.25, 0.25, 0.4], base=0.55, collide=False)
    thing("floor-lamp", "floor lamp", "フロアランプ", "ldk", 1, [4.4, 3.55], [0.4, 0.4, 1.6])
    thing("living-bookshelf", "bookshelf", "本棚（リビング）", "ldk", 1, [4.8, 2.7], [0.8, 0.3, 1.2], -pi / 2)
    thing("living-books", "book", "本", "ldk", 1, [4.8, 2.7], [0.7, 0.22, 0.25], -pi / 2, base=0.4, collide=False, coco="book")
    thing("sideboard", "sideboard", "サイドボード", "ldk", 1, [3.6, -4.75], [1.8, 0.4, 0.8], pi)
    thing("sideboard-books", "book", "本", "ldk", 1, [3.0, -4.75], [0.4, 0.2, 0.25], pi, base=0.8, collide=False, coco="book")
    thing("sideboard-vase", "vase", "花瓶", "ldk", 1, [4.1, -4.78], [0.14, 0.14, 0.32], base=0.8, collide=False, coco="vase")
    thing("air-conditioner", "air conditioner", "エアコン", "ldk", 1, [3.6, -4.83], [0.8, 0.24, 0.3], pi, base=2.2, collide=False)
    thing("ldk-clock", "clock", "掛け時計", "ldk", 1, [1.9, -4.93], [0.3, 0.04, 0.3], pi, base=1.8, collide=False, coco="clock")
    thing("living-plant", "potted plant", "観葉植物（リビング）", "ldk", 1, [4.45, 4.4], [0.6, 0.6, 1.4], coco="potted plant")
    thing("window-plant", "potted plant", "観葉植物（窓辺）", "ldk", 1, [-1.35, 4.5], [0.45, 0.45, 1.0], coco="potted plant")
    thing("tatami", "tatami", "畳コーナー", "ldk", 1, [-3.6, -3.6], [2.7, 2.7, 0.02], collide=False)
    thing("low-table", "low table", "座卓", "ldk", 1, [-3.6, -3.6], [0.9, 0.75, 0.33])
    thing("zabuton-0", "cushion", "座布団", "ldk", 1, [-3.6, -4.3], [0.55, 0.6, 0.08], collide=False)
    thing("zabuton-1", "cushion", "座布団", "ldk", 1, [-3.6, -2.9], [0.55, 0.6, 0.08], pi, collide=False)
    thing("tatami-chest", "dresser", "チェスト（畳コーナー）", "ldk", 1, [-4.72, -2.85], [0.9, 0.45, 0.75], pi / 2)

    # ---- 1F south block
    thing("shoe-cabinet", "shoe cabinet", "下駄箱", "hall-1f", 1, [4.79, -6.05], [1.5, 0.32, 0.9], -math.pi / 2)
    thing("umbrella-stand", "umbrella stand", "傘立て", "entrance", 1, [4.75, -7.55], [0.25, 0.25, 0.5], base=-GENKAN_DROP)
    thing("umbrella", "umbrella", "傘", "entrance", 1, [4.75, -7.55], [0.12, 0.12, 0.85], base=-GENKAN_DROP,
          collide=False, coco="umbrella")
    thing("entrance-shoes", "shoes", "靴", "entrance", 1, [3.9, -7.2], [0.3, 0.28, 0.12], base=-GENKAN_DROP, collide=False)
    thing("toilet-1f-bowl", "toilet", "トイレ（便器）", "toilet-1f", 1, [-4.6, -5.55], [0.42, 0.7, 0.8], math.pi / 2,
          coco="toilet")
    thing("washroom-vanity", "sink", "洗面台", "washroom", 1, [-2.05, -7.48], [0.75, 0.55, 0.85], math.pi, coco="sink")
    thing("washroom-mirror", "mirror", "洗面台の鏡", "washroom", 1, [-2.05, -7.73], [0.7, 0.03, 0.75], math.pi, base=1.05, collide=False)
    thing("washing-machine", "washing machine", "洗濯機", "washroom", 1, [-2.95, -7.45], [0.6, 0.6, 1.0], math.pi)
    thing("hair-drier", "hair drier", "ドライヤー", "washroom", 1, [-1.85, -7.42], [0.2, 0.08, 0.08], base=0.85,
          collide=False, coco="hair drier")
    thing("toothbrush-cup", "toothbrush", "歯ブラシ", "washroom", 1, [-2.3, -7.6], [0.07, 0.07, 0.2], base=0.85,
          collide=False, coco="toothbrush")
    thing("bathtub", "bathtub", "浴槽", "bathroom", 1, [-4.15, -7.37], [1.6, 0.75, 0.55], math.pi)
    thing("shower", "shower", "シャワー", "bathroom", 1, [-3.45, -6.2], [0.1, 0.1, 0.3], base=1.6, collide=False)
    thing("bath-stool", "stool", "風呂いす", "bathroom", 1, [-4.7, -6.45], [0.3, 0.25, 0.25])

    # ---- 2F
    thing("master-bed", "bed", "ダブルベッド", "master-bedroom", 2, [-3.0, -4.0], [1.4, 2.0, 0.55], coco="bed")
    for i, x in enumerate((-4.0, -2.0)):
        thing(f"nightstand-{i}", "nightstand", "ナイトテーブル", "master-bedroom", 2, [x, -3.27], [0.45, 0.4, 0.55])
        thing(f"bedside-lamp-{i}", "table lamp", "スタンドライト", "master-bedroom", 2, [x, -3.27], [0.25, 0.25, 0.4],
              base=0.55, collide=False)
    thing("dresser", "dresser", "チェスト", "master-bedroom", 2, [-4.7, -4.95], [0.9, 0.5, 0.85], math.pi / 2)
    thing("master-clock", "clock", "掛け時計", "master-bedroom", 2, [-4.3, -3.07], [0.3, 0.04, 0.3], base=1.9,
          collide=False, coco="clock")
    thing("master-plant", "potted plant", "観葉植物（寝室）", "master-bedroom", 2, [-1.3, -3.3], [0.4, 0.4, 0.9],
          coco="potted plant")
    thing("clothes-rack", "clothes rack", "ハンガーラック", "walk-in-closet", 2, [-4.7, -6.85], [1.75, 0.5, 1.8], math.pi / 2)
    thing("closet-shelf", "shelf", "収納棚", "walk-in-closet", 2, [-3.75, -7.55], [1.4, 0.4, 1.8], math.pi)
    for n, (bed_x, desk_x, chair_x) in enumerate([(-0.47, 1.65, 1.17), (4.47, 2.35, 2.83)], start=1):
        room = f"kids-room-{n}"
        thing(f"kids-bed-{n}", "bed", f"シングルベッド（子ども部屋{n}）", room, 2, [bed_x, -3.98], [0.97, 1.95, 0.5], coco="bed")
        thing(f"kids-desk-{n}", "desk", f"学習机（子ども部屋{n}）", room, 2, [desk_x, -3.7], [1.0, 0.6, 0.72], -math.pi / 2 if n == 1 else math.pi / 2)
        thing(f"kids-chair-{n}", "chair", f"いす（子ども部屋{n}）", room, 2, [chair_x, -3.7], [0.45, 0.45, 0.85],
              math.pi / 2 if n == 1 else -math.pi / 2, coco="chair")
        thing(f"kids-shelf-{n}", "bookshelf", f"本棚（子ども部屋{n}）", room, 2, [desk_x, -5.6], [0.6, 0.3, 1.2], math.pi)
        thing(f"kids-books-{n}", "book", "本", room, 2, [desk_x, -5.6], [0.5, 0.22, 0.25], math.pi, base=0.6, collide=False, coco="book")
    thing("kids-laptop", "laptop", "ノートパソコン", "kids-room-1", 2, [1.65, -3.5], [0.33, 0.23, 0.02], -math.pi / 2, base=0.72,
          collide=False, coco="laptop")
    thing("kids-backpack", "backpack", "ランドセル", "kids-room-1", 2, [1.75, -4.45], [0.3, 0.2, 0.35], coco="backpack")
    thing("teddy-bear", "teddy bear", "くまのぬいぐるみ", "kids-room-1", 2, [-0.47, -3.2], [0.3, 0.2, 0.3], base=0.5,
          collide=False, coco="teddy bear")
    thing("sports-ball", "sports ball", "サッカーボール", "kids-room-2", 2, [3.3, -3.4], [0.22, 0.22, 0.22], coco="sports ball")
    thing("study-desk", "desk", "スタディコーナーの机", "study-corner", 2, [-1.55, -7.48], [1.5, 0.55, 0.72], math.pi)
    thing("study-chair", "chair", "スタディコーナーのいす", "study-corner", 2, [-1.55, -7.0], [0.45, 0.45, 0.85], 0.,
          coco="chair")
    thing("study-monitor", "monitor", "モニター", "study-corner", 2, [-1.55, -7.62], [0.6, 0.15, 0.4], math.pi, base=0.72,
          collide=False, coco="tv")
    thing("toilet-2f-bowl", "toilet", "トイレ（便器）", "toilet-2f", 2, [4.575, -7.4], [0.42, 0.7, 0.8], math.pi, coco="toilet")

    rooms = [{"id": i, "label": label, "floor": floor, "rects": rects,
              "area": round(sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects), 2)} for i, label, floor, rects in ROOMS]
    goals = [{"id": name, "position": [*xy, 0. if next(r for r in rooms if r["id"] == room)["floor"] == 1 else FLOOR_2F],
              "kind": "walk", "room": room} for name, (xy, room) in PLACES.items()]
    goals += seats
    area = {f: round(sum(r["area"] for r in rooms if r["floor"] == f and r["id"] != "stair-well"), 1) for f in (1, 2)}
    area[2] = round(area[2] + next(r["area"] for r in rooms if r["id"] == "stair-well"), 1)   # the well counts on 2F
    scene = {"version": 1, "stage": 3, "seed": 0, "name": "2階建ての家（本番のLDKを1階に持つ約53坪）",
             "geoms": geoms, "spawn": [*PLACES["hall-1f"][0], 0.], "goals": goals,
             "metadata": {"units": "m", "upAxis": "Z", "forwardAxis": "-Y", "spawnReference": "supporting_floor",
                          "geometryOnly": False, "infinitePlane": False, "storeys": 2, "floorHeights": [0., FLOOR_2F],
                          "ceilingHeights": [CEILING_1F, FLOOR_2F + WALL_2F],
                          "stair": {"firstRiserX": STAIR_X0, "topX": STAIR_X1, "y": list(STAIR_Y), "rise": RISE,
                                    "risers": RISERS, "tread": TREAD, "direction": "+X up"},
                          "entranceStep": GENKAN_DROP,
                          "floorArea": {"1F": area[1], "2F": area[2], "total": round(area[1] + area[2], 1),
                                        "tsubo": round((area[1] + area[2]) / TSUBO, 1)},
                          "ldkSource": "room, floor and outer walls of home_scene.create_home_scene; furniture laid out here",
                          "ldkZones": {"kitchen": [-5, 0.8, -2.3, 5], "dining": [-2.3, 0.8, 0.6, 5],
                                       "living": [0.6, -2.0, 5, 5], "tatami": [-4.95, -4.95, -2.25, -2.25]},
                          "doors": "openings without leaves"},
             "rooms": rooms, "objects": objects}
    return validate_scene(scene)


def check(scene) -> dict[str, Any]:
    """Room rectangles tile both floors; every walk place is reachable from the hall on its floor
    (5 cm grid, avatar radius 0.28 m); the bottom and top of the stair are reachable."""
    import numpy as np
    from collections import deque
    report: dict[str, Any] = {}
    for floor, (lo, hi) in ((1, (0.05, 1.8)), (2, (FLOOR_2F + 0.05, FLOOR_2F + 1.8))):
        rects = [r for room in scene["rooms"] if room["floor"] == floor for r in room["rects"]]
        covered = sum((r[2] - r[0]) * (r[3] - r[1]) for r in rects)
        bounds = (-5, FRONT, 5, 5 if floor == 1 else BACK_2F)
        report[f"{floor}F tiled"] = abs(covered - (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) < 1e-6
        cell, x0, y0 = 0.05, -5.2, FRONT - 0.2
        nx, ny = int(10.4 / cell), int((bounds[3] + 0.2 - y0) / cell)
        blocked = np.zeros((nx, ny), bool)
        xs = x0 + (np.arange(nx) + .5) * cell
        ys = y0 + (np.arange(ny) + .5) * cell
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        for g in scene["geoms"]:
            if g["semantic"] in ("floor", "stair"):
                continue
            z, h = g["position"][2], g["size"][2]
            if z + h / 2 < lo or z - h / 2 > hi:
                continue
            c, s = math.cos(g.get("yaw", 0.)), math.sin(g.get("yaw", 0.))
            dx, dy = gx - g["position"][0], gy - g["position"][1]
            u, v = c * dx + s * dy, -s * dx + c * dy
            blocked |= (np.abs(u) <= g["size"][0] / 2 + 0.28) & (np.abs(v) <= g["size"][1] / 2 + 0.28)
        if floor == 1:   # the stair is not walkable floor on 1F; the stair well is not floor on 2F
            blocked |= (gx > STAIR_X0 - 0.02) & (gx < STAIR_X1) & (gy > STAIR_Y[0]) & (gy < STAIR_Y[1])
        else:
            blocked |= (gx > STAIR_X0 - 0.3) & (gx < STAIR_X1) & (gy > STAIR_Y[0]) & (gy < STAIR_Y[1] + 0.3)
        blocked |= (gx < -5) | (gx > 5) | (gy < FRONT) | (gy > bounds[3])
        index = lambda p: (int((p[0] - x0) / cell), int((p[1] - y0) / cell))
        start = index(PLACES["hall-1f" if floor == 1 else "hall-2f"][0])
        seen = np.zeros_like(blocked)
        queue = deque([start])
        seen[start] = True
        while queue:
            i, j = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not blocked[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        places = [p for p, (_, room) in PLACES.items()
                  if next(r for r in scene["rooms"] if r["id"] == room)["floor"] == floor]
        report[f"{floor}F reachable"] = {p: bool(seen[index(PLACES[p][0])]) for p in places}
        stair_end = [STAIR_X0 - 0.35, sum(STAIR_Y) / 2] if floor == 1 else [STAIR_X1 + 0.4, sum(STAIR_Y) / 2]
        report[f"{floor}F stair {'bottom' if floor == 1 else 'top'} reachable"] = bool(seen[index(stair_end)])
    return report


def plot(scene, path):
    """Floor plans of both floors (walls, furniture, objects, places)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Polygon, Rectangle
    for name in ("Noto Sans CJK JP", "Noto Serif CJK JP"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    fig, axes = plt.subplots(1, 2, figsize=(17, 12.5), gridspec_kw={"width_ratios": [1, 1]})
    colours = {"wall": "#3b3f45", "railing": "#8a6d3b", "stair": "#b9a27c"}
    for ax, floor in zip(axes, (1, 2)):
        lo, hi = ((-0.2, 1.2) if floor == 1 else (FLOOR_2F - 0.01, FLOOR_2F + 1.2))   # plan cut at 1.2 m
        for room in scene["rooms"]:
            if room["floor"] != floor:
                continue
            for r in room["rects"]:
                ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], facecolor="#f4efe6", edgecolor="none"))
        for o in scene["objects"]:   # floor finishes: tatami and rugs
            if o["floor"] == floor and o["class"] in ("tatami", "rug"):
                c, s_ = math.cos(o["yaw"]), math.sin(o["yaw"])
                w, d = o["size"][0] / 2, o["size"][1] / 2
                pts = [(o["position"][0] + c * dx - s_ * dy, o["position"][1] + s_ * dx + c * dy) for dx, dy in ((-w, -d), (w, -d), (w, d), (-w, d))]
                ax.add_patch(Polygon(pts, closed=True, facecolor="#d9d6a8" if o["class"] == "tatami" else "#e2d8c6", edgecolor="#a39d78", lw=0.6))
                ax.text(o["position"][0], o["position"][1] - (0.55 if o["class"] == "tatami" else 0.9), o["label"], ha="center", fontsize=7, color="#5b5a3a")
        for g in scene["geoms"]:
            z, h = g["position"][2], g["size"][2]
            top, bottom = z + h / 2, z - h / 2
            if g["semantic"] == "floor" or bottom >= hi - 1e-6 or top <= lo + 1e-6:
                continue
            if floor == 2 and g["semantic"] == "stair":
                continue
            c, s = math.cos(g.get("yaw", 0.)), math.sin(g.get("yaw", 0.))
            corners = [(g["position"][0] + c * dx - s * dy, g["position"][1] + s * dx + c * dy)
                       for dx, dy in [(-g["size"][0] / 2, -g["size"][1] / 2), (g["size"][0] / 2, -g["size"][1] / 2),
                                      (g["size"][0] / 2, g["size"][1] / 2), (-g["size"][0] / 2, g["size"][1] / 2)]]
            colour = colours.get(g["semantic"], "#c9b79c" if g["semantic"] != "furniture" else "#a99f92")
            ax.add_patch(Polygon(corners, closed=True, facecolor=colour, edgecolor="#55493a" if g["semantic"] not in
                                 ("wall", "railing") else colour, linewidth=0.4, alpha=0.95))
        if floor == 2:
            ax.add_patch(Rectangle((STAIR_X0, STAIR_Y[0]), STAIR_X1 - STAIR_X0, STAIR_Y[1] - STAIR_Y[0],
                                   facecolor="none", edgecolor="#8a6d3b", hatch="//", linewidth=0.8))
        else:
            ax.annotate("", xy=(STAIR_X1 - 0.15, sum(STAIR_Y) / 2), xytext=(STAIR_X0 + 0.15, sum(STAIR_Y) / 2),
                        arrowprops={"arrowstyle": "->", "color": "#5a3f1c", "lw": 1.4})
            ax.text(sum((STAIR_X0, STAIR_X1)) / 2, sum(STAIR_Y) / 2 + 0.2, f"上り {RISERS}段", ha="center", fontsize=8,
                    color="#5a3f1c")
        for room in scene["rooms"]:
            if room["floor"] != floor or room["id"] in ("stairs", "stair-well"):
                continue
            r = max(room["rects"], key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            if room["id"] == "ldk":
                cx, cy = 0.2, 0.0
            ax.text(cx, cy, f"{room['label']}\n{room['area']:.1f}㎡（{room['area'] / 1.62:.1f}畳）", ha="center",
                    va="center", fontsize=8.5 if room["area"] > 3 else 7, color="#1d2b3a", weight="bold")
        for o in scene["objects"]:
            if o["floor"] == floor and o["room"] != "ldk":
                ax.text(o["position"][0], o["position"][1], o["label"].split("（")[0], ha="center", va="center",
                        fontsize=6.5, color="#4a3418")
        for o in scene["objects"]:
            if o["floor"] == floor and o["room"] == "ldk" and o["id"] in ("refrigerator", "cupboard", "kitchen-counter", "tall-cupboard",
                                                                        "sofa", "tv-board", "dining-table", "armchair", "sideboard", "low-table", "tatami-chest"):
                ax.text(o["position"][0], o["position"][1], o["label"], ha="center", va="center", fontsize=6.5,
                        color="#4a3418")
        for g in scene["goals"]:
            if g["kind"] == "walk" and (g["position"][2] > 1) == (floor == 2):
                ax.plot(*g["position"][:2], "o", color="#2f7fd0", markersize=4)
        ax.set_xlim(-5.6, 5.6)
        ax.set_ylim(FRONT - 0.6, 5.6)
        ax.set_aspect("equal")
        ax.set_title(f"{floor}階（床 {0 if floor == 1 else FLOOR_2F:.2f} m）  床面積 {scene['metadata']['floorArea'][f'{floor}F']}㎡",
                     fontsize=12)
        ax.text(4.05, FRONT - 0.35, "玄関ドア ↑" if floor == 1 else "", ha="center", fontsize=8)
        ax.text(0, 5.3, "庭（大きな窓）" if floor == 1 else "（1階 LDK の屋根）", ha="center", fontsize=8, color="#4f6b48")
        ax.set_xticks(range(-5, 6))
        ax.set_yticks(range(-7, 6))
        ax.grid(alpha=0.15)
    a = scene["metadata"]["floorArea"]
    fig.suptitle(f"2階建ての家（延床 {a['total']}㎡・約 {a['tsubo']}坪）  1階のLDKは本番の家の部屋（家具は一般的な配置）  青い点＝行き先の正解位置",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    scene = create_house()
    (out / "house.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    report = check(scene)
    plot(scene, out / "plan.png")
    print(json.dumps({"floorArea": scene["metadata"]["floorArea"], "geoms": len(scene["geoms"]),
                      "objects": len(scene["objects"]), "check": report}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
