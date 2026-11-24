"""G7: the executor - a local VLM decides from the head camera what to do next; the body does it.

    python -m world_model.house_agent --output .runtime/world-model/house2f/agent [--only kitchen]

One episode = one Japanese instruction in the visible two-storey house (``house2f``), starting on the first
floor, closed loop in native MuJoCo with the trained whole-body tracker:

* look: the head depth image (``house_tour_depth.Map``: obstacles and seen-free floor), the colour
  image of the same camera (``tools/house_render_server.mjs``, the house page), YOLO on it
  (``vision_worker``), and every detection placed in 3-D with the depth image (object memory)
* scan: after every look Qwen3-VL frames, in that one image, the destination itself, things only found
  there (clues) and the ways out of the room (doors); the frames are placed with the depth image too
* decide: Qwen3-VL (ollama, ``--vlm``) gets the instruction, the latest colour images, the remembered
  objects and places and the unexplored places (direction and distance from the body, stable ids) and
  answers one tool as JSON: look_around / go_to_object / explore / finish
* act: walking uses the depth map only (A*, stop-and-look segments of at most 2.5 m), the captured
  walk bent onto the route and turns on the spot, followed by the tracker (``house_tour_depth``)
* stairs: the VLM frames the stairs; going to them walks up (or down) the flight (``use_stairs``) and the
  other floor gets its own depth map and remembered places

No scene geometry or object list is read for deciding or planning; they only score the result. The one
exception is the staircase, a fixture the body is set up for (where it starts and its step sizes).
Every look and decision is logged with its images for the web page (/world-model/house).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / ".runtime/world-model/tracker"
PUBLIC = ROOT / ".runtime/world-model/monitor/public"
VISION_PYTHON = ROOT / ".runtime/world-model/vision-venv/bin/python"
YOLO_WEIGHTS = ROOT / ".runtime/world-model/house2f/vision/yolo11x.pt"
EYE_PITCH, HFOV, WIDTH, HEIGHT = 15., 90., 640, 480
HEAD_ROOM = 1.9          # m: obstacles end here (the body is 1.7 m tall; door lintels at 2.0 m are not in the way)
HEAD_TURN = 100.         # degrees the head turns (body still) to see a remembered destination off to one side
MAX_DECISIONS = 14
SIDE_CAMERAS = (60., -60.)   # degrees: two more depth cameras on the head, turned left and right (depth only, no colour)
DROP_Z = -.06            # m: floor seen this far below the floor (the 12 cm step down to the entrance) is not walked on
SHADOW = .5              # m2: unseen pockets smaller than this inside seen floor are furniture shadows, not new places
SPAWN = (0.45, -4.3)     # just inside the LDK door from the hall, facing north (the hall is 1.1 m wide)
MAX_SEGMENTS_PER_MOVE = 10
STAND_OFF = 1.2          # m from a remembered object to where the body stops (outside narrow aisles)
TOUCH = .6               # m: never stop nearer to a remembered object (its front surface) than this
FLOORS = {1: 0., 2: 3.04}   # floor heights (house2f.FLOOR_2F)
STAIR_WALKS = ROOT / ".runtime/world-model/tracker/house-stair-walks-v15.json"   # walks the tracker follows on these stairs
HARD, SOFT = .3, .75     # m: routes keep at least HARD from obstacles (the 1F toilet door is 0.75 m) and pay for less than SOFT
CLOSE_ENOUGH = .6        # m: nearer to the goal than this the body stays where it is if getting there needs a big turn in a tight spot
NARROW = .45             # m: a passage this tight (a door) is walked along its centre line, not cut across
TURN_ROOM = .55          # m: turning on the spot swings a toe 0.42 m out and drifts 0.16 m; closer obstacles get kicked
# a tighter captured turn (「前から後ろに振り向く」, to the right): toe 0.33 m, drift 0.08 m from -45 to -190 degrees with
# the v14 tracker in physics; mirrored to the left it drifts 0.9 m or falls, so it only turns right
TIGHT_TURN, TIGHT_ROOM = "0008/02/004/001/005.bvh", .42
ARMS_IN = .25            # rad on the upper arm's y hinge (left +, right -): hands 0.12-0.15 m from the body line instead
                         # of 0.21-0.27 m, as a person keeps the arms in walking through a narrow door
JA = {"refrigerator": "冷蔵庫", "oven": "コンロ", "sink": "シンク・洗面台", "microwave": "電子レンジ", "toaster": "トースター",
      "dining table": "テーブル", "chair": "椅子", "couch": "ソファ", "tv": "テレビ", "potted plant": "観葉植物",
      "vase": "花瓶", "bowl": "器", "clock": "時計", "toilet": "トイレ（便器）", "bed": "ベッド", "laptop": "ノートパソコン",
      "book": "本", "umbrella": "傘", "backpack": "かばん", "teddy bear": "ぬいぐるみ", "sports ball": "ボール",
      "hair drier": "ドライヤー", "toothbrush": "歯ブラシ", "bench": "ベンチ", "cup": "カップ", "knife": "ナイフ",
      "bottle": "びん", "wine glass": "グラス", "remote": "リモコン", "cell phone": "スマートフォン", "keyboard": "キーボード"}
BIG = {"refrigerator", "couch", "bed", "dining table", "tv", "toilet", "oven", "sink"}
# floor-standing things whose thin legs the 64x48 depth image misses: a recognised one is added to the map as a disc
FOOTPRINT = {"chair": .42, "dining table": .65, "couch": .8, "bed": .8, "bench": .5, "toilet": .3, "potted plant": .3,
             "refrigerator": .45}
# Instructions and how they are scored (ground truth from house2f; never shown to the agent)
TASKS = [
    {"id": "kitchen", "text": "キッチンに行って", "kind": "zone", "target": "kitchen"},
    {"id": "fridge", "text": "冷蔵庫の前まで来て", "kind": "object", "target": "refrigerator"},
    {"id": "tv", "text": "テレビの前に行って", "kind": "object", "target": "tv"},
    {"id": "toilet", "text": "トイレに行って", "kind": "room", "target": "toilet-1f"},
    {"id": "washroom", "text": "洗面所に行って", "kind": "room", "target": "washroom"},
    {"id": "tatami", "text": "畳のコーナーに行って", "kind": "zone", "target": "tatami"},
    {"id": "kids", "text": "子ども部屋に行って", "kind": "room", "target": ["kids-room-1", "kids-room-2"]},
    {"id": "garage", "text": "車庫に行って", "kind": "unable", "target": None},
]
PROMPT = """あなたは家庭用ロボット（学長アバター）の実行係です。ユーザーの指示を達成するために、次の行動を1つだけ選びます。
家は2階建てで、ロボットはいま{floor}階にいます。家の間取りは分かりません。部屋は今いる部屋の外（出入口の先の廊下など）にもあります。
【階段】へ go_to_object すると、階段を{stairs}の階へ移動します。寝室・子ども部屋は2階にあることが多いです。
今いる部屋がLDK（リビング・ダイニング・キッチン）なら、トイレ・洗面所・浴室・玄関・寝室・子ども部屋はその外（出入口の先）にあります。

指示：「{instruction}」

添付の画像（ロボットの頭のカメラ、新しい順）：
{images}

これまでに見つけた物・場所（ロボットから見た向きと距離）：
{objects}

まだ見ていない所（行って確かめられる候補）：
{frontiers}

これまでの行動：
{history}

選べる行動：
- look_around：その場で一回りして周りを見る
- go_to_object：見つけた物・場所・出入口・階段へ行く（target に O の番号。階段なら別の階へ移動する）
- explore：まだ見ていない所へ行く（target に F の番号）
- finish：終わる（success に true か false、message に一言）

決め方：
1. 【行き先】が見つかっていれば、遠くても部屋の外かどうかに関係なく、まずその前へ go_to_object。前に着いて画像で確かめられたら finish（success true）
2. 【手がかり】（行き先の部屋にある物）が見つかっていれば、その前へ行って確かめる
3. 今いる部屋で見つからなければ、【出入口】の先や、まだ見ていない所へ行く。この階にないと分かったら【階段】で別の階へ。新しい場所に入ったら look_around
4. 同じ所へ何度も行かない。まだ見ていない所が残っている間は、見つからないという理由で finish（false）にしない。家のどこにもない（そもそも家にない物・部屋）と確かに分かったときだけ finish（false）

JSON だけで答えてください：{{"thought": "状況の判断（短く）", "action": "look_around|go_to_object|explore|finish", "target": "", "success": true, "message": ""}}"""
SCAN = """ロボットの頭のカメラ画像です。ユーザーの指示は「{instruction}」。
この画像に次のものが写っていれば挙げてください。
- target：指示の行き先そのもの（部屋・場所・物）
- clue：行き先が部屋や場所のとき、その部屋にしかない物（例：キッチンなら冷蔵庫・コンロ・シンク、トイレなら便器、洗面所なら洗面台・洗濯機）
- door：今いる場所から別の場所へ出る出入口（扉・開口・廊下の先）。画像の端に写っていても挙げる。窓と階段は含めない
- stairs：別の階へ行く階段（上り口・下り口）
位置は画像の枠 [x0,y0,x1,y1]（左上0・右下1000 の相対座標）。place は、部屋や場所（床の広がり）なら true、物なら false。
target と clue は確かに見分けられる物だけを、同じ物は一度だけ（端に少しだけ写っている物や、似ているだけの物は挙げない）。部屋を target に
するのは、その部屋の中が確かに写っているときだけ（扉や廊下しか見えなければ door）。なければ空のリスト。
JSON だけで答えてください：{{"items": [{{"kind": "target|clue|door|stairs", "what": "名前", "place": false, "box": [x0,y0,x1,y1]}}]}}"""
VERIFY = """ロボットの頭のカメラの画像です（1枚目がいまの正面。床は約1.2m先から写ります）。ユーザーの指示は「{instruction}」。
ロボットはいま、この指示の行き先に着いていますか？行き先そのものが目の前に確かに写っているかで判断してください。
部屋なら、その部屋の中に入っているか（廊下や別の部屋から扉越しに見えているだけなら着いていません）、コーナーなどの場所ならその上か目の前か。
似ているだけの物なら着いていません。
JSON だけで答えてください：{{"arrived": true, "reason": "短く"}}"""
KIND = {"target": "行き先", "clue": "手がかり", "door": "出入口", "stairs": "階段"}


class Stop(Exception):
    pass


def direction_text(angle):
    """Angle of a point in the body frame (radians, + = left) as a Japanese direction."""
    deg = math.degrees(angle)
    for limit, name in ((22.5, "正面"), (67.5, "左前"), (112.5, "左"), (157.5, "左後ろ")):
        if abs(deg) <= limit:
            return name if deg >= 0 or name == "正面" else name.replace("左", "右")
    return "後ろ"


class Agent:
    def __init__(self, args, out):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax.numpy as jp
        import mujoco
        from .house2f import create_house
        from .tracker import body as B
        from .tracker.home_navigation import _turn_clip
        from .tracker import depth_map
        from .tracker.house_tour_depth import Map
        from .tracker.navigation_eval import _load, scene_elements
        # rays hitting the underside of a door lintel land at 2.0 m and used to close the doorway (this process only)
        depth_map.OBSTACLE_Z = (depth_map.OBSTACLE_Z[0], HEAD_ROOM)
        from .tracker.path_reference import _align, _yaw_quat
        self.jp, self.mujoco, self.B, self.out, self.args = jp, mujoco, B, out, args
        self.scene = create_house()
        run_dir = TRACKER / "runs" / args.model
        library = TRACKER / "motion-library-v1"
        self.run, self.policy, self.config, self.env, self.observe_fn, bank = _load(run_dir, run_dir / "final_params.pkl", library)
        self.walk, self.bank = bank[0][1], dict(bank)
        from .tracker.home_navigation import _clip
        self.turn, self.plain = _turn_clip(library), B.build_model()
        self.tight = _clip(library, TIGHT_TURN).astype(np.float64)
        self.walk_in = self.walk.copy()       # the same walk with the arms kept in (doors)
        for side, sign in (("left", 1.), ("right", -1.)):
            self.walk_in[:, self.plain.jnt_qposadr[self.plain.joint(f"{side}_upper_arm_y").id]] += sign * ARMS_IN
        model = B.build_model(floor=None, extra_world=scene_elements(self.scene))
        model.opt.iterations, model.opt.ls_iterations = self.config.solver_iterations, self.config.solver_ls_iterations
        self.model, self.data = model, mujoco.MjData(model)
        nb = self.nb = len(B.SEGMENTS)
        self.body_geoms = {g for g in range(model.ngeom) if 1 <= model.geom_bodyid[g] <= nb}
        self.names = {model.geom(g["id"]).id: g["id"] for g in self.scene["geoms"] if g.get("semantic") != "floor"}
        self.treads = {model.geom(g["id"]).id for g in self.scene["geoms"] if g.get("semantic") == "stair"}
        self.floor, self.on_stairs, self.floors = 1, False, {}   # the floor it is on; other floors' maps
        self.map = Map((-5.2, -8.0), (5.2, 5.2))
        self.table_size = 40000
        self.table = np.zeros((self.table_size, model.nq), np.float32)
        spawn = np.asarray(SPAWN, float)
        anchor = self.turn[0].copy()
        anchor[:2] = spawn
        anchor[3:7] = _yaw_quat(np.asarray(math.pi))                 # facing north, into the LDK door
        self.reference = _align(self.turn[:1].copy(), anchor)
        self.table[:20] = self.reference[0]
        self.data.qpos[:] = self.reference[0]
        mujoco.mj_forward(model, self.data)
        self.state = {"frame": 0, "contacts": [], "fell": None,
                      "info": {"clip": jp.int32(0), "frame": jp.int32(0), "last_act": jp.zeros(model.nu), "linvel": jp.zeros((nb, 3))}}
        self.rng = __import__("jax").random.PRNGKey(0)
        self.trajectory = [self.data.qpos.copy()]
        self.map.frame_index = lambda: len(self.trajectory) - 1
        self.memory, self.events, self.recent_images, self.history = [], [], [], []
        self.looks = 0
        self.doing, self.route = "", None
        self.views = {}                                   # look -> (eye, pixel rays, depth image) for placing frames
        self.discs = np.zeros_like(self.map.grid.occupied)   # cells marked from recognised objects
        self.hits = self.map.grid.occupied.copy()            # cells marked from depth hits (and steps down)
        self.walked = np.zeros_like(self.map.grid.occupied)  # cells the body has stood on: never an object disc
        self.frontier_ids, self.dead = [], []             # stable frontier ids; frontiers already gone to
        self.instruction, self.refused = None, {"give_up": 0, "far": 0, "check": 0}
        self.rejected = []                                # places the VLM framed as the destination but were not
        self.tried = set()                                # remembered things already gone to
        self.unreachable = set()                          # remembered things with no route to them
        self.renderer = subprocess.Popen(["node", str(ROOT / "tools/house_render_server.mjs"), args.base], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, text=True, bufsize=1)
        self.detector = subprocess.Popen([str(VISION_PYTHON), str(ROOT / "world_model/vision_worker.py"), str(YOLO_WEIGHTS)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=out)
        for proc, name in ((self.renderer, "描画"), (self.detector, "物体検出")):
            line = proc.stdout.readline().strip()
            if line != "ready":
                raise RuntimeError(f"{name}の係が起動しない: {line}")

    def close(self):
        for proc in (self.renderer, self.detector):
            proc.stdin.close()
            proc.wait(timeout=60)

    # ------------------------------------------------------------------ body (as house_tour_depth)
    def append(self, segment, blend=True, placed=False):
        """Add ``segment`` to the reference. Walks and turns are made on a floor at height 0 and are raised to the
        floor the body is on; a ``placed`` one (a hold, a stair flight) is in the world already."""
        from .tracker.house_tour import blend_join
        if not placed and FLOORS[self.floor]:
            segment = segment.copy()
            segment[:, 2] += FLOORS[self.floor]
        self.reference = blend_join(self.reference, segment) if blend else np.concatenate([self.reference, segment])
        if len(self.reference) + 20 > self.table_size:
            raise Stop("動作が長すぎる")
        self.table[:len(self.reference)] = self.reference
        self.table[len(self.reference):len(self.reference) + 20] = self.reference[-1]

    def hold(self, frames):
        self.append(np.repeat(self.reference[-1:], frames, axis=0), blend=False, placed=True)

    def simulate(self):
        jp, mujoco, data, model, state, nb = self.jp, self.mujoco, self.data, self.model, self.state, self.nb
        jtable, jlength = jp.asarray(self.table), jp.int32(len(self.reference) + 20)
        while state["frame"] < len(self.reference) - 1:
            frame = state["frame"] + 1
            view = {"xpos": jp.asarray(data.xpos), "xquat": jp.asarray(data.xquat), "cvel": jp.asarray(data.cvel),
                    "qpos": jp.asarray(data.qpos), "qvel": jp.asarray(data.qvel)}
            action = np.clip(np.asarray(self.policy(self.observe_fn(view, state["info"], jtable, jlength), self.rng)[0]), -1., 1.)
            data.ctrl[:] = self.table[frame, 7:] + self.config.action_scale * action
            prev = data.xpos[1:nb + 1].copy()
            for _ in range(self.B.CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
                for c in data.contact[:data.ncon]:
                    pair = (c.geom1, c.geom2)
                    hit = [g for g in pair if g in self.names and not (self.on_stairs and g in self.treads)]
                    if hit and any(g in self.body_geoms for g in pair) and not any(
                            c["obstacle"] == self.names[hit[0]] and data.time - c["time"] < 1. for c in state["contacts"][-5:]):
                        part = next(g for g in pair if g in self.body_geoms)
                        state["contacts"].append({"time": round(float(data.time), 2), "obstacle": self.names[hit[0]],
                                                  "part": model.body(model.geom_bodyid[part]).name, "doing": self.doing,
                                                  "frame": int(state["frame"]),
                                                  "pelvis": [round(float(v), 2) for v in data.qpos[:2]]})
            state["frame"] = frame
            self.trajectory.append(data.qpos.copy())
            state["info"] = {"clip": jp.int32(0), "frame": jp.int32(frame), "last_act": jp.asarray(action),
                             "linvel": jp.asarray((data.xpos[1:nb + 1] - prev) * 50.)}
            up = data.xmat[1].reshape(3, 3)[2, 2]
            if up < .5 or data.qpos[2] - self.ground(data.qpos[0]) < .55:
                state["fell"] = float(data.time)
                raise Stop("転倒")


    def ground(self, x):
        """Height of what the body stands on: the stair under it on the flight, else the floor it is on."""
        from .tracker.house_stairs import surface
        return float(surface(x)) if self.on_stairs else FLOORS[self.floor]

    def heading(self):
        r = self.data.xmat[1].reshape(3, 3)
        return math.atan2(r[1, 0], r[0, 0])

    def turn_plan(self, angle):
        """Captured turns (clip, angle) that turn the body by ``angle`` (radians, + = left). Right turns, and left turns
        past 165 degrees done the other way round, use the tight turn; a smaller left turn uses the wider captured
        left turn where there is room for it, else two tight right turns of half the way round."""
        if angle < 0:
            return [(self.tight, angle)]
        if angle >= math.radians(165):
            return [(self.tight, angle - 2 * math.pi)]
        if self.clearance(self.reference[-1, :2]) >= TURN_ROOM:
            return [(self.turn, angle)]
        half = (angle - 2 * math.pi) / 2
        return [(self.tight, half), (self.tight, half)]

    def turning(self, angle):
        """The reference of the turns of ``turn_plan``, starting where the body stands now."""
        from .tracker.path_reference import _align, scale_turn
        out = None
        for clip, part in self.turn_plan(angle):
            step = _align(scale_turn(clip, part, self.plain), self.reference[-1] if out is None else out[-1])
            out = step if out is None else np.concatenate([out, step[1:]])
        return out

    def turn_to(self, yaw):
        """Turn on the spot to body yaw ``yaw`` (captured turn, scaled)."""
        from .tracker.path_reference import root_yaw
        angle = (yaw - root_yaw(self.reference[-1:])[0] + math.pi) % (2 * math.pi) - math.pi
        if abs(angle) < math.radians(20.):
            return False
        self.doing = "その場で" + "・".join(f"{math.degrees(a):+.0f}°" for _, a in self.turn_plan(angle)) + "回る"
        self.append(self.turning(angle))
        self.hold(25)
        self.simulate()
        return True

    def walk_route(self, route, arc=False):
        """Walk ``route``: turn on the spot to its first direction, then walk; with ``arc`` start walking the way
        the body faces and bend onto the route while walking (no turn on the spot). Through a narrow passage the
        arms are kept in."""
        from .tracker.path_reference import _align, _quat_mul, _yaw_quat, follow, join, root_yaw
        direction = math.atan2(*(route[1] - route[0])[::-1])
        samples = np.concatenate([np.linspace(a, b, max(int(np.linalg.norm(b - a) / .05), 2)) for a, b in zip(route[:-1], route[1:])])
        narrow = min(self.clearance(p) for p in samples) < NARROW
        walk = self.walk_in if narrow else self.walk
        if arc:
            walked, _ = follow(walk, route, start_direction=self.heading() - math.pi / 2, ramp=.5)
            leg, angle = _align(walked, self.reference[-1]), 0.
        else:
            walked, _ = follow(walk, route, start_direction=direction)
            angle = (root_yaw(walked[:1])[0] - root_yaw(self.reference[-1:])[0] + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) < math.radians(20.):
                leg, angle = walked, 0.
            else:
                # as path_reference.turn_then, with the turns of turn_plan: placed to end where the walk starts
                turned = self.turning(angle)
                delta = root_yaw(walked[:1])[0] - root_yaw(turned[-1:])[0]
                rel = turned[:, :2] - turned[-1, :2]
                c, s_ = math.cos(delta), math.sin(delta)
                placed = turned.copy()
                placed[:, :2] = walked[0, :2] + np.stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]], 1)
                placed[:, 3:7] = _quat_mul(_yaw_quat(np.full(len(placed), delta)), turned[:, 3:7])
                leg = join(placed, walked)
        self.doing = (f"{math.degrees(angle):+.0f}°回ってから" if angle else "") + \
            f"{'腕を寄せて' if narrow else ''}歩く（{float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()):.1f}m、{len(self.reference)}フレーム目から）"
        self.route = route
        self.append(leg)
        self.hold(25)
        self.simulate()

    # ------------------------------------------------------------------ perception
    def look(self, why):
        from .depth_perception import DepthCameraConfig, _camera_pose, _pixel_rays
        from types import SimpleNamespace
        g = self.map.grid
        g.occupied = self.hits.copy()          # the object discs are drawn again below from the memory
        self.map.observe(self.model, self.data)
        frame, depth, obstacle = self.map.shots[-1]
        for xy in np.asarray(self.trajectory)[::10, :2].tolist() + [self.data.qpos[:2].tolist()]:
            i, j = g.index(xy)
            r = int(.3 / g.cell)
            self.walked[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1] = True
        cfg = DepthCameraConfig()
        _, origin, basis, _ = _camera_pose(SimpleNamespace(model=self.model, data=self.data), cfg)
        heading = self.heading()
        self.looks += 1
        k = self.looks
        self.renderer.stdin.write(json.dumps({"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2]), "heading": heading,
                                              "pitch": EYE_PITCH, "width": WIDTH, "height": HEIGHT, "hfov": HFOV}) + "\n")
        reply = json.loads(self.renderer.stdout.readline())
        rgb = self.out / f"look-{k:02d}.jpg"
        rgb.write_bytes(base64.b64decode(reply["jpeg"]))
        self.detector.stdin.write(str(rgb) + "\n")
        detections = json.loads(self.detector.stdout.readline()).get("detections", [])
        rays = _pixel_rays(cfg) @ basis
        self.views[k] = (origin, rays, depth)
        # floor seen below the floor is a step down (the entrance): kept off like an obstacle
        hits = origin + rays * np.maximum(depth.reshape(-1), 0.)[:, None]
        drop = (depth.reshape(-1) > 0) & (hits[:, 2] - FLOORS[self.floor] < DROP_Z)
        self.map.grid.mark(hits[drop, :2])
        side = sum(self.glance(math.radians(a)) for a in SIDE_CAMERAS)
        self.hits = g.occupied.copy()
        placed = []
        for d in detections:
            point = self.place(k, d["box"])
            if point is None:
                continue
            if d["coco"] in FOOTPRINT:     # the depth hits are its front: its middle is further away
                ray = point[:2] - origin[:2]
                point[:2] += ray / max(np.linalg.norm(ray), 1e-6) * FOOTPRINT[d["coco"]] * .5
            d["position"] = [round(float(v), 2) for v in point]
            placed.append(d)
            self.remember(d)
        self.mark_objects()
        self.recent_images.append(rgb)
        event = {"kind": "look", "why": why, "index": k, "time": round(frame * .02, 2), "frame": int(frame),
                 "pose": [round(float(self.data.qpos[0]), 2), round(float(self.data.qpos[1]), 2), round(heading, 3)],
                 "eye": [round(float(v), 3) for v in origin], "rgb": rgb.name, "detections": placed,
                 "obstacles": int(obstacle.sum()), "drops": int(drop.sum()), "sideObstacles": int(side)}
        if self.instruction:
            event["vlm"] = self.scan(k, rgb)
        self.draw_look(event, depth, obstacle)
        self.events.append(event)
        return event

    def glance(self, yaw):
        """One shot of a side depth camera (the front one turned ``yaw`` about the vertical): obstacles, steps down
        and seen floor go into the map as from the front camera (``house_tour_depth.Map.observe``)."""
        from types import SimpleNamespace
        from .depth_perception import DepthCameraConfig, _camera_pose, _cast, _filtered_model, _pixel_rays
        from .tracker.depth_map import OBSTACLE_Z
        cfg = DepthCameraConfig()
        _, origin, basis, root = _camera_pose(SimpleNamespace(model=self.model, data=self.data), cfg)
        c, s = math.cos(yaw), math.sin(yaw)
        basis = basis @ np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]).T
        sensor, query, _ = _filtered_model(self.model, root)
        self.mujoco.mj_copyData(query, self.model, self.data)
        rays = np.ascontiguousarray(_pixel_rays(cfg) @ basis)
        ranges, ids = _cast(sensor, query, origin, rays, cfg.far_meters, "multi_ray")
        hit = (ids >= 0) & (ranges >= 0)
        pts = origin + rays * np.where(hit, ranges, cfg.far_meters)[:, None]
        above = pts[:, 2] - FLOORS[self.floor]
        obstacle = hit & (((above > OBSTACLE_Z[0]) & (above < OBSTACLE_Z[1])) | (above < DROP_Z))
        g = self.map.grid
        g.mark(pts[obstacle, :2])
        # every ray crosses free floor up to its end
        o, ends = origin[:2], pts[:, :2]
        steps = np.maximum((np.linalg.norm(ends - o, axis=1) / g.cell).astype(int), 1)
        ray = np.repeat(np.arange(len(ends)), steps)
        frac = (np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)) / np.repeat(steps, steps)
        ij = g.index(o + (ends[ray] - o) * frac[:, None])
        ok = np.all((ij >= 0) & (ij < self.map.seen.shape), axis=1)
        self.map.seen[ij[ok, 0], ij[ok, 1]] = True
        return int(obstacle.sum())

    def box_points(self, k, box):
        """Every depth hit inside the pixel box of look ``k`` (the depth image's columns run to the body's
        left, see publish_results.encode_shot); heights above the floor the body is on."""
        origin, rays, depth = self.views[k]
        u0, v0, u1, v1 = box
        rows = range(max(0, int(v0 / 10)), min(48, int(math.ceil(v1 / 10))))
        cols = range(max(0, 63 - int(math.ceil(u1 / 10)) + 1), min(64, 63 - int(u0 / 10) + 1))
        pts = [origin + rays[r * 64 + c] * float(depth[r, c]) for r in rows for c in cols if depth[r, c] > 0]
        return origin, np.asarray(pts).reshape(-1, 3) - (0., 0., FLOORS[self.floor])

    def place(self, k, box):
        """Where a framed object stands: only hits at obstacle heights belong to it (the rest passed between
        chair legs or under a table to the floor or a wall behind), and of those its front surface."""
        origin, pts = self.box_points(k, box)
        pts = pts[(pts[:, 2] > .08) & (pts[:, 2] < 2.)]
        if len(pts) < 3:
            return None
        near = np.linalg.norm(pts[:, :2] - origin[:2], axis=1)
        return np.median(pts[near <= np.sort(near)[1] + .5], axis=0)

    def scan(self, k, rgb):
        """Ask the VLM what of the instruction this one image shows; frames become remembered places."""
        text, seconds = self.ask(SCAN.format(instruction=self.instruction), [rgb])
        try:
            items = json.loads(text).get("items", [])
        except (ValueError, AttributeError):
            items = []
        found, names = [], set()
        for item in items if isinstance(items, list) else []:
            kind, what, box = item.get("kind"), str(item.get("what", "")).strip(), item.get("box")
            if kind not in KIND or not what or what in names or not (isinstance(box, list) and len(box) == 4):
                continue
            names.add(what)
            box = [float(np.clip(v, 0, 1000)) / 1000. * s for v, s in zip(box, (WIDTH, HEIGHT, WIDTH, HEIGHT))]
            origin, pts = self.box_points(k, box)
            floor = pts[np.abs(pts[:, 2]) < .06] if len(pts) else pts
            solid = pts[(pts[:, 2] > .08) & (pts[:, 2] < 2.)] if len(pts) else pts
            if kind == "stairs":
                # the flight (treads above the floor going up, below it going down): the nearer half of what is framed
                if len(pts) < 3:
                    continue
                near = np.linalg.norm(pts[:, :2] - origin[:2], axis=1)
                point, on_floor = np.median(pts[near <= np.median(near)], axis=0), False
            elif kind == "door":
                # a way out is where the floor goes on beyond it: the far half of the floor seen through the frame
                if len(floor) < 3:
                    continue
                far = np.linalg.norm(floor[:, :2] - origin[:2], axis=1)
                point, on_floor = np.median(floor[far >= np.median(far)], axis=0), True
            elif len(floor) >= 3 and (box[2] - box[0]) * (box[3] - box[1]) >= .02 * WIDTH * HEIGHT and \
                    (item.get("place") or len(solid) < max(3, .3 * len(pts))):
                point, on_floor = np.median(floor, axis=0), True     # a place: its floor (the tatami corner, the kitchen)
            elif len(solid) >= 3:
                point, on_floor = self.place(k, box), False
            else:
                continue
            if kind not in ("door", "stairs") and any(np.linalg.norm(point[:2] - r) < 1.2 for r in self.rejected):
                continue
            entry = {"kind": kind, "what": what, "box": [round(v, 1) for v in box],
                     "position": [round(float(v), 2) for v in point], "floor": on_floor}
            entry["id"] = self.remember_place(entry)
            found.append(entry)
        return {"items": found, "seconds": seconds}

    def remember_place(self, e):
        p = np.asarray(e["position"][:2])
        for m in self.known():
            if m["coco"] == "vlm" and m["kind"] == e["kind"] and (m["name"] == e["what"] or e["kind"] == "stairs") and \
                    np.linalg.norm(np.asarray(m["position"][:2]) - p) < {"door": .8, "stairs": 2.}.get(e["kind"], 1.2):
                n = m["seen"]
                m["position"] = [round((a * n + b) / (n + 1), 2) for a, b in zip(m["position"], e["position"])]
                m["seen"] = n + 1
                return m["id"]
        self.memory.append({"id": f"O{len(self.memory) + 1}", "coco": "vlm", "kind": e["kind"], "name": e["what"],
                            "position": e["position"], "seen": 1, "conf": 1., "floor": e["floor"], "level": self.floor})
        return self.memory[-1]["id"]

    def remember(self, d):
        p = np.asarray(d["position"][:2])
        reach = 1.0 if d["coco"] in BIG else .6
        for m in self.known():
            if m["coco"] == d["coco"] and np.linalg.norm(np.asarray(m["position"][:2]) - p) < reach:
                n = m["seen"]
                m["position"] = [round((a * n + b) / (n + 1), 2) for a, b in zip(m["position"], d["position"])]
                m["seen"] = n + 1
                m["conf"] = max(m["conf"], d["conf"])
                return
        self.memory.append({"id": f"O{len(self.memory) + 1}", "coco": d["coco"], "name": JA.get(d["coco"], d["coco"]),
                            "position": d["position"], "seen": 1, "conf": d["conf"], "level": self.floor})

    def known(self):
        """The remembered things and places of the floor the body is on."""
        return [m for m in self.memory if m.get("level", 1) == self.floor]

    def mark_objects(self):
        """Recognised floor-standing objects become obstacle discs in the depth map (drawn again from the memory
        after every look, as the remembered positions settle; never where the body has stood)."""
        g = self.map.grid
        self.discs[:] = False
        for m in self.known():
            radius = FOOTPRINT.get(m["coco"])
            if radius is None or (m["conf"] < .5 and m["seen"] < 2):
                continue
            ci, cj = g.index(m["position"][:2])
            r = int(math.ceil(radius / g.cell))
            ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
            keep = (ii ** 2 + jj ** 2) * g.cell ** 2 <= radius ** 2
            ii, jj = ii[keep] + ci, jj[keep] + cj
            ok = (ii >= 0) & (jj >= 0) & (ii < g.occupied.shape[0]) & (jj < g.occupied.shape[1])
            self.discs[ii[ok], jj[ok]] = True
        self.discs &= ~self.walked
        g.occupied = self.hits | self.discs

    def clearance(self, point, unseen_blocks=False):
        """Distance (m) from ``point`` to the nearest obstacle cell (or unseen cell)."""
        from scipy.ndimage import distance_transform_edt
        g = self.map.grid
        solid = g.occupied | (~self.map.seen if unseen_blocks else False)
        i, j = g.index(point)
        if not (0 <= i < solid.shape[0] and 0 <= j < solid.shape[1]):
            return 0.
        r = int(1.5 / g.cell)
        window = solid[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
        if not window.any():
            return 1.5
        dist = distance_transform_edt(~window) * g.cell
        return float(dist[min(i, r), min(j, r)])

    def relative(self, point):
        x, y = self.data.qpos[0], self.data.qpos[1]
        h = self.heading()
        d = np.asarray(point[:2]) - (x, y)
        forward, left = d @ (math.sin(h), -math.cos(h)), d @ (math.cos(h), math.sin(h))
        return direction_text(math.atan2(left, forward)), float(np.hypot(*d))

    def frontiers(self):
        """Edges between seen free floor and unseen floor, largest first; ids stay with a place across decisions
        and places already gone to are dropped."""
        from scipy import ndimage
        g = self.map.grid
        seen = self.map.seen
        free = seen & ~g.blocked(.3)
        unseen = ~seen & ~g.occupied
        # small unseen pockets inside seen floor are shadows (under a table, behind a counter), not new places
        labels, _ = ndimage.label(unseen)
        sizes = np.bincount(labels.ravel())
        unseen &= sizes[labels] * g.cell ** 2 >= SHADOW
        edge = free & ndimage.binary_dilation(unseen, iterations=1)
        # next to a recognised chair or table the unseen floor is its shadow too (walls and door frames do not count:
        # the way into the hall is right beside them)
        if self.discs.any():
            edge &= ndimage.distance_transform_edt(~self.discs) * g.cell >= .6
        labels, count = ndimage.label(edge, structure=np.ones((3, 3)))
        found = []
        for i in range(1, count + 1):
            cells = np.argwhere(labels == i)
            if len(cells) < 8:
                continue
            centre = cells.mean(0)
            cell = cells[np.argmin(np.linalg.norm(cells - centre, axis=1))]
            found.append((len(cells), g.centre(cell)))
        found.sort(key=lambda t: -t[0])
        out, used = [], set()
        for n, p in found:
            if any(np.linalg.norm(p - d) < 1. for d in self.dead):
                continue
            known = next((f for f in self.frontier_ids if f["id"] not in used and np.linalg.norm(p - f["position"]) < 1.), None)
            if known is None:
                known = {"id": f"F{len(self.frontier_ids) + 1}"}
                self.frontier_ids.append(known)
            known["position"] = p
            used.add(known["id"])
            out.append({"id": known["id"], "position": [round(float(p[0]), 2), round(float(p[1]), 2)], "size": int(n)})
        return out[:5]

    # ------------------------------------------------------------------ deciding
    def ask(self, prompt, images):
        body = {"model": self.args.vlm_model, "stream": False, "format": "json", "think": False,
                "options": {"temperature": 0, "num_ctx": 8192},
                "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(p.read_bytes()).decode() for p in images]}]}
        started = time.time()
        request = urllib.request.Request(f"{self.args.vlm}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            text = json.loads(response.read()).get("message", {}).get("content", "")
        return text, round(time.time() - started, 2)

    def decide(self, instruction):
        here = self.data.qpos[:2]
        places = sorted((m for m in self.known() if m["coco"] == "vlm"), key=lambda m: ("tcds".index(m["kind"][0]), m["id"]))
        seen = sorted((m for m in self.known() if m["coco"] != "vlm" and (m["conf"] >= .45 or m["seen"] >= 2)),
                      key=lambda m: np.linalg.norm(np.asarray(m["position"][:2]) - here))[:15]
        objects = places + seen
        lines = []
        for m in objects:
            where, dist = self.relative(m["position"])
            label = f"【{KIND[m['kind']]}】" if m["coco"] == "vlm" else ""
            lines.append(f"{m['id']} {label}{m['name']}（{where} {dist:.1f}m{'、行けなかった' if m['id'] in self.unreachable else ''}）")
        fronts = self.frontiers()
        flines = []
        for f in fronts:
            where, dist = self.relative(f["position"])
            flines.append(f"{f['id']} {where} {dist:.1f}m（広さ{'大' if f['size'] > 40 else '中' if f['size'] > 15 else '小'}）")
        images = list(reversed(self.recent_images[-4:]))
        captions = []
        for i, p in enumerate(images):
            shot = next(e for e in self.events if e["kind"] == "look" and e["rgb"] == p.name)
            if i == 0 and shot is self.last_look():
                captions.append(f"画像1：いまの正面（最新）")
                continue
            where = direction_text((shot["pose"][2] - self.heading() + math.pi) % (2 * math.pi) - math.pi)
            moved = float(np.hypot(shot["pose"][0] - here[0], shot["pose"][1] - here[1]))
            captions.append(f"画像{i + 1}：いまの向きから見て{where}を向いて撮影" + (f"（{moved:.1f}m 離れた所から）" if moved > .5 else ""))
        prompt = PROMPT.format(floor=self.floor, stairs="上って2" if self.floor == 1 else "下りて1", instruction=instruction,
                               images="\n".join(captions), objects="\n".join(lines) or "（まだない）",
                               frontiers="\n".join(flines) or "（なし）", history="\n".join(self.history) or "（まだない）")
        text, seconds = self.ask(prompt, images)
        try:
            answer = json.loads(text)
        except ValueError:
            answer = {"action": "look_around", "thought": f"（答えを読めなかった：{text[:80]}）"}
        event = {"kind": "decision", "time": round((len(self.trajectory) - 1) * .02, 2), "frame": len(self.trajectory) - 1,
                 "pose": [round(float(self.data.qpos[0]), 2), round(float(self.data.qpos[1]), 2), round(self.heading(), 3)],
                 "images": [p.name for p in images], "objects": lines, "frontiers": flines, "frontierPoints": fronts,
                 "memory": [dict(m) for m in objects], "answer": answer, "seconds": seconds}
        self.events.append(event)
        self.recent_images = self.recent_images[-1:]
        return answer, objects, fronts

    def verify(self, target=None):
        """Before finishing: does the head camera show the destination? When the remembered destination is off to
        one side the head turns to it (at most HEAD_TURN, the body stays) for one more colour image."""
        images = list(reversed(self.recent_images[-2:]))
        glance = None
        if target is not None:
            d = np.asarray(target[:2]) - self.data.qpos[:2]
            off = (math.atan2(d[1], d[0]) + math.pi / 2 - self.heading() + math.pi) % (2 * math.pi) - math.pi
            if abs(off) > math.radians(30):
                glance = self.head_turn(float(np.clip(off, -math.radians(HEAD_TURN), math.radians(HEAD_TURN))))
                images = [glance] + images[:1]
        text, seconds = self.ask(VERIFY.format(instruction=self.instruction), images)
        try:
            answer = json.loads(text)
        except ValueError:
            answer = {"arrived": True, "reason": f"（答えを読めなかった：{text[:60]}）"}
        goals = [m for m in self.known() if m.get("kind") == "target"]
        far = min((self.relative(m["position"])[1] for m in goals), default=0.)
        if answer.get("arrived") and far > 2.5:
            # seen, but the remembered destination is still some way off: not there yet
            answer = {**answer, "arrived": False, "seenFar": True, "reason": f"{answer.get('reason', '')}（ただし行き先はまだ {far:.1f}m 先）"}
        self.events.append({"kind": "check", "time": round((len(self.trajectory) - 1) * .02, 2), "frame": len(self.trajectory) - 1,
                            "images": [p.name for p in images], "answer": answer, "seconds": seconds,
                            **({"headTurn": glance.name} if glance else {})})
        return answer

    def head_turn(self, yaw):
        """A colour image from the head camera turned ``yaw`` (radians, + = left) without turning the body."""
        from types import SimpleNamespace
        from .depth_perception import DepthCameraConfig, _camera_pose
        _, origin, _, _ = _camera_pose(SimpleNamespace(model=self.model, data=self.data), DepthCameraConfig())
        self.renderer.stdin.write(json.dumps({"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2]),
                                              "heading": self.heading() + yaw, "pitch": EYE_PITCH, "width": WIDTH,
                                              "height": HEIGHT, "hfov": HFOV}) + "\n")
        reply = json.loads(self.renderer.stdout.readline())
        path = self.out / f"look-{self.looks:02d}-head{math.degrees(yaw):+.0f}.jpg"
        path.write_bytes(base64.b64decode(reply["jpeg"]))
        return path

    def last_look(self):
        return next(e for e in reversed(self.events) if e["kind"] == "look")

    # ------------------------------------------------------------------ acting
    def go(self, goal, face=None, why=""):
        """Walk to ``goal`` on the depth map in stop-and-look segments; then face ``face`` (a point).
        ``goal`` may be a function: it is asked again after every look, as the map grows."""
        from .tracker.house_tour_depth import FINISH, SEGMENT, truncate
        for _ in range(MAX_SEGMENTS_PER_MOVE):
            goal = np.asarray(goal() if callable(goal) else goal, float)
            here = self.reference[-1, :2]
            route = self.plan(here, goal)
            if route is None:
                return "経路が見つからない"
            total = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
            direction = math.atan2(*(route[1] - route[0])[::-1])
            signed = (direction + math.pi / 2 - self.heading() + math.pi) % (2 * math.pi) - math.pi
            turn = abs(signed)
            tight = turn > math.radians(30) and self.clearance(here) < TIGHT_ROOM
            if total < .25 or (total < CLOSE_ENOUGH and tight and turn > math.radians(60)):
                break                     # there, or a few steps that would need a big turn in a tight spot
            if tight and self.escape():
                continue                          # stepped out of the tight spot first; plan again from there
            arc = tight and turn <= math.radians(70)     # no room to turn on the spot: bend while walking
            go = min(SEGMENT, self.map.seen_until(route) - .4)
            door = self.narrow_ahead(route)
            if door is not None and door > .8:
                # stop in front of a door, face it there and walk straight through (walking round the corner into it
                # cuts across the frame)
                go = min(go, door - .45)
            if total <= max(go, FINISH) + 1e-6:
                self.walk_route(route, arc=arc)
                self.look(f"{why}：到着")
                break
            if go < .3:
                if self.clearance(here) >= TIGHT_ROOM and self.turn_to(direction + math.pi / 2):
                    self.look(f"{why}：先を見る")
                    continue
                go = .5
            self.walk_route(truncate(route, self.roomy_stop(route, go)), arc=arc)
            self.look(f"{why}：途中で止まって見る")
        if face is not None:
            d = np.asarray(face[:2]) - self.data.qpos[:2]
            left = (math.atan2(d[1], d[0]) + math.pi / 2 - self.heading() + math.pi) % (2 * math.pi) - math.pi
            # turning round (over 120 degrees) drifted 0.4 m and fell in the 1.1 m hall: only where there is room
            room = TURN_ROOM if abs(left) > math.radians(120) else TIGHT_ROOM
            if abs(left) > math.radians(40) and self.clearance(self.data.qpos[:2]) >= room and \
                    self.turn_to(math.atan2(d[1], d[0]) + math.pi / 2):
                self.look(f"{why}：向きを合わせる")
        return "到着"

    def use_stairs(self, obj):
        """Up or down the staircase. Seeing the stairs and choosing to take them are the VLM's; the flight itself is
        a fixture the body is set up for (where it starts and its step sizes, as a robot is set up for the stairs of
        its home), so remembered stairs that are not over it are not the stairs. The body walks to the foot (or the
        top), faces along the flight and walks a stair reference (``house_stairs.stair_walk``) made from a walk the
        tracker follows on these stairs and that keeps clear of the walls; then the other floor is its own map."""
        from .house2f import STAIR_X0, STAIR_X1, STAIR_Y
        from .tracker.house_stairs import BOTTOM, CENTRE_Y, TOP, _touches, stair_walk
        from .tracker.path_reference import root_yaw
        p = np.asarray(obj["position"][:2])
        if not (STAIR_X0 - 1. <= p[0] <= STAIR_X1 + 1. and STAIR_Y[0] - .8 <= p[1] <= STAIR_Y[1] + .8):
            return "階段ではなかった"
        up = self.floor == 1
        start = np.asarray([BOTTOM if up else TOP, CENTRE_Y])
        status = self.go(start, why="階段の" + ("下" if up else "上") + "へ")
        if status != "到着" or np.linalg.norm(self.data.qpos[:2] - start) > .35:
            return "階段の" + ("下" if up else "上") + "に着けなかった"
        self.turn_to((0. if up else math.pi) + math.pi / 2)      # along the flight (+x is up)
        here = self.reference[-1]
        check = self.mujoco.MjData(self.model)
        blocked = {g for g in self.names if g not in self.treads}
        end, ahead = (TOP, 0.) if up else (BOTTOM, math.pi)
        best = None
        for name in json.loads(STAIR_WALKS.read_text())["up" if up else "down"]:
            try:
                flight = stair_walk(self.plain, self.bank[name], up, float(here[0]), end,
                                    float(np.clip(here[1] - CENTRE_Y, -.08, .08)))[0]["qpos"].astype(np.float64)
            except ValueError:        # its steps do not end on the floor beyond the flight
                continue
            if _touches(self.model, check, flight, blocked, self.body_geoms):
                continue
            # ending in the middle of the small landing (not at the edge of the flight) and facing along it
            turned = abs((root_yaw(flight[-1:])[0] - ahead - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
            cost = abs(flight[-1, 0] - end) + .5 * turned
            if best is None or cost < best[0]:
                best = (cost, name, flight)
        if best is None:
            return "階段を歩く動きが作れなかった"
        _, name, flight = best
        self.doing = f"階段を{'上る' if up else '下りる'}（{name}）"
        self.on_stairs = True
        self.append(flight, placed=True)
        self.hold(25)
        self.simulate()
        self.on_stairs = False
        self.change_floor(2 if up else 1)
        return "到着"

    def change_floor(self, floor):
        """Now on ``floor``: its own depth map and the places seen there (the floor left is kept for coming back)."""
        from .tracker.house_tour_depth import Map
        self.floors[self.floor] = (self.map, self.hits, self.discs, self.walked, self.frontier_ids, self.dead, self.rejected)
        if floor in self.floors:
            self.map, self.hits, self.discs, self.walked, self.frontier_ids, self.dead, self.rejected = self.floors[floor]
        else:
            from .house2f import STAIR_X0, STAIR_X1, STAIR_Y
            self.map = Map((-5.2, -8.0), (5.2, 5.2))
            self.map.floor_z = FLOORS[floor]
            self.map.frame_index = lambda: len(self.trajectory) - 1
            # the flight just walked (the well on the floor above) is known before it is seen behind the body
            xs, ys = np.meshgrid(np.arange(STAIR_X0, STAIR_X1, self.map.cell), np.arange(STAIR_Y[0], STAIR_Y[1], self.map.cell))
            self.map.grid.mark(np.stack([xs.ravel(), ys.ravel()], 1))
            self.hits = self.map.grid.occupied.copy()
            self.discs, self.walked = np.zeros_like(self.hits), np.zeros_like(self.hits)
            self.frontier_ids, self.dead, self.rejected = [], [], []
        self.floor = floor
        self.recent_images = []

    def narrow_ahead(self, route):
        """Distance along ``route`` to the next narrow passage (a door: less than NARROW room) after some wider
        floor (the body may be standing in one now); None when there is none."""
        from .tracker.house_tour_depth import truncate
        total = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
        wide = False
        for d in np.arange(.1, total, .05):
            room = self.clearance(truncate(route, d)[-1])
            if room >= NARROW + .05:
                wide = True
            elif wide and room < NARROW:
                return float(d)
        return None

    def roomy_stop(self, route, go):
        """How far along ``route`` to stop (at most ``go``): the farthest spot within the last metre before it where
        the body has room to turn round, else ``go``."""
        from .tracker.house_tour_depth import truncate
        for d in np.arange(go, max(go - 1., .3) - 1e-6, -.1):
            if self.clearance(truncate(route, d)[-1]) >= TURN_ROOM:
                return float(d)
        return go

    def escape(self):
        """Out of a tight spot without turning on the spot: walk (bending at most 55 degrees off the way the body
        faces) to the nearest spot with room to turn round. False when there is none."""
        here = self.data.qpos[:2].copy()
        ahead = self.heading() - math.pi / 2
        best, now = None, self.clearance(here)
        for dist in (.5, .7, .9, 1.1):
            for off in (0, 20, -20, 40, -40, 55, -55):
                a = ahead + math.radians(off)
                q = here + dist * np.asarray([math.cos(a), math.sin(a)])
                room = self.clearance(q)
                if room < now + .1 or min(self.clearance(here + f * (q - here)) for f in np.linspace(.25, 1., 5)) < HARD:
                    continue
                if best is None or room > best[0] + .05:
                    best = (room, q)
            if best is not None and best[0] >= .6:
                break
        if best is None:
            return False
        self.walk_route(np.stack([here, best[1]]), arc=True)
        self.look("狭い所から出る")
        return True

    def plan(self, start, goal):
        """Cheapest 8-connected route on the depth map where a cell closer than SOFT to an obstacle costs more
        (so routes keep to the middle of aisles and doors) and closer than HARD is closed; the body may leave
        a tighter spot it stands in and the goal may be in one. Shortcut by line of sight where that does not
        bring the route closer to an obstacle."""
        from scipy.ndimage import distance_transform_edt
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        g = self.map.grid
        s, t = g.index(start), g.index(goal)
        pad = int(3. / g.cell)
        lo = np.maximum(np.minimum(s, t) - pad, 0)
        hi = np.minimum(np.maximum(s, t) + pad + 1, g.occupied.shape)
        room = distance_transform_edt(~g.occupied[lo[0]:hi[0], lo[1]:hi[1]]) * g.cell
        s, t = s - lo, t - lo
        ii, jj = np.indices(room.shape)
        ends = np.minimum(np.hypot(ii - s[0], jj - s[1]), np.hypot(ii - t[0], jj - t[1])) * g.cell
        open_ = (room >= HARD) | ((ends <= HARD) & (room > g.cell))
        cost = 1. + 8. * np.clip((SOFT - room) / (SOFT - HARD), 0., 1.) ** 2
        n, w = room.size, room.shape[1]
        index = np.arange(n).reshape(room.shape)
        rows, cols, weights = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = (slice(max(-di, 0), room.shape[0] - max(di, 0)), slice(max(-dj, 0), room.shape[1] - max(dj, 0)))
            b = (slice(a[0].start + di, a[0].stop + di), slice(a[1].start + dj, a[1].stop + dj))
            ok = open_[a] & open_[b]
            rows.append(index[a][ok])
            cols.append(index[b][ok])
            weights.append((math.hypot(di, dj) * .5 * (cost[a] + cost[b]))[ok])
        rows, cols, weights = map(np.concatenate, (rows, cols, weights))
        graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        source, sink = int(index[tuple(s)]), int(index[tuple(t)])
        dist, prev = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
        if not np.isfinite(dist[sink]):
            return None
        cells = [sink]
        while cells[-1] != source:
            cells.append(int(prev[cells[-1]]))
        cells = [np.asarray(divmod(c, w)) for c in reversed(cells)]
        # in narrow passages (doors) the centred route itself is kept, a point every 20 cm, so the body goes
        # straight through the middle instead of cutting across the frame
        narrow = [k for k, c in enumerate(cells) if room[tuple(c)] < NARROW]
        anchors = sorted(set(narrow[::4] + narrow[-1:]))
        keep, i = [cells[0]], 0
        while i < len(cells) - 1:
            j = next((a for a in anchors if a > i), len(cells) - 1)
            while j > i + 1:
                floor_room = min(room[tuple(c)] for c in cells[i:j + 1])
                line = np.rint(np.linspace(cells[i], cells[j], int(np.abs(cells[j] - cells[i]).max()) + 1)).astype(int)
                if room[line[:, 0], line[:, 1]].min() >= min(floor_room, SOFT) - 1e-9:
                    break
                j -= 1
            keep.append(cells[j])
            i = j
        return np.asarray([np.asarray(start, float)] + [g.centre(c + lo) for c in keep[1:-1]] + [np.asarray(goal, float)])

    def safe_point(self, point, clearance=.6, reach=1.5):
        """The nearest spot to ``point`` at least ``clearance`` from every known obstacle (less where nothing has
        that much: a 1.1 m wide toilet)."""
        point = np.asarray(point[:2], float)
        for need in sorted({clearance, min(clearance, .45), min(clearance, .35)}, reverse=True):
            if self.clearance(point) >= need:
                return point
            for radius in np.arange(.1, reach + 1e-6, .1):
                for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                    q = point + radius * np.asarray([math.cos(a), math.sin(a)])
                    if self.clearance(q) >= need:
                        return q
        return point

    def in_sight(self, a, b, skip=0.):
        """Seen free floor all the way from ``a`` to ``b`` (the last ``skip`` m before ``b`` is the object itself):
        a spot to stand at is on the near side of the walls, not behind one (a wall not seen yet counts too)."""
        g = self.map.grid
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)
        length = float(np.linalg.norm(b - a))
        for f in np.arange(0., max(length - skip, 0.), g.cell) / max(length, 1e-9):
            i, j = g.index(a + (b - a) * f)
            if not (0 <= i < g.occupied.shape[0] and 0 <= j < g.occupied.shape[1]) or g.occupied[i, j] or not self.map.seen[i, j]:
                return False
        return True

    def place_point(self, place, near=.8):
        """Where to stand for a place on the floor: the nearest spot within ``near`` of it with room to turn round,
        else the nearest with a little room (inside a small room such as the toilet)."""
        place = np.asarray(place[:2], float)
        for need, reach in ((TURN_ROOM, near), (.45, 1.5)):
            if self.clearance(place) >= need:
                return place
            for radius in np.arange(.1, reach + 1e-6, .1):
                for a in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                    q = place + radius * np.asarray([math.cos(a), math.sin(a)])
                    if self.clearance(q) >= need and self.in_sight(q, place):
                        return q
        return place

    def stand_point(self, target):
        """A free spot STAND_OFF from ``target``, on the side facing the body; where the body stands already when it
        is about that near and sees it (in the 1 m kitchen aisle in front of the fridge no spot 1.2 m off has room)."""
        target = np.asarray(target[:2], float)
        here = self.data.qpos[:2].copy()
        if TOUCH <= np.linalg.norm(here - target) <= STAND_OFF + .1 and self.in_sight(here, target, .5):
            return here
        # room to turn round there (the arms reach about 0.5 m while turning on the spot); a spot next to
        # space not seen yet does not count as roomy
        for clearance, radii in ((.6, (STAND_OFF, 1.5, 1.8, 2.2)), (TURN_ROOM, (STAND_OFF, 1.5, 1.8, 2.2)),
                                 (.45, (STAND_OFF, 1.5, 1.8, 2.2)), (.4, (.9, .75))):   # last: nearer in an aisle
            best = None
            for radius in radii:
                for a in np.linspace(-math.pi, math.pi, 48, endpoint=False):
                    p = target + radius * np.asarray([math.cos(a), math.sin(a)])
                    if self.clearance(p, unseen_blocks=True) < clearance or not self.in_sight(p, target, .5):
                        continue
                    cost = np.linalg.norm(p - here)
                    if best is None or cost < best[0]:
                        best = (cost, p)
                if best is not None:
                    return best[1]
        return self.safe_point(target, .4)

    def episode(self, task):
        self.instruction = task["text"]
        self.look("開始")
        self.history = []
        if self.clearance(self.data.qpos[:2]) >= TIGHT_ROOM:   # first of all, look round where it starts (turning right)
            for k in range(3):
                self.turn_to(self.heading() - math.pi / 2)
                self.look(f"最初に見回す（{(k + 1) * 90}°）")
            self.history.append("0. 最初にその場で見回した")
        result = {"finished": False}
        try:
            for step in range(MAX_DECISIONS):
                answer, objects, fronts = self.decide(task["text"])
                action, target = str(answer.get("action", "")), str(answer.get("target", "")).strip()
                untried = [m for m in objects if m.get("kind") == "target" and m["id"] not in self.tried]
                if action in ("explore", "look_around") and untried:
                    # a destination is in sight: it is checked before anything else is explored
                    goal = min(untried, key=lambda m: self.relative(m["position"])[1])
                    self.events[-1]["override"] = f"【行き先】{goal['name']}（{goal['id']}）が見つかっているので、先にそこへ向かう"
                    action, target = "go_to_object", goal["id"]
                if action == "finish":
                    success = bool(answer.get("success"))
                    goals = [m for m in self.known() if m.get("kind") == "target"]
                    far = min((self.relative(m["position"])[1] for m in goals), default=0.)
                    if not success and fronts and self.refused["give_up"] < 2:
                        # not yet: unexplored places are left (asked at most twice)
                        self.refused["give_up"] += 1
                        self.history.append(f"{step + 1}. 終了しようとしたが、まだ見ていない所（{'・'.join(f['id'] for f in fronts)}）が"
                                            "残っているので探し続ける")
                        continue
                    if success and goals and far > 2.5 and self.refused["far"] < 1:
                        self.refused["far"] += 1
                        self.history.append(f"{step + 1}. 終了しようとしたが、行き先はまだ {far:.1f}m 先なので近くまで行く")
                        continue
                    if success and self.refused["check"] < 2:
                        check = self.verify(min(goals, key=lambda m: self.relative(m["position"])[1])["position"] if goals else None)
                        if not check.get("arrived"):
                            # the frame it went to was not the destination: forget it and go on
                            self.refused["check"] += 1
                            near = [m for m in goals if self.relative(m["position"])[1] < 1.5]
                            for m in near:
                                self.memory.remove(m)
                                self.rejected.append(np.asarray(m["position"][:2]))
                            self.history.append(f"{step + 1}. 終了しようとしたが、画像で確かめると行き先ではなかった"
                                                f"（{check.get('reason', '')}）" + (f"。{'・'.join(m['id'] for m in near)} は記憶から消した" if near else ""))
                            continue
                    result = {"finished": True, "claimed": success, "message": answer.get("message", "")}
                    self.history.append(f"{step + 1}. 終了：{answer.get('message', '')}")
                    break
                if action == "go_to_object":
                    obj = next((m for m in objects if m["id"] == target), None)
                    if obj is None:
                        self.history.append(f"{step + 1}. {target} は見つかっていない物だった")
                        continue
                    last = [e for e in self.events if e["kind"] == "decision"][-2:]
                    same = len(last) == 2 and last[0]["answer"].get("target") == target and last[0]["pose"][:2] == last[1]["pose"][:2]
                    if obj["id"] in self.unreachable or same:
                        # asked again for a place it found no route to: look further instead
                        front = min(fronts, key=lambda f: self.relative(f["position"])[1]) if fronts else None
                        self.events[-1]["override"] = f"{obj['id']} へは{'もう来ている' if same else '経路がなかった'}ので、" + \
                            (f"近くのまだ見ていない所（{front['id']}）へ行く" if front else "その場で見回す")
                        if front is not None:
                            self.dead.append(np.asarray(front["position"]))
                            status = self.go(lambda: self.safe_point(front["position"]), why=f"まだ見ていない所（{front['id']}）へ")
                            self.history.append(f"{step + 1}. {obj['id']} へは行けないので、まだ見ていない所 {front['id']} へ移動：{status}")
                        else:
                            self.look_around(step)
                        continue
                    name = f"{obj['name']}（{obj['id']}）"
                    self.tried.add(obj["id"])
                    if obj.get("kind") == "stairs":        # up or down the flight, onto the other floor
                        status = self.use_stairs(obj)
                        self.history.append(f"{step + 1}. {name}で別の階へ：{status}" + (f"（いまは{self.floor}階）" if status == "到着" else ""))
                        if status == "到着":
                            self.look(f"{self.floor}階に着いた")
                        continue
                    if obj.get("kind") == "door":          # through the way out: to the floor beyond it
                        status = self.go(lambda: self.safe_point(obj["position"], .45), why=f"出入口 {name} の先へ")
                    elif obj.get("floor"):                 # a place on the floor: onto it, or next to it where there is room
                        status = self.go(lambda: self.place_point(obj["position"]), face=obj["position"], why=f"{name}へ")
                    else:
                        status = self.go(lambda: self.stand_point(obj["position"]), face=obj["position"], why=f"{name}へ")
                    if status == "経路が見つからない" and obj.get("kind") in ("target", "clue") and \
                            np.linalg.norm(np.asarray(obj["position"][:2]) - self.data.qpos[:2]) <= 1.5 and \
                            self.in_sight(self.data.qpos[:2], obj["position"], .5):
                        # no room to step nearer, but it is right there (inside the 1F toilet): checked from here
                        status = "到着（すぐ近くにいて、それ以上は近づけない）"
                    self.history.append(f"{step + 1}. {name}の{'先' if obj.get('kind') == 'door' else '前'}へ移動：{status}")
                    if status == "経路が見つからない":
                        self.unreachable.add(obj["id"])
                    if obj.get("kind") in ("target", "clue") and status.startswith("到着") and self.refused["check"] < 4:
                        # at the destination (or a clue to it): the executor looks whether this is it, and ends there
                        # when it is; a wrong one is forgotten
                        check = self.verify(obj["position"])
                        if check.get("arrived"):
                            result = {"finished": True, "claimed": True, "message": f"{obj['name']}に着いたことを画像で確かめた",
                                      "checkedOnArrival": True}
                            self.history.append(f"{step + 1}. 着いて画像で確かめた：{check.get('reason', '')}")
                            break
                        if check.get("seenFar"):
                            self.history.append(f"{step + 1}. ここから行き先が見えるが、まだ先：{check.get('reason', '')}")
                            continue
                        self.refused["check"] += 1
                        if obj in self.memory:
                            self.memory.remove(obj)
                        self.rejected.append(np.asarray(obj["position"][:2]))
                        self.history.append(f"{step + 1}. 着いて画像で確かめると行き先ではなかった（{check.get('reason', '')}）。"
                                            f"{obj['id']} は記憶から消した")
                        self.unreachable.add(obj["id"])
                elif action == "explore":
                    front = next((f for f in fronts if f["id"] == target), None)
                    if front is None:
                        self.history.append(f"{step + 1}. {target} は候補になかった")
                        continue
                    self.dead.append(np.asarray(front["position"]))
                    status = self.go(lambda: self.safe_point(front["position"]), why=f"まだ見ていない所（{front['id']}）へ")
                    self.history.append(f"{step + 1}. まだ見ていない所 {front['id']} へ移動：{status}")
                else:
                    self.look_around(step)
        except Stop as stop:
            result = {"finished": False, "stoppedBy": str(stop)}
        result.update(self.score(task))
        self.events.append({"kind": "end", "time": round((len(self.trajectory) - 1) * .02, 2), "frame": len(self.trajectory) - 1,
                            "result": result})
        return result

    def look_around(self, step):
        if self.clearance(self.data.qpos[:2]) < TIGHT_ROOM:
            self.history.append(f"{step + 1}. 周りが狭くて、その場では回れなかった")
            return
        for k in range(3):
            self.turn_to(self.heading() - math.pi / 2)
            self.look(f"見回す（{(k + 1) * 90}°）")
        self.history.append(f"{step + 1}. その場で見回した")

    def score(self, task):
        x, y = map(float, self.data.qpos[:2])
        level = 2 if self.data.qpos[2] > FLOORS[2] else 1      # the floor the body ended on (the pelvis is above it)
        fell = self.state["fell"] is not None
        if task["kind"] == "zone":
            x0, y0, x1, y1 = self.scene["metadata"]["ldkZones"][task["target"]]
            reached = level == 1 and x0 <= x <= x1 and y0 <= y <= y1
        elif task["kind"] == "room":
            targets = task["target"] if isinstance(task["target"], list) else [task["target"]]
            rooms = [r for r in self.scene["rooms"] if r["id"] in targets and r.get("floor", 1) == level]
            reached = any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for room in rooms for r in room["rects"])
            # a secondary record (success stays "inside"): within 0.5 m of the room, i.e. at its door
            at_door = any(r[0] - .5 <= x <= r[2] + .5 and r[1] - .5 <= y <= r[3] + .5 for room in rooms for r in room["rects"])
        elif task["kind"] == "object":
            near = min((np.hypot(o["position"][0] - x, o["position"][1] - y) for o in self.scene["objects"]
                        if (o["class"] == task["target"] or o.get("coco") == task["target"]) and o.get("floor", 1) == level), default=99.)
            reached = bool(near <= 1.4)
        elif task["kind"] == "free":   # an instruction from the conversation: no ground truth, the VLM's own check
            reached = any(e["kind"] == "check" and e["answer"].get("arrived") for e in self.events[-3:]) or \
                any(e["kind"] == "decision" and e["answer"].get("action") == "finish" and not e["answer"].get("success")
                    for e in self.events[-2:])
        else:   # the agent should say it cannot do it
            last = next((e for e in reversed(self.events) if e["kind"] == "decision"), None)
            reached = bool(last and last["answer"].get("action") == "finish" and not last["answer"].get("success"))
        return {"success": bool(reached and not fell and not self.state["contacts"]), "reached": bool(reached),
                **({"atDoor": bool(at_door)} if task["kind"] == "room" else {}),
                "position": [round(x, 2), round(y, 2)], "floor": level, "fell": fell, "contactCount": len(self.state["contacts"]),
                "contacts": self.state["contacts"][:5], "decisions": sum(e["kind"] == "decision" for e in self.events),
                "looks": self.looks, "seconds": round((len(self.trajectory) - 1) * .02, 1)}

    # ------------------------------------------------------------------ records for the web page
    def draw_look(self, event, depth, obstacle):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image, ImageDraw, ImageFont
        k = event["index"]
        image = Image.open(self.out / event["rgb"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 15)
        for d in event["detections"]:
            box = d["box"]
            draw.rectangle(box, outline=(255, 200, 0), width=3)
            text = f"{JA.get(d['coco'], d['coco'])} {d['conf']:.2f}"
            w = draw.textlength(text, font=font)
            draw.rectangle([box[0], box[1] - 20, box[0] + w + 8, box[1]], fill=(255, 200, 0))
            draw.text((box[0] + 4, box[1] - 20), text, fill=(20, 20, 20), font=font)
        for d in event.get("vlm", {}).get("items", []):
            box = d["box"]
            draw.rectangle(box, outline=(40, 210, 120), width=3)
            text = f"VLM {KIND[d['kind']]}：{d['what']}"
            w = draw.textlength(text, font=font)
            draw.rectangle([box[0], box[3], box[0] + w + 8, box[3] + 20], fill=(40, 210, 120))
            draw.text((box[0] + 4, box[3]), text, fill=(10, 30, 20), font=font)
        image.save(self.out / f"look-{k:02d}-boxes.jpg", quality=88)
        shown = np.fliplr(np.where(depth > 0, depth.astype(float), np.nan))
        fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
        ax.imshow(shown, cmap="magma_r", vmin=0, vmax=8, interpolation="nearest")
        over = np.zeros(depth.shape + (4,))
        over[np.fliplr(obstacle)] = [0.2, 0.9, 1.0, 0.5]
        ax.imshow(over, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(self.out / f"depth-{k:02d}.png", facecolor="#0b1220")
        plt.close(fig)
        self.draw_map(self.out / f"map-{k:02d}.png", event)
        event.update(boxes=f"look-{k:02d}-boxes.jpg", depth=f"depth-{k:02d}.png", map=f"map-{k:02d}.png")

    def draw_map(self, path, event):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrow
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
        g = self.map.grid
        nx, ny = g.occupied.shape
        ext = [g.low[0], g.low[0] + nx * g.cell, g.low[1], g.low[1] + ny * g.cell]
        img = np.ones((nx, ny, 4))
        img[self.map.seen] = [0.86, 0.90, 0.94, 1]
        img[g.occupied] = [0.85, 0.28, 0.23, 1]
        fig, ax = plt.subplots(figsize=(4.8, 5.6), dpi=100)
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext)
        q = np.asarray(self.trajectory)
        ax.plot(q[:, 0], q[:, 1], color="#E69138", lw=1.4)
        for m in self.known():
            if m["conf"] < .4 and m["seen"] < 2:
                continue
            ax.plot(*m["position"][:2], "s" if m["coco"] == "vlm" else "o", ms=6,
                    color="#1E9E5A" if m["coco"] == "vlm" else "#2F6FA8")
            ax.annotate(f"{m['id']} {m['name']}", m["position"][:2], xytext=(3, 3), textcoords="offset points", fontsize=7.5, color="#1F2933")
        for f in self.frontiers():
            ax.plot(*f["position"], "D", ms=6, color="#6AA84F")
            ax.annotate(f["id"], f["position"], xytext=(3, -9), textcoords="offset points", fontsize=7.5, color="#3d6b2a")
        x, y, h = event["pose"]
        ax.add_patch(FancyArrow(x, y, .6 * math.sin(h), -.6 * math.cos(h), width=.08, head_width=.3, head_length=.25, color="#1F2933"))
        ax.set_xlim(-5.4, 5.4)
        ax.set_ylim(-8.2, 5.4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{event['time']:.1f}秒の{self.floor}階の地図（赤＝障害物・灰＝見た床・◆＝未確認の境目）", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def save(self, task, result):
        from .tracker.publish_viewer import AVATAR_BODY_MAP
        mujoco, B = self.mujoco, self.B
        q = np.asarray(self.trajectory)
        data = mujoco.MjData(self.model)
        frames = []
        for i in list(range(0, len(q), 4)) + ([len(q) - 1] if (len(q) - 1) % 4 else []):
            data.qpos[:] = q[i]
            mujoco.mj_kinematics(self.model, data)
            frames.append({"time": round(i * .02, 2), "bodies": [[round(float(v), 4) for v in (*data.xpos[b], *data.xquat[b])]
                                                                 for b in range(1, self.nb + 1)]})
        plain = B.build_model()
        rest = mujoco.MjData(plain)
        mujoco.mj_kinematics(plain, rest)
        trace = {"version": 1, "bodyNames": [plain.body(b).name for b in range(1, self.nb + 1)], "frames": frames,
                 "avatarProfile": json.loads(B.PROFILE_PATH.read_text()), "avatarBodyMap": AVATAR_BODY_MAP,
                 "physicsRestBodies": [{"name": plain.body(b).name, "position": [round(float(v), 4) for v in rest.xpos[b]],
                                        "quaternion": [round(float(v), 4) for v in rest.xquat[b]]} for b in range(1, self.nb + 1)]}
        (self.out / "trace.json").write_text(json.dumps(trace, separators=(",", ":")))
        n = len(q)
        np.savez_compressed(self.out / "roots.npz", reference=self.reference[:n, :3], actual=q[:, :3])   # for checking the tracking
        g = self.map.grid
        np.savez_compressed(self.out / "map.npz", occupied=g.occupied, seen=self.map.seen, discs=self.discs, low=g.low, cell=g.cell)
        run = {"version": 1, "task": {k: task[k] for k in ("id", "text")}, "model": {"tracker": self.args.model, "vlm": self.args.vlm_model,
               "detector": YOLO_WEIGHTS.name}, "result": result, "events": self.events,
               "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="v14", help="tracker run")
    parser.add_argument("--vlm", default="http://127.0.0.1:11999")
    parser.add_argument("--vlm-model", default="qwen3-vl:30b-a3b-instruct")
    parser.add_argument("--base", default="http://127.0.0.1:4010", help="the house page used to render the colour images")
    parser.add_argument("--only", default="", help="comma-separated task ids")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    tasks = [t for t in TASKS if not args.only or t["id"] in args.only.split(",")]
    summary = []
    for task in tasks:
        folder = out / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*"):
            old.unlink()
        agent = Agent(args, folder)
        started = time.time()
        try:
            result = agent.episode(task)
            agent.save(task, result)
        finally:
            agent.close()
        summary.append({"id": task["id"], "text": task["text"], **{k: result.get(k) for k in ("success", "finished", "claimed", "message",
                        "stoppedBy", "decisions", "looks", "seconds")}, "wall": round(time.time() - started, 1)})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    index = {"version": 1, "runs": summary, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    publish(out)


def publish(out):
    """Copy the records to the monitor folder (/training-monitor/agent/), read by /world-model/house."""
    import shutil
    target = PUBLIC / "agent"
    index = json.loads((out / "index.json").read_text())
    known = {r["id"]: r for r in (json.loads((target / "index.json").read_text())["runs"] if (target / "index.json").exists() else [])}
    for run in index["runs"]:
        src, dst = out / run["id"], target / run["id"]
        if not (src / "run.json").exists():
            continue
        tmp = target / f".{run['id']}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in src.iterdir():
            if f.name in ("run.json", "trace.json") or f.name.endswith(("-boxes.jpg", ".png")) or "-head" in f.name:
                shutil.copy2(f, tmp / f.name)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        known[run["id"]] = run
    order = [t["id"] for t in TASKS]
    runs = sorted(known.values(), key=lambda r: order.index(r["id"]) if r["id"] in order else 99)
    (target / "index.json.tmp").write_text(json.dumps({"version": 1, "runs": runs, "updatedAt": index["updatedAt"]},
                                                      ensure_ascii=False, indent=1))
    (target / "index.json.tmp").replace(target / "index.json")


if __name__ == "__main__":
    main()
