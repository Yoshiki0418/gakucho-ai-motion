"""学習中の重みで、学長の身体を 9 体リアルタイムに動かして配信する(画面 /world-model/live)。

GPU の学習とは別に、CPU の MuJoCo で最新のチェックポイントの方策を実時間(50Hz)で動かす。
学習中の世界(GPU 上で 8192 体を同時に、1 体あたり実時間の 1/6 程度の速さで動かし、探索の揺らぎも入る)
をそのまま映すとスローモーションでガタつくため、「いまの重みでどこまでできるか」を実時間で見せる。
新しいチェックポイントが出ると数分おきに重みを差し替える。

    GET /stream   Server-Sent Events。最初に layout(身体の形)、以降 25Hz で frame(各体の姿勢と状態)
    GET /status   いまの重みと各体の状態(JSON)

学習に使っていない保留モーキャプだけを再生する。追従が崩れたら(体の平均ずれ 0.5m 超)その場で止め、
1.5 秒後に次の動作へ進む。GPU は使わない。
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / ".runtime/world-model/tracker/runs"
HZ = 50
SEND_EVERY = 2          # 25Hz で配信
PAUSE_AFTER_END = 0.8   # 動作の終わりで止まる時間(秒)
PAUSE_AFTER_FAIL = 1.5  # 崩れたときに止める時間(秒)
MAX_CLIP_SECONDS = 9.0
CLIPS_PER_SLOT = 3

# 9 体それぞれの動作(本文の語で保留モーキャプから選ぶ)。椅子は座る学習のときだけ。
SLOTS = [  # 3 列 × 3 行。手前の行から。移動する動作は奥の行に置く
    ("手を振る", "立っている状態から、手を振る", False),
    ("お辞儀", "立っている状態から、お辞儀", False),
    ("しゃがむ", "立っている状態から、しゃが", False),
    ("椅子に座る", "立っている状態から、椅子に座る", True),
    ("椅子から立つ", "椅子に座っている状態から、立ち上が", True),
    ("ジャンプ", "立っている状態から、ジャンプ", False),
    ("歩く", "立っている状態から、歩く", False),
    ("走る", "立っている状態から、走る", False),
    ("見回す", "立っている状態から、見回す", False),
]
STANDING_FALLBACK = [("敬礼", "敬礼", False), ("スキップ", "スキップ", False)]


def _latest_run() -> tuple[Path, Path] | None:
    """いちばん新しい学習と、その最新の重み。"""
    runs = sorted((p for p in RUNS.iterdir() if p.is_dir() and (p / "run.json").exists() and "sanity" not in p.name),
                  key=lambda p: (p / "run.json").stat().st_mtime, reverse=True)
    for run in runs:
        final = run / "final_params.pkl"
        checkpoints = sorted((run / "checkpoints").glob("params_*.pkl"))
        params = final if final.exists() else (checkpoints[-1] if checkpoints else None)
        if params is not None:
            return run, params
    return None


def _step_of(run: Path, params: Path) -> int:
    from .monitor import RESUMED_FROM
    offset = RESUMED_FROM.get(run.name, 0)
    if params.name.startswith("params_"):
        return int(params.stem.split("_")[1]) + offset
    rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line.strip()]
    return int(rows[-1]["step"]) + offset if rows else offset


class Brain:
    """ある学習の、ある重みで方策と観測を用意する(読み込みは別スレッドで行い、できたら差し替える)。"""

    def __init__(self, run: Path, params: Path) -> None:
        import jax
        import jax.numpy as jp

        from .evaluate_tracker import _policy
        from .mjx_tracking import TrackingEnv, default_config, load_clips

        self.run, self.params = run, params
        meta, self.policy = _policy(str(run), str(params))
        self.step = _step_of(run, params)
        self.seated = bool(meta["args"].get("include_seated"))
        library = meta["args"]["library"]
        library = str(ROOT / library) if not os.path.isabs(library) else library
        _, every, ref = load_clips(library, split="test", include_seated=self.seated, keys=("qpos", "qvel", "xpos"))
        playlist = []
        slots = [s for s in SLOTS if self.seated or not s[2]]
        slots += STANDING_FALLBACK[:len(SLOTS) - len(slots)]
        for title, keyword, chair in slots:
            found = [c for c in every if keyword in c["text"] and c["dataset"] == "0001"
                     and (c["needsSeat"] == chair or not chair and not c["needsSeat"])]
            playlist.append((title, [c["path"] for c in found[:CLIPS_PER_SLOT]]))
        self.playlist = [(title, paths) for title, paths in playlist if paths]

        config = default_config()
        for name, value in meta["config"].items():
            if name in config and not isinstance(value, dict):
                config[name] = tuple(value) if isinstance(value, list) else value
        config.impl, config.split, config.include_seated = "jax", "test", self.seated
        config.clip_weights = ""   # training-time clip sampling only
        wanted = sorted({p for _, paths in self.playlist for p in paths})
        self.env = TrackingEnv(library, config, clips=wanted)
        self.config = config
        self.model = self.env.mj_model
        index = {c["path"]: i for i, c in enumerate(self.env.clips)}
        # 参照の動作は経路で引けるように持ち直す
        self.ref = {}
        for clip in every:
            if clip["path"] in index:
                s, n = clip["start"], clip["length"]
                self.ref[clip["path"]] = {"qpos": ref["qpos"][s:s + n], "qvel": ref["qvel"][s:s + n],
                                          "xpos": ref["xpos"][s:s + n]}
        self.index = index
        env = self.env
        tables = (env._qpos, env._starts, env._lengths)

        def observe(view, info, table, starts, lengths):
            saved = env._qpos, env._starts, env._lengths
            env._qpos, env._starts, env._lengths = table, starts, lengths
            try:
                return env._observation(SimpleNamespace(**view), info)
            finally:
                env._qpos, env._starts, env._lengths = saved

        batched = jax.jit(jax.vmap(observe, in_axes=(0, 0, None, None, None)))
        self.observe = lambda views, infos: batched(views, infos, *tables)
        self.jp = jp
        self.key = jax.random.PRNGKey(0)
        # 先に一度通してコンパイルを済ませる(差し替えの瞬間に止まらないように)
        self.warm()

    def warm(self) -> None:
        import mujoco
        from . import body as B
        n = len(self.playlist)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        views = {k: np.repeat(np.asarray(getattr(data, k))[None], n, 0) for k in ("xpos", "xquat", "cvel", "qpos", "qvel")}
        infos = {"clip": np.zeros(n, np.int32), "frame": np.zeros(n, np.int32), "last_act": np.zeros((n, 63), np.float32),
                 "linvel": np.zeros((n, len(B.SEGMENTS), 3), np.float32)}
        obs = self.observe(views, infos)
        self.policy(obs, self.key)


class Slot:
    def __init__(self, title: str, paths: list[str]) -> None:
        self.title, self.paths, self.turn = title, paths, 0
        self.data = None
        self.path = paths[0]
        self.frame = 0
        self.state = "pause"
        self.wait = 0.0
        self.err = 0.0
        self.last_act = np.zeros(63, np.float32)
        self.linvel = None
        self.seat = None
        self.origin = [0.0, 0.0]
        self.failed_at = None


class Live:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.brain: Brain | None = None
        self.slots: list[Slot] = []
        self.frame_json = None
        self.layout_json = None
        self.status = {"state": "loading"}
        self.tick = 0

    # ---------------------------------------------------------------- 重みの差し替え

    def watch_weights(self) -> None:
        current = None
        while True:
            try:
                found = _latest_run()
                if found and (current is None or found != current):
                    print(f"[live] loading {found[0].name} {found[1].name}", flush=True)
                    brain = Brain(*found)
                    with self.lock:
                        same_playlist = self.brain is not None and [p for _, p in self.brain.playlist] == [p for _, p in brain.playlist]
                        self.brain = brain
                        if not same_playlist:
                            self.slots = [Slot(title, paths) for title, paths in brain.playlist]
                            for slot in self.slots:
                                self._restart(slot, first=True)
                        self.layout_json = self._layout()
                    current = found
                    print(f"[live] using {brain.run.name} step={brain.step}", flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"[live] weight load failed: {error}", flush=True)
            time.sleep(30)

    # ---------------------------------------------------------------- シミュレーション

    def _restart(self, slot: Slot, first: bool = False) -> None:
        import mujoco
        from .mjx_tracking import seat_pose
        brain = self.brain
        if not first:
            slot.turn = (slot.turn + 1) % len(slot.paths)
        slot.path = slot.paths[slot.turn]
        ref = brain.ref[slot.path]
        clip = brain.env.clips[brain.index[slot.path]]
        data = mujoco.MjData(brain.model)
        data.qpos[:] = ref["qpos"][0]
        data.qvel[:] = ref["qvel"][0]
        if brain.seated:
            data.mocap_pos[0], data.mocap_quat[0] = seat_pose(clip.get("seat"))
            slot.seat = ([round(float(v), 3) for v in data.mocap_pos[0]] if clip.get("seat") is not None else None)
        else:
            slot.seat = None
        mujoco.mj_forward(brain.model, data)
        slot.data, slot.frame, slot.state, slot.wait, slot.err = data, 0, "tracking", 0.0, 0.0
        slot.last_act = np.zeros(63, np.float32)
        slot.linvel = np.zeros((22, 3), np.float32)  # 先頭フレームの参照速度は 0(_ref_linvel と同じ)
        # 区画の中心に来る位置: 参照の腰の軌跡の中点(歩く・走るでも区画の中を通る)
        end = min(len(ref["qpos"]) - 1, int(MAX_CLIP_SECONDS * HZ))
        slot.origin = [float(ref["qpos"][0][0] + ref["qpos"][end][0]) / 2, float(ref["qpos"][0][1] + ref["qpos"][end][1]) / 2]
        slot.failed_at = None
        slot.text = clip["text"]

    def simulate(self) -> None:
        import mujoco
        from . import body as B
        nb = len(B.SEGMENTS)
        period = 1.0 / HZ
        next_time = time.perf_counter()
        while True:
            next_time += period
            with self.lock:
                brain, slots = self.brain, list(self.slots)
            if brain is None or not slots:
                time.sleep(0.2)
                next_time = time.perf_counter()
                continue
            active = [s for s in slots if s.state == "tracking"]
            for slot in slots:
                if slot.state != "tracking":
                    slot.wait -= period
                    if slot.wait <= 0:
                        self._restart(slot)
            if active:
                # 形を固定して全員ぶん計算する(動いている体の数で形が変わると、そのたびに再コンパイルになる)
                views = {k: np.stack([np.asarray(getattr(s.data, k)) for s in slots])
                         for k in ("xpos", "xquat", "cvel", "qpos", "qvel")}
                infos = {"clip": np.asarray([brain.index[s.path] for s in slots], np.int32),
                         "frame": np.asarray([s.frame for s in slots], np.int32),
                         "last_act": np.stack([s.last_act for s in slots]),
                         "linvel": np.stack([s.linvel for s in slots]).astype(np.float32)}
                obs = brain.observe(views, infos)
                actions, _ = brain.policy(obs, brain.key)
                actions = np.clip(np.asarray(actions), -1.0, 1.0)
                for slot, action in zip(slots, actions):
                    if slot.state != "tracking":
                        continue
                    ref = brain.ref[slot.path]
                    row = min(slot.frame + 1, len(ref["qpos"]) - 1)
                    data = slot.data
                    data.ctrl[:] = ref["qpos"][row, 7:] + brain.config.action_scale * action
                    prev = data.xpos[1:nb + 1].copy()
                    for _ in range(B.CONTROL_SUBSTEPS):
                        mujoco.mj_step(brain.model, data)
                    slot.frame += 1
                    slot.last_act = action.astype(np.float32)
                    slot.linvel = ((data.xpos[1:nb + 1] - prev) * HZ).astype(np.float32)
                    err = float(np.linalg.norm(data.xpos[1:nb + 1] - ref["xpos"][row], axis=-1).mean())
                    slot.err = 0.9 * slot.err + 0.1 * err
                    if err > brain.config.termination_distance or not np.isfinite(data.qpos).all():
                        slot.state, slot.wait, slot.failed_at = "failed", PAUSE_AFTER_FAIL, slot.frame / HZ
                    elif slot.frame >= min(len(ref["qpos"]) - 1, int(MAX_CLIP_SECONDS * HZ)):
                        slot.state, slot.wait = "done", PAUSE_AFTER_END
            self.tick += 1
            if self.tick % SEND_EVERY == 0:
                self._publish(brain, slots, nb)
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:  # 大きく遅れたら追いつこうとしない
                next_time = time.perf_counter()

    def _publish(self, brain: Brain, slots: list[Slot], nb: int) -> None:
        frame = {"t": round(time.time(), 3), "step": brain.step, "run": brain.run.name, "slots": []}
        for slot in slots:
            data = slot.data
            poses = np.concatenate([data.xpos[1:nb + 1], data.xquat[1:nb + 1]], axis=1).round(3).ravel().tolist()
            ref = brain.ref[slot.path]
            frame["slots"].append({
                "title": slot.title, "text": slot.text, "state": slot.state,
                "seconds": round(slot.frame / HZ, 1), "length": round(min(len(ref["qpos"]) - 1, MAX_CLIP_SECONDS * HZ) / HZ, 1),
                "errCm": round(slot.err * 100, 1), "failedAt": slot.failed_at, "origin": [round(v, 3) for v in slot.origin],
                "seat": slot.seat, "poses": poses,
            })
        text = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        with self.cond:
            self.frame_json = text
            self.status = {"state": "running", "run": brain.run.name, "step": brain.step,
                           "slots": [{k: v for k, v in s.items() if k != "poses"} for s in frame["slots"]]}
            self.cond.notify_all()

    def _layout(self) -> str:
        """身体の形(体節ごとの幾何)と椅子の大きさ。描画側が姿勢と組み合わせて描く。"""
        import mujoco
        from . import body as B
        from .seats import SEAT_HALF
        model = self.brain.model
        nb = len(B.SEGMENTS)
        kinds = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                 int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}
        geoms = []
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            kind = kinds.get(int(model.geom_type[g]))
            if kind is None or not 1 <= body <= nb:
                continue
            geoms.append({"body": body - 1, "type": kind, "size": [round(float(v), 4) for v in model.geom_size[g]],
                          "pos": [round(float(v), 4) for v in model.geom_pos[g]],
                          "quat": [round(float(v), 5) for v in model.geom_quat[g]]})
        return json.dumps({"bodies": nb, "geoms": geoms, "seatHalf": list(SEAT_HALF),
                           "slots": [s.title for s in self.slots]}, ensure_ascii=False)


LIVE = Live()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _headers(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")

    def do_GET(self):
        if self.path.startswith("/status"):
            body = json.dumps(LIVE.status, ensure_ascii=False).encode()
            self._headers("application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self.path.startswith("/stream"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._headers("text/event-stream")
        self.end_headers()
        sent_layout = None
        try:
            while True:
                with LIVE.cond:
                    LIVE.cond.wait(timeout=5)
                    frame, layout = LIVE.frame_json, LIVE.layout_json
                if layout and layout is not sent_layout:
                    self.wfile.write(f"event: layout\ndata: {layout}\n\n".encode())
                    sent_layout = layout
                if frame:
                    self.wfile.write(f"event: frame\ndata: {frame}\n\n".encode())
                else:
                    self.wfile.write(b": waiting\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    threading.Thread(target=LIVE.watch_weights, daemon=True).start()
    threading.Thread(target=LIVE.simulate, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[live] serving on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
