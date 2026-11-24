"""会話画面の動作生成モードと、層構成方式の 1 ターンぶんの処理。

モード(Web 画面から切り替え、`motion_mode` クエリで届く)
    current          現行。行動タグごとに T2M で生成する
    layered          新方式。意味の層(事前作成クリップ)+ リズムの層(SentiAvatar)+ 待機の層
    layered-semantic 新方式からリズムの層を抜いたもの(比較用)

層構成では、発話 1 つにつき 1 本の BVH を作る。音声合成が終わった時点で
意味の層だけの版(revision 0)をすぐ送り、SentiAvatar の結果が揃ったら
リズムを足した版(revision 1)で差し替える。フロントエンドは同じ group の
新しい版を、再生中なら現在位置から引き継いで切り替える。
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
import urllib.request

from .bvh import write_bvh
from .composer import Overlay, Pose, compose, pose_at
from .library import ASSET_DIR, ClipLibrary, get_library, library_available
from .rhythm import RhythmRetargeter
from .session import MotionSession, PlannedTag, get_session

MODES = ("current", "layered", "layered-semantic")
DEFAULT_MODE = "current"

# SentiAvatar(研究評価用)は tools/sentiavatar/run-container.sh で同じ Docker ネットワークに置く
SENTIAVATAR_URL = os.getenv("SENTIAVATAR_SERVER_URL", "http://sentiavatar:8110").rstrip(
    "/"
)
SENTIAVATAR_TIMEOUT = float(os.getenv("SENTIAVATAR_TIMEOUT", "30"))
RHYTHM_ACTION_TEXT = (
    "动作：说话"  # リズムの層は「話している」だけを渡し、意味の動作は重ねる側で出す
)
RHYTHM_AMPLITUDE = float(os.getenv("LAYERED_RHYTHM_AMPLITUDE", "1.0"))

ACK_CLIP = "MOT_702"  # 受け取りのうなずき(待機・傾聴の層)


def parse_mode(value: str | None) -> str:
    value = (value or "").strip().lower()
    return value if value in MODES else DEFAULT_MODE


def layered_available() -> bool:
    return library_available()


_retargeter: RhythmRetargeter | None = None
_retargeter_lock = threading.Lock()


def get_retargeter(library: ClipLibrary) -> RhythmRetargeter:
    global _retargeter
    with _retargeter_lock:
        if _retargeter is None:
            reference_path = ASSET_DIR / "sentiavatar_reference.json"
            reference = (
                json.loads(reference_path.read_text())
                if reference_path.exists()
                else None
            )
            _retargeter = RhythmRetargeter(
                library.skeleton,
                library.neutral,
                library.root_height,
                library.fps,
                reference,
            )
        return _retargeter


def request_rhythm(audio_wav: bytes) -> tuple[str, dict]:
    body = json.dumps(
        {
            "audio_wav_base64": base64.b64encode(audio_wav).decode("ascii"),
            "action_text": RHYTHM_ACTION_TEXT,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{SENTIAVATAR_URL}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=SENTIAVATAR_TIMEOUT) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["bvh_raw"], result.get("timing", {})


def _describe(overlays: list[Overlay]) -> list[dict]:
    return [
        {
            "tag": o.tag,
            "id": o.clip.id,
            "label": o.clip.label[:40],
            "source": o.clip.source,
            "start": round(o.start, 2),
            "stroke": round(o.stroke_time, 2),
            "length": round(o.length, 2),
            "amplitude": o.amplitude,
            "explicit": o.explicit,
        }
        for o in overlays
    ]


class LayeredTurn:
    """1 回の応答ぶん。ルータから発話ごとに呼ばれ、BVH を作ってチャネルへ積む。"""

    def __init__(self, channel, session_key: str, *, rhythm: bool) -> None:
        self.channel = channel
        self.rhythm = rhythm
        self.library = get_library()
        self.session: MotionSession = get_session(session_key)
        self.session.begin_turn()
        self.end_poses: dict[int, Pose] = {}
        self.jobs: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.worker_lock = threading.Lock()
        self.mode = "layered" if rhythm else "layered-semantic"

    # ------------------------------------------------------------ 送信形式

    def _payload(
        self,
        segment_id: int,
        motion,
        info: dict,
        revision: int,
        *,
        play_now: bool = False,
        latency: float = 0.0,
    ) -> dict:
        tags = [o["tag"] for o in info.get("overlays", [])]
        return {
            "status": "ok",
            "prompt": "layered: "
            + (" + ".join(tags) if tags else "idle")
            + (" + rhythm" if info["base"] == "rhythm" else ""),
            "bvh": write_bvh(motion, rotation_digits=2, position_digits=4),
            "latency_seconds": latency,
            "backend": "layered",
            "segment_index": segment_id,
            "start_offset_ratio": 0.0,
            "layered": {
                "mode": self.mode,
                "group": f"seg-{segment_id}",
                "revision": revision,
                "playNow": play_now,
                **info,
            },
        }

    # ------------------------------------------------------------ 待機・傾聴の層

    def acknowledge(self) -> dict | None:
        """ユーザーの発話を受け取った合図の浅いうなずき。応答の音声より先に出る。"""
        clip = self.library.get(ACK_CLIP)
        if clip is None:
            return None
        overlay = Overlay(
            clip, 0.05, min(clip.duration, clip.stroke + 0.7), tag="ack", amplitude=0.55
        )
        composition = compose(self.library, 1.6, [overlay], clock=self.session.clock)
        info = {"base": "idle", "overlays": _describe([overlay]), "layer": "listening"}
        return self._payload(0, composition.motion, info, 0, play_now=True)

    # ------------------------------------------------------------ 発話ごと

    def on_sentence(
        self,
        segment_id: int,
        duration_ms: float,
        tags: list[PlannedTag],
        audio_wav: bytes | None,
    ) -> dict:
        """意味の層だけの版を作って返す。リズムの層は裏で作ってチャネルへ積む。"""
        started = time.perf_counter()
        duration = max(duration_ms / 1000.0, 0.3)
        plan = self.session.plan(self.library, segment_id, duration, tags)
        composition = compose(
            self.library,
            duration,
            plan.overlays,
            start_pose=plan.start_pose,
            clock=plan.clock,
        )
        end_pose = pose_at(composition.motion, duration)
        self.end_poses[segment_id] = end_pose
        self.session.update_end_pose(end_pose)
        info = {
            "base": "idle",
            "overlays": _describe(plan.overlays),
            "skipped": plan.skipped,
            "duration": round(duration, 3),
            "rhythm": "pending" if (self.rhythm and audio_wav) else "off",
        }
        payload = self._payload(
            segment_id,
            composition.motion,
            info,
            0,
            latency=time.perf_counter() - started,
        )
        if self.rhythm and audio_wav:
            self._submit(plan, audio_wav)
        return payload

    def _submit(self, plan, audio_wav: bytes) -> None:
        self.channel.start_job()
        with self.worker_lock:
            self.jobs.put((plan, audio_wav))
            if self.worker is None:
                self.worker = threading.Thread(
                    target=self._run, name="layered-rhythm", daemon=True
                )
                self.worker.start()

    def _run(self) -> None:
        # 発話順に 1 本ずつ処理する(前の発話の最終姿勢を次へ引き継ぐため)
        while True:
            with self.worker_lock:
                try:
                    plan, audio_wav = self.jobs.get_nowait()
                except queue.Empty:
                    self.worker = None  # 次の発話が来たら新しく立てる
                    return
            started = time.perf_counter()
            try:
                bvh_raw, timing = request_rhythm(audio_wav)
                base = get_retargeter(self.library).retarget(
                    bvh_raw, amplitude=RHYTHM_AMPLITUDE
                )
                start_pose = self.end_poses.get(plan.segment_id - 1, plan.start_pose)
                composition = compose(
                    self.library,
                    plan.duration,
                    plan.overlays,
                    base=base,
                    start_pose=start_pose,
                    clock=plan.clock,
                )
                end_pose = pose_at(composition.motion, plan.duration)
                self.end_poses[plan.segment_id] = end_pose
                if max(self.end_poses) == plan.segment_id:
                    self.session.update_end_pose(end_pose)
                info = {
                    "base": "rhythm",
                    "overlays": _describe(plan.overlays),
                    "skipped": plan.skipped,
                    "duration": round(plan.duration, 3),
                    "rhythm": "ok",
                    "rhythmTiming": timing,
                }
                self.channel.put(
                    self._payload(
                        plan.segment_id,
                        composition.motion,
                        info,
                        1,
                        latency=time.perf_counter() - started,
                    )
                )
            except Exception as error:  # noqa: BLE001
                print(f"[layered] rhythm failed segment={plan.segment_id}: {error}")
                self.channel.put(
                    {
                        "status": "error",
                        "prompt": "rhythm",
                        "message": f"リズムの層の生成に失敗: {error}",
                        "segment_index": plan.segment_id,
                        "start_offset_ratio": 0.0,
                    }
                )
            finally:
                self.channel.finish_job()


def resolve_layered_request(prompt: str) -> tuple[str, bool] | None:
    """自由文の動作指示を、事前に作ったクリップ (catalog_id, 左右反転するか) に当てる。

    当たらないもの(座る・立つ・カタログに無い動作)は None を返し、呼び出し側は現行の T2M に回す。
    収録は右手なので、「左手で〜」は右手の動作として引いてから鏡映する。
    """
    import re

    from ..motion_catalog import normalize_motion_prompt

    library = get_library()
    left = bool(re.search(r"左(?:手|腕|の手)", prompt)) and not re.search(
        r"両(?:手|腕)", prompt
    )
    if not left:
        match = normalize_motion_prompt(prompt)
        if match.catalog_id and library.get(match.catalog_id):
            return match.catalog_id, False
        return None
    swapped = normalize_motion_prompt(prompt.replace("左", "右"))
    if swapped.catalog_id and library.get(swapped.catalog_id):
        return swapped.catalog_id, True
    return None
