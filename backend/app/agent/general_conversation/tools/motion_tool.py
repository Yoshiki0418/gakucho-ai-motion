"""
モーション生成ツール - Text-to-Motion サーバーと連携

ツール呼び出し自体は即時に完了させ、重いT2M生成はバックグラウンドで実行します。
エージェントはジョブ受理後すぐに応答生成を再開し、完成したBVHはリクエストごとの
MotionJobChannelを通じてSSEへ非同期配信されます。
"""

import contextvars
import json
import os
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from agents import function_tool

from .motion_backends import MotionRequest, get_motion_backend
from .motion_catalog import normalize_motion_prompt


T2M_SERVER_URL = os.getenv("T2M_SERVER_URL", "http://t2m:8099").rstrip("/")
T2M_MODEL_NAME = os.getenv("T2M_MODEL_NAME", "t2m_gpt")
T2M_PRELOAD_MODEL = os.getenv("T2M_PRELOAD_MODEL", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
T2M_GENERATE_TIMEOUT = float(os.getenv("T2M_GENERATE_TIMEOUT", "45"))
T2M_DOWNLOAD_TIMEOUT = float(os.getenv("T2M_DOWNLOAD_TIMEOUT", "15"))
MOTION_JOB_WAIT_TIMEOUT = T2M_GENERATE_TIMEOUT * 2 + T2M_DOWNLOAD_TIMEOUT + 15

_realtime_model_loaded = False
_realtime_model_lock = threading.Lock()


class MotionJobChannel:
    """バックグラウンド生成結果をSSEへ安全に受け渡すリクエスト単位のチャネル。"""

    def __init__(self, explicit_prompt: Optional[str] = None) -> None:
        # A direct user motion request owns this turn.  It must run once,
        # without reply-intent gestures or later sentences restarting it.
        self.explicit_prompt = explicit_prompt
        self._explicit_claimed = False
        self._results: queue.Queue[dict] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        # オーケストレータが発話を送出するたびに進めるカウンタ。
        # ツール呼び出し時点の値を控えておくことで、生成された BVH を
        # 「どの発話に合わせて再生すべきか」へ結び付けられる。
        self._segment_index = 0
        # 投入済みの (発話番号, 動作) の組。行動タグと generate_motion ツールの
        # 両方から同じ動作が来ることがあるため、同じ発話への重複だけを弾く。
        # (プロンプトで「両方出すな」と書いても守られないことがある)
        #
        # 発話をまたいだ繰り返しは弾かない。長い応答で 2 回うなずくのは
        # 自然な振る舞いであり、ここで潰すと長文がまた無動作になる。
        self._requested: set[tuple[int, str]] = set()
        # 音声駆動バックエンド用。ツールから要求された動作を、
        # TTS が終わるまでここに預けておく(音声が無いと生成できないため)。
        # 層構成方式でも、事前作成クリップで表せる動作は同じくここに預け、次の発話に重ねる。
        self._pending_actions: list[tuple[str, Optional[str]]] = []
        self.layered = False

    def request_action(self, prompt: str, catalog_id: Optional[str] = None) -> None:
        """音声が揃うまで動作要求を預かる。"""
        with self._lock:
            self._pending_actions.append((prompt, catalog_id))

    def drain_actions(self) -> list[tuple[str, Optional[str]]]:
        """預かっていた動作要求を取り出す。"""
        with self._lock:
            items = list(self._pending_actions)
            self._pending_actions.clear()
            return items

    def claim_prompt(self, prompt: str, segment_index: int) -> bool:
        """その発話にまだ投入していない動作なら True を返し、記録する。"""
        key = " ".join(prompt.strip().split())
        if not key:
            return False
        with self._lock:
            if self.explicit_prompt is not None:
                requested = " ".join(self.explicit_prompt.strip().split())
                if key != requested or self._explicit_claimed:
                    return False
                self._explicit_claimed = True
            entry = (segment_index, key)
            if entry in self._requested:
                return False
            self._requested.add(entry)
            return True

    @property
    def segment_index(self) -> int:
        with self._lock:
            return self._segment_index

    def advance_segment(self) -> int:
        """発話を 1 つ送出したことを記録し、新しいインデックスを返す。"""
        with self._lock:
            self._segment_index += 1
            return self._segment_index

    def start_job(self) -> None:
        with self._lock:
            self._pending += 1

    def finish_job(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def put(self, result: dict) -> None:
        self._results.put(result)

    def get(self, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[dict]:
        items: list[dict] = []
        while True:
            try:
                items.append(self._results.get_nowait())
            except queue.Empty:
                return items

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    @property
    def is_idle(self) -> bool:
        return self.pending_count == 0 and self._results.empty()


motion_data_queue_var: contextvars.ContextVar[Optional[MotionJobChannel]] = (
    contextvars.ContextVar("motion_data_queue_var", default=None)
)

# 同じ指示の再生成を避ける小さなインメモリキャッシュ。
_motion_cache: dict[str, dict] = {}
_motion_cache_lock = threading.Lock()
_MOTION_CACHE_MAX_ITEMS = 12


def _post_json(path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{T2M_SERVER_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(path: str, timeout: float) -> dict:
    request = urllib.request.Request(f"{T2M_SERVER_URL}{path}", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ensure_realtime_model_loaded(force: bool = False) -> None:
    global _realtime_model_loaded

    if _realtime_model_loaded and not force:
        return

    with _realtime_model_lock:
        if _realtime_model_loaded and not force:
            return

        try:
            status = _get_json("/api/realtime/status", timeout=5)
            if status.get("current_model") == T2M_MODEL_NAME:
                _realtime_model_loaded = True
                return
        except Exception as error:
            print(f"[motion_tool] realtime status check failed: {error}")

        result = _post_json(
            "/api/realtime/load",
            {"model": T2M_MODEL_NAME},
            timeout=T2M_GENERATE_TIMEOUT,
        )
        if result.get("status") != "success":
            raise RuntimeError(f"T2Mモデルのロードに失敗しました: {result}")

        _realtime_model_loaded = True
        print(f"[motion_tool] realtime model loaded: {T2M_MODEL_NAME}")


def _get_cached_motion(key: str) -> Optional[dict]:
    with _motion_cache_lock:
        return _motion_cache.get(key)


def _cache_motion(key: str, motion_data: dict) -> None:
    with _motion_cache_lock:
        if len(_motion_cache) >= _MOTION_CACHE_MAX_ITEMS:
            _motion_cache.pop(next(iter(_motion_cache)))
        _motion_cache[key] = motion_data


def _request_motion_generation(motion_description: str) -> dict:
    payload = {"text": motion_description}

    try:
        _ensure_realtime_model_loaded()
        return _post_json(
            "/api/realtime/generate",
            payload,
            timeout=T2M_GENERATE_TIMEOUT,
        )
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8")
        error.close()
        try:
            detail = json.loads(error_body).get("detail", "")
        except json.JSONDecodeError:
            detail = error_body

        if isinstance(detail, str) and (
            "No model is currently loaded" in detail
            or (error.code == 409 and "not loaded" in detail.lower())
        ):
            _ensure_realtime_model_loaded(force=True)
            return _post_json(
                "/api/realtime/generate",
                payload,
                timeout=T2M_GENERATE_TIMEOUT,
            )

        raise RuntimeError(
            f"T2Mサーバーでエラーが発生しました (HTTP {error.code}): "
            f"{detail or error_body}"
        ) from error


def _generate_motion_sync(
    motion_description: str,
    cache_key: str,
    catalog_id: Optional[str] = None,
    audio_wav: Optional[bytes] = None,
) -> dict:
    """モーションを 1 本生成する。実際の生成はバックエンド層が担う。"""
    backend = get_motion_backend()
    result = backend.generate(
        MotionRequest(
            prompt=motion_description,
            catalog_id=catalog_id,
            audio_wav=audio_wav,
        )
    )
    motion_data = {
        "status": "ok",
        "prompt": motion_description,
        "bvh": result.bvh,
        "latency_seconds": result.latency_seconds,
        "backend": result.backend,
    }
    # Audio-driven output belongs to this exact speech waveform, not merely
    # its action label or per-turn segment number.
    if audio_wav is None:
        _cache_motion(cache_key, motion_data)
    return motion_data


def _generate_motion_legacy(motion_description: str, cache_key: str) -> dict:
    result = _request_motion_generation(motion_description)
    bvh_path = result.get("bvh_path", "")
    latency = result.get("latency_seconds", 0)
    if not bvh_path:
        raise RuntimeError("モーション生成に失敗しました（BVHパスが空です）")

    bvh_text = _download_bvh_text(bvh_path)

    motion_data = {
        "status": "success",
        "prompt": motion_description,
        "latency_seconds": latency,
        "bvh": bvh_text,
    }
    _cache_motion(cache_key, motion_data)
    return motion_data


def _download_bvh_text(bvh_path: str) -> str:
    candidates = [f"{T2M_SERVER_URL}/bvh/{urllib.parse.quote(bvh_path)}"]
    static_prefix = "web_viewer/static/"
    if bvh_path.startswith(static_prefix):
        candidates.append(
            f"{T2M_SERVER_URL}/static/{urllib.parse.quote(bvh_path.removeprefix(static_prefix))}"
        )

    last_error: Exception | None = None
    for url in candidates:
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(
                request,
                timeout=T2M_DOWNLOAD_TIMEOUT,
            ) as response:
                return response.read().decode("utf-8")
        except Exception as error:
            last_error = error

    raise RuntimeError(f"BVHの取得に失敗しました: {last_error}")


def _run_motion_job(
    motion_description: str,
    cache_key: str,
    channel: MotionJobChannel,
    segment_index: int,
    catalog_id: Optional[str] = None,
    position: float = 0.0,
    audio_wav: Optional[bytes] = None,
) -> None:
    try:
        motion_data = _generate_motion_sync(
            motion_description, cache_key, catalog_id=catalog_id, audio_wav=audio_wav
        )
        # 生成完了時刻ではなく「どの発話の、どのあたりか」を持たせる。
        # 再生開始時刻はフロントエンドがこの情報から決める。
        motion_data = {
            **motion_data,
            "segment_index": segment_index,
            "start_offset_ratio": position,
        }
        channel.put(motion_data)
        print(
            f"[motion_tool] background job completed "
            f"prompt={motion_description!r} "
            f"bvh_len={len(motion_data['bvh'])} "
            f"latency={motion_data['latency_seconds']}"
        )
    except Exception as error:
        channel.put(
            {
                "status": "error",
                "prompt": motion_description,
                "message": str(error),
                "segment_index": segment_index,
                "start_offset_ratio": position,
            }
        )
        print(
            f"[motion_tool] background job failed "
            f"prompt={motion_description!r} error={error}"
        )
    finally:
        channel.finish_job()


def motion_backend_requires_audio() -> bool:
    """現在のバックエンドが発話音声を必要とするか。"""
    return bool(getattr(get_motion_backend(), "requires_audio", False))


def enqueue_motion_job(
    channel: MotionJobChannel,
    motion_description: str,
    segment_index: int,
    *,
    catalog_id: Optional[str] = None,
    position: float = 0.0,
    audio_wav: Optional[bytes] = None,
) -> bool:
    """応答本文の行動タグからモーション生成を起動する。

    LLM のツール呼び出しを経由しないので、タグが本文に現れた時点で
    即座に走らせられる。1 応答に複数のタグがあれば、その数だけ並ぶ。

    Args:
        segment_index: 合わせる発話の通し番号
        position: 文中のどのあたりか(0.0=先頭, 1.0=末尾)

    Returns:
        ジョブを受理したら True。キャッシュ命中時もその場で結果を積んで True。
    """
    cache_key = " ".join(motion_description.strip().split())
    if not cache_key:
        return False

    # Deduplicate the action itself.  Audio cache suffixes must not change
    # the prompt compared against an explicit user request.
    if not channel.claim_prompt(cache_key, segment_index):
        print(f"[motion_tool] duplicate skipped (tag): seg={segment_index} {cache_key!r}")
        return False

    # A repeated segment number across turns does not identify the same
    # speech waveform, so audio-driven generation never reuses text caches.
    cached = None if audio_wav is not None else _get_cached_motion(cache_key)
    if cached is not None:
        channel.put({
            **cached,
            "segment_index": segment_index,
            "start_offset_ratio": position,
        })
        return True

    channel.start_job()
    worker = threading.Thread(
        target=_run_motion_job,
        args=(motion_description, cache_key, channel, segment_index),
        kwargs={"catalog_id": catalog_id, "position": position, "audio_wav": audio_wav},
        name="motion-generation-tag",
        daemon=True,
    )
    worker.start()
    return True


@function_tool
def generate_motion(motion_description: str) -> str:
    """
    テキストで指定された動作に基づいて、3Dモーション（BVHデータ）を生成します。
    ジョブは即時に受理され、生成完了を待たずに会話応答を続行できます。

    motion_descriptionは必ず日本語で簡潔に記述してください。
    Custom-T2M-Labのカタログに近い表現ほど安定します。
    アバターが座っている文脈では「椅子に座った状態で、右手を振る」のように、
    現在の身体状態を含めてください。立つ指示があるまで勝手に立たせないでください。
    例: "右手を肩の高さに上げ、手のひらを前方に向けて軽く左右に振って挨拶する",
        "頭部をゆっくり上下に動かして同意のうなずきをする",
        "右手のひらを自分の胸に当てて「ありがとう・感謝しています」を示す",
        "立っている状態から、椅子に座る", "椅子に座っている状態から、手を振る。",
        "座った状態から、立ち上がる"

    Args:
        motion_description: 生成したいモーションの日本語テキスト説明
    """
    channel = motion_data_queue_var.get()
    if channel is None:
        return json.dumps(
            {
                "status": "unavailable",
                "message": (
                    "Motion delivery channel is unavailable. "
                    "Continue the response without mentioning this."
                ),
            }
        )

    if channel.explicit_prompt is not None:
        return json.dumps({
            "status": "accepted",
            "motion_description": channel.explicit_prompt,
            "message": (
                "The user's requested motion is already scheduled once. "
                "Continue the spoken reply without additional gestures."
            ),
        })

    catalog_match = normalize_motion_prompt(motion_description)
    normalized_description = catalog_match.prompt
    cache_key = " ".join(normalized_description.strip().split())

    # 行動タグですでに同じ動作が投入されていれば、ここでは何もしない。
    # プロンプトで「両方出すな」と書いても守られないことがあるので構造で防ぐ。
    if not channel.claim_prompt(cache_key, channel.segment_index + 1):
        print(f"[motion_tool] duplicate skipped (tool): {cache_key!r}")
        return json.dumps({
            "status": "duplicate",
            "motion_description": normalized_description,
            "message": (
                "This motion was already requested for this response. "
                "Continue without mentioning it."
            ),
        })

    # 層構成方式では、事前作成クリップで表せる動作は T2M を呼ばずに次の発話へ重ねる。
    # 座る・立つなどクリップに無いものは、下の現行経路(T2M)で生成する。
    if channel.layered:
        from .conversation_motion.pipeline import resolve_layered_request

        resolved = resolve_layered_request(motion_description)
        if resolved is not None:
            channel.request_action(motion_description, resolved[0])
            return json.dumps({
                "status": "accepted",
                "motion_description": normalized_description,
                "catalog_id": resolved[0],
                "message": "Queued as a prepared gesture. Continue the response without mentioning it.",
            })

    # 音声駆動バックエンドでは、音声が確定するまで生成できない。
    # ここでは要求だけ預け、TTS 完了後にオーケストレータが拾う。
    if motion_backend_requires_audio():
        channel.request_action(normalized_description, catalog_match.catalog_id)
        return json.dumps({
            "status": "accepted",
            "motion_description": normalized_description,
            "catalog_id": catalog_match.catalog_id,
            "message": "Queued for audio-driven generation. Continue the response.",
        })

    # ツール呼び出し時点で N 文が送出済みなら、いま組み立て中の文が N+1 になる。
    # モーションはその文に合わせて再生したいので +1 する。
    # audio_chunk の segment_id と直接突き合わせられる値になる。
    segment_index = channel.segment_index + 1

    cached = _get_cached_motion(cache_key)
    if cached is not None:
        channel.put({
            **cached,
            "segment_index": segment_index,
            "start_offset_ratio": 0.0,
        })
        return json.dumps(
            {
                "status": "success",
                "source": "cache",
                "motion_description": normalized_description,
                "catalog_id": catalog_match.catalog_id,
                "normalization": catalog_match.reason,
                "message": (
                    "Cached motion was queued. "
                    "Continue the response immediately without mentioning this."
                ),
            }
        )

    channel.start_job()
    worker = threading.Thread(
        target=_run_motion_job,
        args=(normalized_description, cache_key, channel, segment_index),
        kwargs={"catalog_id": catalog_match.catalog_id},
        name="motion-generation",
        daemon=True,
    )
    worker.start()

    return json.dumps(
        {
            "status": "accepted",
            "motion_description": normalized_description,
            "catalog_id": catalog_match.catalog_id,
            "normalization": catalog_match.reason,
            "message": (
                "Motion generation started in the background. "
                "Do not wait for completion. Continue the response immediately "
                "and do not mention this action."
            ),
        }
    )


def _preload_realtime_model_in_background() -> None:
    if not T2M_PRELOAD_MODEL:
        return

    def worker() -> None:
        try:
            _ensure_realtime_model_loaded()
        except Exception as error:
            print(f"[motion_tool] realtime model preload skipped: {error}")

    threading.Thread(
        target=worker,
        name="motion-realtime-preload",
        daemon=True,
    ).start()


_preload_realtime_model_in_background()
