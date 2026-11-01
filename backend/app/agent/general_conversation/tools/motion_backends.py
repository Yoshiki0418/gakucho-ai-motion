"""モーション生成バックエンドの切り替え層。

`MOTION_BACKEND` 環境変数で実装を選ぶ。

    MOTION_BACKEND=custom-t2m   (既定) テキスト → モーション。現行の Custom-T2M-Lab
    MOTION_BACKEND=sentiavatar  音声 + 行動タグ → モーション。SentiAvatar

なぜ抽象化するのか
------------------
現行の Custom-T2M-Lab は **text-to-motion** で、音声波形を一切見ていない。
そのため「意味は合うが時間は合わない」という限界が原理的にある。
一方 SentiAvatar は **audio + action text → motion** なので、
時間整合が生成器の内側で解ける。

両者は入力が違う(音声を渡せるかどうか)ので、呼び出し側が
どちらでも同じ形で扱えるようにここで境界を作る。バックエンドを差し替えても
`motion_tool.py` 以降のジョブ管理・SSE 配信・フロントエンドは変更不要。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .motion_handedness import (
    RIGHT_RAISE_CAPTION,
    correct_generated_hand_raise,
    mirror_bvh_sagittal,
    simple_hand_raise_constraint,
    simple_left_generation_caption,
)

MOTION_BACKEND = os.getenv("MOTION_BACKEND", "custom-t2m").strip().lower()


@dataclass
class MotionRequest:
    """モーション 1 本ぶんの生成依頼。"""

    prompt: str
    """日本語の動作記述(Custom-T2M-Lab はこれだけを使う)"""

    catalog_id: Optional[str] = None
    """カタログ ID。SentiAvatar の行動タグへの変換に使う"""

    audio_wav: Optional[bytes] = None
    """合わせる発話の音声。音声駆動バックエンドが使う"""

    metadata: dict = field(default_factory=dict)


@dataclass
class MotionResult:
    bvh: str
    prompt: str
    latency_seconds: float
    backend: str


class MotionBackend(Protocol):
    name: str
    requires_audio: bool
    """生成に発話音声が要るか。True なら TTS 完了後にしか呼べない。"""

    def generate(self, request: MotionRequest) -> MotionResult:
        ...

    def health(self) -> dict:
        ...


# ---------------------------------------------------------------- Custom-T2M


class CustomT2MBackend:
    """現行の Custom-T2M-Lab サーバー(text-to-motion)。"""

    name = "custom-t2m"
    # テキストだけで生成できるので、TTS を待たずに走らせられる
    requires_audio = False

    def __init__(self) -> None:
        self.server_url = os.getenv("T2M_SERVER_URL", "http://t2m:8099").rstrip("/")
        self.model_name = os.getenv("T2M_MODEL_NAME", "t2m_gpt")
        self.generate_timeout = float(os.getenv("T2M_GENERATE_TIMEOUT", "45"))
        self.download_timeout = float(os.getenv("T2M_DOWNLOAD_TIMEOUT", "15"))
        # Optional per-project sampling overrides.  Unset values leave the
        # server's manifest defaults in force and preserve the text-only API.
        self.generation_options: dict[str, float | int] = {}
        for env_name, option, minimum, maximum in (
            ("T2M_CFG_SCALE", "cfg_scale", 0.0, 10.0),
            ("T2M_TEMPERATURE", "temperature", 0.05, 5.0),
        ):
            raw_value = os.getenv(env_name)
            if raw_value is None or not raw_value.strip():
                continue
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{env_name} must be a finite number between {minimum} and {maximum}"
                ) from error
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"{env_name} must be a finite number between {minimum} and {maximum}"
                )
            self.generation_options[option] = value
        raw_top_k = os.getenv("T2M_TOP_K")
        if raw_top_k is not None and raw_top_k.strip():
            try:
                top_k = int(raw_top_k)
            except ValueError as error:
                raise ValueError("T2M_TOP_K must be an integer greater than or equal to 0") from error
            if top_k < 0:
                raise ValueError("T2M_TOP_K must be an integer greater than or equal to 0")
            self.generation_options["top_k"] = top_k
        self._model_loaded = False
        self._lock = threading.Lock()

    # -- HTTP ヘルパ --------------------------------------------------

    def _post_json(self, path: str, payload: dict, timeout: float) -> dict:
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get_json(self, path: str, timeout: float) -> dict:
        request = urllib.request.Request(f"{self.server_url}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    # -- モデルのロード ------------------------------------------------

    def ensure_model_loaded(self, force: bool = False) -> None:
        if self._model_loaded and not force:
            return

        with self._lock:
            if self._model_loaded and not force:
                return

            try:
                status = self._get_json("/api/realtime/status", timeout=5)
                if status.get("current_model") == self.model_name:
                    self._model_loaded = True
                    return
            except Exception as error:  # noqa: BLE001
                print(f"[motion_backends] realtime status check failed: {error}")

            result = self._post_json(
                "/api/realtime/load",
                {"model": self.model_name},
                timeout=self.generate_timeout,
            )
            if result.get("status") != "success":
                raise RuntimeError(f"T2Mモデルのロードに失敗しました: {result}")

            self._model_loaded = True
            print(f"[motion_backends] realtime model loaded: {self.model_name}")

    # -- 生成 ----------------------------------------------------------

    def _request_generation(self, prompt: str) -> dict:
        # The realtime endpoint uses the model selected by /realtime/load.
        # Its strict request schema accepts generation options only.
        payload = {"text": prompt, **self.generation_options}
        try:
            self.ensure_model_loaded()
            return self._post_json(
                "/api/realtime/generate", payload, timeout=self.generate_timeout
            )
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8")
            error.close()
            try:
                detail = json.loads(error_body).get("detail", "")
            except json.JSONDecodeError:
                detail = error_body

            # モデルが落ちていた場合は 1 度だけ載せ直して再試行する
            if isinstance(detail, str) and (
                "No model is currently loaded" in detail
                or (error.code == 409 and "not loaded" in detail.lower())
            ):
                self.ensure_model_loaded(force=True)
                return self._post_json(
                    "/api/realtime/generate", payload, timeout=self.generate_timeout
                )

            raise RuntimeError(
                f"T2Mサーバーでエラーが発生しました (HTTP {error.code}): "
                f"{detail or error_body}"
            ) from error

    def _download_bvh(self, bvh_path: str) -> str:
        candidates = [f"{self.server_url}/bvh/{urllib.parse.quote(bvh_path)}"]
        static_prefix = "web_viewer/static/"
        if bvh_path.startswith(static_prefix):
            candidates.append(
                f"{self.server_url}/static/"
                f"{urllib.parse.quote(bvh_path.removeprefix(static_prefix))}"
            )

        last_error: Optional[Exception] = None
        for url in candidates:
            try:
                request = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(
                    request, timeout=self.download_timeout
                ) as response:
                    return response.read().decode("utf-8")
            except Exception as error:  # noqa: BLE001
                last_error = error

        raise RuntimeError(f"BVHの取得に失敗しました: {last_error}")

    def generate(self, request: MotionRequest) -> MotionResult:
        raise_constraint = simple_hand_raise_constraint(request.prompt)
        generation_caption = (
            RIGHT_RAISE_CAPTION if raise_constraint is not None
            else simple_left_generation_caption(request.prompt)
        )
        result = self._request_generation(generation_caption or request.prompt)
        bvh_path = result.get("bvh_path", "")
        if not bvh_path:
            raise RuntimeError(f"T2Mサーバーの応答にBVHが含まれていません: {result}")

        bvh = self._download_bvh(bvh_path)
        if raise_constraint is not None:
            bvh, generated_side, mirrored = correct_generated_hand_raise(bvh, raise_constraint)
            print(
                "[motion_backends] qualified generated hand raise "
                f"request={request.prompt!r} generation_caption={generation_caption!r} "
                f"requested_side={raise_constraint.requested_side!r} "
                f"generated_side={generated_side!r} mirrored={mirrored}"
            )
        elif generation_caption is not None:
            bvh = mirror_bvh_sagittal(bvh)
            print(
                "[motion_backends] generated left-hand motion by sagittal reflection "
                f"request={request.prompt!r} generation_caption={generation_caption!r}"
            )
        return MotionResult(
            bvh=bvh,
            prompt=request.prompt,
            latency_seconds=result.get("latency_seconds", 0),
            backend=self.name,
        )

    def health(self) -> dict:
        try:
            return {"backend": self.name, "ok": True,
                    "status": self._get_json("/api/realtime/status", timeout=5)}
        except Exception as error:  # noqa: BLE001
            return {"backend": self.name, "ok": False, "error": str(error)}


# --------------------------------------------------------------- SentiAvatar


class SentiAvatarBackend:
    """SentiAvatar(audio + action text → motion)。

    公式の推論スクリプトは `--audio_path` と `--action_text` を取り、
    BVH / UE 向け JSON / 同期済み WAV を出力する。
    ここではそれを HTTP で包んだサーバーを叩く前提にしている
    (`SENTIAVATAR_SERVER_URL`)。

    ライセンス上の注意:
        SentiPulse Non-Commercial Source License v1.0 は、営利法人による利用や
        営利組織の内部業務・本番システムでの利用を禁じている。
        コード・重み・データセットすべてが同条項の対象。
        利用可否を確認するまで、このバックエンドは既定では選択されない。
    """

    name = "sentiavatar"
    # 音声波形から生成するため、TTS が終わるまで呼べない。
    # 呼び出し側は「モーションを先に投げる」流れを切り替える必要がある。
    requires_audio = True

    def __init__(self) -> None:
        self.server_url = os.getenv(
            "SENTIAVATAR_SERVER_URL", "http://127.0.0.1:8110"
        ).rstrip("/")
        self.timeout = float(os.getenv("SENTIAVATAR_TIMEOUT", "30"))

    def generate(self, request: MotionRequest) -> MotionResult:
        if request.audio_wav is None:
            raise RuntimeError(
                "SentiAvatar は音声駆動のため、合わせる発話の WAV が必要です。"
                "音声より先にモーションを生成する経路からは呼べません。"
            )

        started = time.perf_counter()
        payload = {
            "action_text": action_text_for(request),
            "audio_wav_base64": _b64(request.audio_wav),
            "format": "bvh",
        }
        http_request = urllib.request.Request(
            f"{self.server_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

        bvh = result.get("bvh", "")
        if not bvh:
            raise RuntimeError(f"SentiAvatar の応答に BVH がありません: {result}")
        bvh = fix_bvh_frame_time(bvh)

        return MotionResult(
            bvh=bvh,
            prompt=request.prompt,
            latency_seconds=time.perf_counter() - started,
            backend=self.name,
        )

    def health(self) -> dict:
        try:
            http_request = urllib.request.Request(
                f"{self.server_url}/health", method="GET"
            )
            with urllib.request.urlopen(http_request, timeout=5) as response:
                return {
                    "backend": self.name,
                    "ok": True,
                    "status": json.loads(response.read().decode("utf-8")),
                }
        except Exception as error:  # noqa: BLE001
            return {"backend": self.name, "ok": False, "error": str(error)}


# SentiAvatar のモーションは 30fps だが、BVH ライタは Frame Time を 20fps 相当の
# 0.0500 で書き出す。そのままだと尺が一律 1.5 倍になり音声と合わない。
# (中国語のデモ音声でも同じ 1.50 倍になるので、日本語固有の問題ではない)
_FRAME_TIME_RE = re.compile(r"(Frame Time:\s*)([\d.]+)")
SENTIAVATAR_FPS = 30.0


def fix_bvh_frame_time(bvh: str, fps: float = SENTIAVATAR_FPS) -> str:
    """BVH の Frame Time を実際のフレームレートへ直す。"""
    return _FRAME_TIME_RE.sub(lambda m: f"{m.group(1)}{1.0 / fps:.6f}", bvh, count=1)


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


# SentiAvatar の行動タグは中国語。カタログ ID から対応させる。
# 未対応の ID はプロンプトをそのまま渡す(モデル側の汎化に任せる)。
# 推論スクリプトは "动作：<中国語の動作記述>" 形式を想定している。
SENTIAVATAR_ACTION_PREFIX = "动作："

SENTIAVATAR_ACTION_TEXT: dict[str, str] = {
    "MOT_701": "挥手打招呼",
    "MOT_740": "挥手告别",
    "MOT_711": "鞠躬",
    "MOT_702": "点头",
    "MOT_703": "摇头",
    "MOT_714": "竖大拇指",
    "MOT_715": "大拇指向下",
    "MOT_707": "伸手示意停止",
    "MOT_706": "比OK手势",
    "MOT_705": "指向前方",
    "MOT_721": "指向右方",
    "MOT_738": "手放胸前表示感谢",
    "MOT_739": "双手合十请求",
    "MOT_708": "耸肩表示不知道",
    "MOT_748": "手指放在唇边思考",
    "MOT_709": "鼓掌",
    "MOT_735": "握拳加油",
    "MOT_716": "比剪刀手",
    "MOT_710": "敬礼",
    "MOT_719": "招手让人过来",
    "MOT_720": "伸手示意请",
}


def action_text_for(request: MotionRequest) -> str:
    """SentiAvatar へ渡す行動タグを決める。"""
    if request.catalog_id:
        mapped = SENTIAVATAR_ACTION_TEXT.get(request.catalog_id)
        if mapped:
            return f"{SENTIAVATAR_ACTION_PREFIX}{mapped}"
    # 対応表に無い場合は日本語の記述をそのまま渡す(モデルの汎化に任せる)
    return f"{SENTIAVATAR_ACTION_PREFIX}{request.prompt}"


_BACKENDS: dict[str, type] = {
    CustomT2MBackend.name: CustomT2MBackend,
    SentiAvatarBackend.name: SentiAvatarBackend,
}

_instance: Optional[MotionBackend] = None
_instance_lock = threading.Lock()


def get_motion_backend() -> MotionBackend:
    """設定されたバックエンドを返す(プロセス内で 1 つを使い回す)。"""
    global _instance
    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is None:
            factory = _BACKENDS.get(MOTION_BACKEND)
            if factory is None:
                raise RuntimeError(
                    f"未知の MOTION_BACKEND: {MOTION_BACKEND!r} "
                    f"(選択肢: {', '.join(_BACKENDS)})"
                )
            _instance = factory()
            print(f"[motion_backends] using backend: {_instance.name}")
    return _instance
