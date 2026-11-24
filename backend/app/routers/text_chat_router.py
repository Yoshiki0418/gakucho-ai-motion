import asyncio
import base64

# import base64
# import io
import json
import os
import re
import time

# import cv2
# import numpy as np
# import soundfile as sf
from app.agent.agent_factory import build_conversation_orchestrator
from app.agent.general_conversation.tools.conversation_motion.pipeline import (
    LayeredTurn,
    PlannedTag,
    layered_available,
    parse_mode,
    resolve_layered_request,
)
from app.agent.general_conversation.tools.motion_catalog import (
    extract_explicit_motion_request,
)
from app.agent.general_conversation.tools.motion_output import (
    clean_motion_tool_artifacts,
)
from app.agent.general_conversation.tools.motion_tags import (
    extract_action_tags,
    has_incomplete_tag,
    plan_auto_gestures,
)
from app.agent.general_conversation.tools.motion_tool import (
    MOTION_JOB_WAIT_TIMEOUT,
    MotionJobChannel,
    enqueue_motion_job,
    motion_backend_requires_audio,
    motion_data_queue_var,
)
from app.chat_models import ChatModel, model_settings, select_model
from app.models import llm
from app.models.model_registry import create_llm, select_tts, tts_settings
from app.models.tts.irodori_tts import IrodoriTTSError
from app.services.history_service import HistoryService

# from app.models.lipsync.ditto_batch_streamer import AvatarStreamer
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

# import queue as sync_queue
# from typing import Dict


router = APIRouter(prefix="/api/text-chat", tags=["Text Chat"])

# 履歴の既定セッション。検証用バックエンドが本番の会話履歴に混ざらないよう環境変数で分けられる。
DEFAULT_SESSION_ID = os.getenv("CHAT_DEFAULT_SESSION_ID", "default")


@router.get("/model-settings")
def chat_model_settings():
    return {**model_settings(), "tts": tts_settings()}


def _request_model(request: Request) -> ChatModel:
    try:
        return select_model(
            request.query_params.get("llm_provider"),
            request.query_params.get("llm_model"),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


async def _request_tts(request: Request):
    try:
        return await asyncio.to_thread(
            select_tts,
            request.query_params.get("tts_provider"),
            request.query_params.get("tts_voice"),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


async def _stream_tts_errors(stream):
    try:
        async for event in stream:
            yield event
    except IrodoriTTSError as exc:
        yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"


@router.get("/stream")
async def text_chat_stream(request: Request):
    user_input = request.query_params.get("text", "こんにちは！")
    history = request.query_params.get("history", [])

    model = _request_model(request)
    _llm = create_llm(model.provider, llm.system_prompt, model.name)
    _tts = await _request_tts(request)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'message': f'{_llm.model_name} を使用します'})}\n\n"

        text_output = await _llm.generate(user_input, history)

        # TTSを並行で実行
        tts_task = asyncio.create_task(
            asyncio.to_thread(_tts.synthesize_to_base64, text_output)
        )
        yield f"data: {json.dumps({'type': 'text_result', 'content': text_output})}\n\n"

        audio_b64 = await tts_task
        yield f"data: {json.dumps({'type': 'audio_result', 'audio': audio_b64})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'message': '応答完了'})}\n\n"

    return StreamingResponse(
        _stream_tts_errors(event_stream()), media_type="text/event-stream"
    )


@router.get("/char-stream")
async def char_stream(request: Request):
    user_input = request.query_params.get("text", "こんにちは！")
    history = request.query_params.get("history", [])

    model = _request_model(request)
    _llm = create_llm(model.provider, llm.system_prompt, model.name)
    _tts = await _request_tts(request)

    PUNCTUATIONS = {"。", "！", "？", "!", "?"}

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'message': f'{_llm.model_name} を使用します'})}\n\n"

        sentence_buffer = ""

        # LLMのストリーミング出力を逐次処理
        async for chunk in _llm.stream_generate(user_input, history):
            text_piece = str(chunk)
            sentence_buffer += text_piece

            yield f"data: {json.dumps({'type': 'text_chunk', 'content': text_piece})}\n\n"

            # 文の終端を検出したら、その文をTTSに渡す
            if any(p in text_piece for p in PUNCTUATIONS):
                # 並列で音声生成
                current_sentence = sentence_buffer.strip()
                audio_b64 = await asyncio.to_thread(
                    _tts.synthesize_to_base64, current_sentence
                )

                yield f"data: {json.dumps({'type': 'audio_chunk', 'sentence': current_sentence, 'audio': audio_b64})}\n\n"

                # 文バッファをリセット
                sentence_buffer = ""

        # 最後に残った文を処理（句読点なしで終わった場合）
        if sentence_buffer.strip():
            audio_b64 = await asyncio.to_thread(
                _tts.synthesize_to_base64, sentence_buffer.strip()
            )
            yield f"data: {json.dumps({'type': 'audio_chunk', 'sentence': sentence_buffer.strip(), 'audio': audio_b64})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'message': '応答完了'})}\n\n"

    return StreamingResponse(
        _stream_tts_errors(event_stream()), media_type="text/event-stream"
    )


@router.get("/char-stream-agent")
async def char_stream_agent(request: Request):
    user_input = request.query_params.get("text", "こんにちは！")
    # history = request.query_params.get("history", [])

    agent = build_conversation_orchestrator(_request_model(request)).daily_agent
    _tts = await _request_tts(request)

    PUNCTUATIONS = {"。", "！", "？", "!", "?"}

    async def event_stream():
        sentence_buffer = ""

        # LLMのストリーミング出力を逐次処理
        async for chunk in agent.stream_generate(user_id="1", message=user_input):
            text_piece = str(chunk)
            sentence_buffer += text_piece

            yield f"data: {json.dumps({'type': 'text_chunk', 'content': text_piece})}\n\n"

            # 文の終端を検出したら、その文をTTSに渡す
            if any(p in text_piece for p in PUNCTUATIONS):
                # 並列で音声生成
                current_sentence = sentence_buffer.strip()
                audio_b64 = await asyncio.to_thread(
                    _tts.synthesize_to_base64, current_sentence
                )

                yield f"data: {json.dumps({'type': 'audio_chunk', 'sentence': current_sentence, 'audio': audio_b64})}\n\n"

                # 文バッファをリセット
                sentence_buffer = ""

        # 最後に残った文を処理（句読点なしで終わった場合）
        if sentence_buffer.strip():
            audio_b64 = await asyncio.to_thread(
                _tts.synthesize_to_base64, sentence_buffer.strip()
            )
            yield f"data: {json.dumps({'type': 'audio_chunk', 'sentence': sentence_buffer.strip(), 'audio': audio_b64})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'message': '応答完了'})}\n\n"

    return StreamingResponse(
        _stream_tts_errors(event_stream()), media_type="text/event-stream"
    )


# CFG_PKL = "/workspace/backend/app/weights/lipsync/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
# DATA_ROOT = "/workspace/backend/app/weights/lipsync/ditto_pytorch"
# SOURCE_IMAGE = "test.jpg"


# def audio_b64_to_array(audio_b64: str, target_sr: int = 16000) -> np.ndarray:
#     audio_bytes = base64.b64decode(audio_b64)
#     # soundfile でメモリ上から読む
#     data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
#     if data.ndim == 2:
#         data = data.mean(axis=1)  # stereo → mono
#     if sr != target_sr:
#         import librosa

#         data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
#         sr = target_sr
#     return data


# def sse_event(event_type: str, payload: dict) -> str:
#     data = {"type": event_type, **payload}
#     return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# async def push_audio_and_stream_frames(
#     audio_b64: str,
#     avatar_streamer: AvatarStreamer,
#     out_queue: "asyncio.Queue[str]",
#     frame_index_state: Dict[str, int],
#     fps: int = 24,
# ):
#     """
#     1文ぶんの音声を Ditto に push し、
#     その時点までに生成されたフレームを frame_chunk として SSEキューに積む。
#     """

#     # 1) 音声を Ditto に投入
#     def _push():
#         arr = audio_b64_to_array(audio_b64, target_sr=16000)
#         avatar_streamer.push_audio_array(arr, sr=16000)

#     await asyncio.to_thread(_push)

#     # 2) 今キューに溜まっているフレームだけ non-blocking で回収
#     frames = []
#     while True:
#         try:
#             frame = avatar_streamer.frame_output_queue.get_nowait()
#         except sync_queue.Empty:
#             break

#         if frame is None:
#             # 終了シグナル
#             break

#         frames.append(frame)

#     # 3) SSE キューに frame_chunk を積む
#     for frame in frames:
#         frame_bgr = frame[:, :, ::-1]
#         ok, buf = cv2.imencode(".jpg", frame_bgr)
#         if not ok:
#             continue
#         img_b64 = base64.b64encode(buf).decode("ascii")

#         idx = frame_index_state["value"]
#         frame_index_state["value"] += 1

#         await out_queue.put(
#             sse_event(
#                 "frame_chunk",
#                 {
#                     "frame_index": idx,
#                     "fps": fps,
#                     "image": img_b64,
#                 },
#             )
#         )


# # --- Ditto / AvatarStreamer のセットアップ ---
# avatar_streamer = AvatarStreamer(
#     CFG_PKL,
#     DATA_ROOT,
#     online_mode=False,  # 先頭フレーム欠損を避けるため False 推奨
#     output_path="dummy.mp4",  # 使わないが必須なら適当でOK
# )
# avatar_streamer.setup(
#     SOURCE_IMAGE,
#     "./dummy_output.mp4",
#     online_mode=False,
# )


# @router.get("/char-stream-agent-avatar")
# async def char_stream_agent_avatar(request: Request):
#     user_input = request.query_params.get("text", "こんにちは！")

#     # --- Agent / TTS の準備 ---
#     tts_provider = request.query_params.get("tts_provider", None)
#     tts_voice = request.query_params.get("tts_voice", None)

#     if tts_voice:
#         from app.models.model_registry import create_tts

#         _tts = create_tts(tts_provider or "style-bert-vits2")
#     else:
#         _tts = tts

#     PUNCTUATIONS = {"。", "！", "？", "!", "?"}
#     FPS = 25  # Dittoのフレームレート

#     async def event_stream():
#         sentence_buffer = ""

#         async for chunk in conversation_orchestrator.stream_response(user_id="1", text=user_input):
#             text_piece = str(chunk)
#             sentence_buffer += text_piece

#             # テキストチャンクをすぐイベントに
#             yield sse_event("text_chunk", {"content": text_piece})

#             # 文末検出 → 1文ぶんを TTS & リップシンク同時処理
#             if any(p in text_piece for p in PUNCTUATIONS):
#                 current_sentence = sentence_buffer.strip()
#                 if not current_sentence:
#                     continue

#                 # 同期的に TTS + リップシンクを処理
#                 def process_sentence():
#                     # 1. TTS で音声生成
#                     audio_b64 = _tts.synthesize_to_base64(current_sentence)

#                     # 2. 音声をnumpy arrayに変換
#                     audio_array = audio_b64_to_array(audio_b64, target_sr=16000)

#                     # 3. リップシンクフレーム生成 (全フレーム待機)
#                     frames = avatar_streamer.generate_frames_for_audio(audio_array, sr=16000)

#                     # 4. フレームをbase64エンコード
#                     frames_b64 = []
#                     for frame in frames:
#                         frame_bgr = frame[:, :, ::-1]
#                         ok, buf = cv2.imencode(".jpg", frame_bgr)
#                         if ok:
#                             frames_b64.append(base64.b64encode(buf).decode("ascii"))

#                     return audio_b64, frames_b64

#                 audio_b64, frames_b64 = await asyncio.to_thread(process_sentence)

#                 # 音声とフレームを一緒に送信
#                 yield sse_event(
#                     "audio_and_frames",
#                     {
#                         "sentence": current_sentence,
#                         "audio": audio_b64,
#                         "frames": frames_b64,
#                         "fps": FPS,
#                     },
#                 )

#                 # 文バッファをリセット
#                 sentence_buffer = ""

#         # 句読点なしで終わった場合の残りの文
#         if sentence_buffer.strip():
#             last_sentence = sentence_buffer.strip()

#             def process_last_sentence():
#                 audio_b64 = _tts.synthesize_to_base64(last_sentence)
#                 audio_array = audio_b64_to_array(audio_b64, target_sr=16000)
#                 frames = avatar_streamer.generate_frames_for_audio(audio_array, sr=16000)

#                 frames_b64 = []
#                 for frame in frames:
#                     frame_bgr = frame[:, :, ::-1]
#                     ok, buf = cv2.imencode(".jpg", frame_bgr)
#                     if ok:
#                         frames_b64.append(base64.b64encode(buf).decode("ascii"))

#                 return audio_b64, frames_b64

#             audio_b64, frames_b64 = await asyncio.to_thread(process_last_sentence)

#             yield sse_event(
#                 "audio_and_frames",
#                 {
#                     "sentence": last_sentence,
#                     "audio": audio_b64,
#                     "frames": frames_b64,
#                     "fps": FPS,
#                 },
#             )

#         # 最後に完了イベント
#         yield sse_event("done", {"message": "応答完了"})

#     return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- 対話履歴サービスの初期化 ---
try:
    history_service = HistoryService()
except Exception as e:
    print(f"[WARNING] HistoryService の初期化に失敗しました: {e}")
    history_service = None

# --- 音声出力用テキスト浄化 ---
_RE_URL = re.compile(r"https?://\S+")
_RE_PAREN_DOMAIN = re.compile(r"\s*\([a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\)")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_RE_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_RE_MD_BULLET = re.compile(r"^\s*[-*]\s+", re.MULTILINE)


def _sanitize_for_speech(text: str) -> str:
    """音声読み上げ向けにURL・出典・マークダウンを除去する"""
    text = _RE_URL.sub("", text)
    text = _RE_PAREN_DOMAIN.sub("", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = _RE_MD_BOLD.sub(r"\1", text)
    text = _RE_MD_BULLET.sub("", text)
    # 連続空白・改行を圧縮
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text


@router.get("/char-stream-orchestrator")
async def char_stream_orchestrator(request: Request):
    t0 = time.perf_counter()  # === 計測開始 ===
    user_input = request.query_params.get("text", "こんにちは！")
    user_id = request.query_params.get("user_id", "default")
    session_id = request.query_params.get("session_id", DEFAULT_SESSION_ID)
    mode = request.query_params.get("mode", "general")
    # 動作生成モード(画面で切り替え)。current は従来どおり、layered* は層構成方式。
    motion_mode = parse_mode(request.query_params.get("motion_mode"))
    avatar_posture = request.query_params.get("posture", "standing")

    _tts = await _request_tts(request)
    selected_model = _request_model(request)
    conversation_orchestrator = build_conversation_orchestrator(selected_model)

    # --- 対話履歴の取得 ---
    history = []
    if history_service:
        try:
            history = history_service.get_recent_history(
                user_id=user_id, session_id=session_id, n=10
            )
        except Exception as e:
            print(f"[WARNING] 履歴取得に失敗: {e}")
            history = []

    # --- ユーザーメッセージの保存 ---
    if history_service:
        try:
            history_service.save_message(
                user_id=user_id,
                session_id=session_id,
                role="user",
                content=user_input,
            )
        except Exception as e:
            print(f"[WARNING] ユーザーメッセージ保存に失敗: {e}")

    PUNCTUATIONS = {"。", "！", "？", "!", "?"}

    async def event_stream():
        # バックグラウンドのT2Mジョブから結果を受け取るリクエスト専用チャネル。
        explicit_motion = extract_explicit_motion_request(user_input)
        motion_channel = MotionJobChannel(
            explicit_prompt=explicit_motion.prompt if explicit_motion else None,
        )
        motion_data_queue_var.set(motion_channel)

        def serialize_motion_result(motion_data: dict) -> str:
            layered_info = motion_data.get("layered")
            if motion_data.get("status") == "error":
                payload = {
                    "type": "motion_error",
                    "prompt": motion_data.get("prompt", ""),
                    "message": motion_data.get(
                        "message", "モーション生成に失敗しました"
                    ),
                    "segment_id": motion_data.get("segment_index", 0),
                }
            else:
                payload = {
                    "type": "motion_chunk",
                    "bvh": motion_data.get("bvh", ""),
                    "prompt": motion_data.get("prompt", ""),
                    "latency_ms": round(motion_data.get("latency_seconds", 0) * 1000),
                    # このモーションを合わせる発話。フロントエンドは生成完了時刻ではなく
                    # この発話の再生開始時刻を基準にモーションを開始する。
                    "segment_id": motion_data.get("segment_index", 0),
                    # 文中のどのあたりで動くか(0.0=文頭, 1.0=文末)。
                    # 実際の秒数は発話長を知っているフロントエンドが掛け算で出す。
                    "start_offset_ratio": motion_data.get("start_offset_ratio", 0.0),
                    "start_offset_ms": 0,
                }
                if layered_info:
                    payload["layered"] = layered_info
            return "data: " + json.dumps(payload) + "\n\n"

        sentence_buffer = ""
        full_response = ""  # アシスタント応答全文を蓄積
        # 自動補完した動作の通し番号。同じ動作の反復に見えないよう順に回す。
        auto_gesture_count = 0
        # 読み上げる本文が残らなかった文から回収した行動タグ。
        # (例: 文が [act:wave] だけだった場合) 次の発話へ持ち越す。
        carried_tags: list = []

        # 音声駆動バックエンド(SentiAvatar)は発話音声が要るため、
        # 「モーションを TTS より先に投げる」流れが使えない。
        # その場合は TTS 完了後に、発話 1 本につき 1 モーションを生成する。
        needs_audio = motion_backend_requires_audio()

        # 層構成方式: 意味の層(事前作成クリップ)+ リズムの層(SentiAvatar)+ 待機の層。
        # 座位中は上半身だけの合成に未対応なので、そのターンは現行方式で動かす。
        layered_turn = None
        layered_reason = None
        if motion_mode != "current":
            if avatar_posture != "standing":
                layered_reason = "座っている間は現行方式で動かします"
            elif not layered_available():
                layered_reason = "動作クリップが配置されていません"
            else:
                try:
                    layered_turn = LayeredTurn(
                        motion_channel,
                        f"{user_id}:{session_id}",
                        rhythm=motion_mode == "layered",
                    )
                except Exception as error:  # noqa: BLE001
                    layered_reason = f"層構成の準備に失敗しました: {error}"
                    print(f"[layered] init failed: {error}")
        motion_channel.layered = layered_turn is not None
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "motion_mode",
                    "requested": motion_mode,
                    "effective": layered_turn.mode if layered_turn else "current",
                    "reason": layered_reason,
                }
            )
            + "\n\n"
        )
        if layered_turn is not None:
            acknowledgement = await asyncio.to_thread(layered_turn.acknowledge)
            if acknowledgement:
                yield serialize_motion_result(acknowledgement)

        # 直接の動作指示が事前作成クリップで表せるなら、最初の発話に重ねる
        explicit_layered = None
        if explicit_motion is not None and layered_turn is not None:
            resolved = resolve_layered_request(explicit_motion.prompt)
            if resolved is not None:
                explicit_layered = PlannedTag(
                    "explicit",
                    resolved[0],
                    0.0,
                    explicit=True,
                    mirrored=resolved[1],
                )

        def layered_tags_for(planned_tags: list) -> list:
            nonlocal explicit_layered
            tags = [
                PlannedTag(
                    t.name, t.catalog_id, t.position, auto=t.name.startswith("auto:")
                )
                for t in planned_tags
            ]
            for prompt, catalog_id in motion_channel.drain_actions():
                resolved = resolve_layered_request(prompt) or (
                    (catalog_id, False) if catalog_id else None
                )
                if resolved is not None:
                    tags.append(
                        PlannedTag("tool", resolved[0], 0.15, mirrored=resolved[1])
                    )
            if explicit_layered is not None:
                tags.append(explicit_layered)
                explicit_layered = None
            return tags

        if explicit_motion is not None and explicit_layered is not None:
            print(
                f"[motion_request] explicit action (layered clip): {explicit_motion.prompt!r}"
            )
        elif explicit_motion is not None:
            if needs_audio:
                motion_channel.request_action(
                    explicit_motion.prompt, explicit_motion.catalog_id
                )
            else:
                enqueue_motion_job(
                    motion_channel,
                    explicit_motion.prompt,
                    1,
                    catalog_id=explicit_motion.catalog_id,
                    position=0.0,
                )
            print(f"[motion_request] explicit action: {explicit_motion.prompt!r}")
        first_text_logged = False
        first_audio_logged = False

        # エージェントがツール実行中でもSSEを無通信にしない。
        # 10秒ごとのコメントはEventSourceには表示されず、Nginxの504を防ぐ。
        response_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        async def produce_response():
            try:
                async for response_chunk in conversation_orchestrator.stream_response(
                    user_id=user_id,
                    text=user_input,
                    history=history,
                    mode=mode,
                ):
                    await response_queue.put(("chunk", response_chunk))
            except Exception as exc:
                await response_queue.put(("error", exc))
            finally:
                await response_queue.put(("done", None))

        producer_task = asyncio.create_task(produce_response())
        last_keepalive = time.monotonic()

        while True:
            for motion_data in motion_channel.drain():
                yield serialize_motion_result(motion_data)

            try:
                response_type, payload = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=0.5,
                )
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    producer_task.cancel()
                    break
                if time.monotonic() - last_keepalive >= 10:
                    yield ": keep-alive\n\n"
                    last_keepalive = time.monotonic()
                continue

            if response_type == "done":
                break
            if response_type == "error":
                # Provider errors may contain credentials or request data. Expose only the next action.
                print(
                    f"[chat] {selected_model.provider} inference failed: {type(payload).__name__}"
                )
                message = (
                    "ローカルLLMに接続できないか、モデルが応答できません。接続先とモデルIDを確認してください。"
                    if selected_model.provider == "local"
                    else "OpenAIの応答を取得できません。APIキーとモデルIDを確認してください。"
                )
                yield "data: " + json.dumps(
                    {"type": "error", "message": message}
                ) + "\n\n"
                return

            chunk = payload
            raw_piece = str(chunk)
            sentence_buffer += raw_piece

            # --- 初回テキストチャンクの計測 ---
            if not first_text_logged:
                ttft = (time.perf_counter() - t0) * 1000
                print(f"[TIMING] TTFT (Time To First Text): {ttft:.0f}ms")
                first_text_logged = True

            # 受理済み動作の内部ツール記法が本文に出た場合は読み上げない。
            # JSON内の句点で文を切らず、ブロックが閉じるまで元バッファを保持する。
            if explicit_motion:
                clean_buffer, incomplete_motion_tool = clean_motion_tool_artifacts(
                    sentence_buffer
                )
                if incomplete_motion_tool:
                    continue
                sentence_buffer = clean_buffer

            # 行動タグがチャンク境界で割れている間は文を確定しない。
            # `…です。[act:` の状態で確定すると、タグが本文として画面・TTS へ漏れ、
            # モーションも検出できなくなる。
            if has_incomplete_tag(sentence_buffer):
                continue

            # 文の終端を検出したら、完成した文をサニタイズして出力
            if any(p in raw_piece for p in PUNCTUATIONS):
                raw_sentence_for_metadata = sentence_buffer.strip()

                # --- URLを抽出してメタデータとして送信 ---
                # "www." または "http(s)://" から始まり、改行などで途切れるまでを抽出
                _url_regex = re.compile(
                    r"(https?://[^\s()（）「」]+|www\.[^\s()（）「」]+)"
                )
                urls = _url_regex.findall(raw_sentence_for_metadata)
                if urls:
                    for url in urls:
                        # 句読点などが末尾についていれば除去
                        clean_url = re.sub(r"[。！？!?()（）「」]+$", "", url)
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "metadata_chunk",
                                    "dataType": "url",
                                    "content": clean_url,
                                }
                            )
                            + "\n\n"
                        )

                # 応答本文に埋め込まれた行動タグを取り出し、本文からは取り除く。
                # タグは画面にも TTS にも出さない。
                raw_sentence_for_metadata, action_tags = extract_action_tags(
                    raw_sentence_for_metadata
                )

                # 文全体に対してサニタイズ（URL・出典を確実に除去）
                current_sentence = _sanitize_for_speech(raw_sentence_for_metadata)
                # 空白だけの文をそのまま TTS に渡すと ValueError で
                # ストリーム全体が落ちるため、strip して判定する。
                if not current_sentence.strip():
                    # 本文が消えても行動タグは捨てず、次の発話へ持ち越す
                    carried_tags.extend(action_tags)
                    sentence_buffer = ""
                    continue

                # 長い発話でタグが足りない分は機械的に補う。
                # モデルの遵守に依存すると、11秒や13秒の文が丸ごと無動作になる。
                rhythm_layer = layered_turn is not None and layered_turn.rhythm
                auto_tags = (
                    []
                    if explicit_motion or rhythm_layer
                    else plan_auto_gestures(
                        current_sentence, action_tags, auto_gesture_count
                    )
                )
                auto_gesture_count += len(auto_tags)

                # この文に紐づくモーションを、TTS を待たずに走らせる。
                # T2M は 1 秒前後かかるので、少しでも早く投げておきたい。
                planned = (
                    [] if explicit_motion else carried_tags + action_tags + auto_tags
                )
                carried_tags = []
                next_segment = motion_channel.segment_index + 1
                if planned and not needs_audio and layered_turn is None:
                    for tag in planned:
                        enqueue_motion_job(
                            motion_channel,
                            tag.prompt,
                            next_segment,
                            catalog_id=tag.catalog_id,
                            position=tag.position,
                        )
                    print(
                        f"[motion_tags] segment={next_segment} "
                        f"tags={[t.name for t in planned]}"
                    )

                full_response += current_sentence

                # テキストを送信
                yield (
                    "data: "
                    + json.dumps({"type": "text_chunk", "content": current_sentence})
                    + "\n\n"
                )

                # 並列で音声生成
                try:
                    audio_b64, duration_ms = await asyncio.to_thread(
                        _tts.synthesize_with_duration,
                        current_sentence,
                    )
                except IrodoriTTSError:
                    producer_task.cancel()
                    raise

                # この発話に採番する。モーションはこのIDを指して再生位置を決める。
                segment_id = motion_channel.advance_segment()

                # 層構成: 発話の長さが決まったので 1 本に合成し、音声より先に送る
                if layered_turn is not None:
                    layered_payload = await asyncio.to_thread(
                        layered_turn.on_sentence,
                        segment_id,
                        duration_ms,
                        layered_tags_for(planned),
                        base64.b64decode(audio_b64) if layered_turn.rhythm else None,
                    )
                    yield serialize_motion_result(layered_payload)

                # 音声駆動バックエンドはここで初めて呼べる。
                # 生成されるモーションは発話全体を覆うので、この発話につき 1 本だけ。
                if needs_audio and layered_turn is None:
                    # タグ/自動補完に加え、ツールから預かった要求も候補に入れる
                    requested = motion_channel.drain_actions()
                    candidates = [(t.prompt, t.catalog_id, t.name) for t in planned]
                    candidates += [(p, c, "tool") for p, c in requested]
                    if candidates:
                        prompt, catalog_id, source = candidates[0]
                        enqueue_motion_job(
                            motion_channel,
                            prompt,
                            segment_id,
                            catalog_id=catalog_id,
                            position=0.0,
                            audio_wav=base64.b64decode(audio_b64),
                        )
                        print(
                            f"[motion_tags] segment={segment_id} audio-driven "
                            f"lead={source} (候補 {[c[2] for c in candidates]})"
                        )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "audio_chunk",
                            "sentence": current_sentence,
                            "audio": audio_b64,
                            "segment_id": segment_id,
                            "duration_ms": round(duration_ms),
                        }
                    )
                    + "\n\n"
                )

                # --- 初回音声チャンクの計測 ---
                if not first_audio_logged:
                    ttfa = (time.perf_counter() - t0) * 1000
                    print(f"[TIMING] TTFA (Time To First Audio): {ttfa:.0f}ms")
                    first_audio_logged = True

                sentence_buffer = ""

        if not producer_task.done():
            producer_task.cancel()

        # 最後に残った文を処理
        if explicit_motion:
            sentence_buffer, _ = clean_motion_tool_artifacts(
                sentence_buffer, final=True
            )
            # 音声時計がないと動作キューを開始できないため、内部記法しか
            # 残らなかった返答にも短い受理の一言を配信する。
            readable_buffer, _ = extract_action_tags(sentence_buffer)
            if (
                not full_response.strip()
                and not _sanitize_for_speech(readable_buffer).strip()
            ):
                sentence_buffer = "承知しました。"

        if sentence_buffer.strip():
            raw_sentence_for_metadata = sentence_buffer.strip()

            # --- URLを抽出してメタデータとして送信 ---
            _url_regex = re.compile(
                r"(https?://[^\s()（）「」]+|www\.[^\s()（）「」]+)"
            )
            urls = _url_regex.findall(raw_sentence_for_metadata)
            if urls:
                for url in urls:
                    clean_url = re.sub(r"[。！？!?()（）「」]+$", "", url)
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "metadata_chunk",
                                "dataType": "url",
                                "content": clean_url,
                            }
                        )
                        + "\n\n"
                    )

            raw_sentence_for_metadata, action_tags = extract_action_tags(
                raw_sentence_for_metadata
            )
            current_sentence = _sanitize_for_speech(raw_sentence_for_metadata)
            if current_sentence.strip():
                rhythm_layer = layered_turn is not None and layered_turn.rhythm
                auto_tags = (
                    []
                    if explicit_motion or rhythm_layer
                    else plan_auto_gestures(
                        current_sentence, action_tags, auto_gesture_count
                    )
                )
                auto_gesture_count += len(auto_tags)
                planned = (
                    [] if explicit_motion else carried_tags + action_tags + auto_tags
                )
                carried_tags = []
                if planned and not needs_audio and layered_turn is None:
                    next_segment = motion_channel.segment_index + 1
                    for tag in planned:
                        enqueue_motion_job(
                            motion_channel,
                            tag.prompt,
                            next_segment,
                            catalog_id=tag.catalog_id,
                            position=tag.position,
                        )
                full_response += current_sentence
                yield (
                    "data: "
                    + json.dumps({"type": "text_chunk", "content": current_sentence})
                    + "\n\n"
                )
                audio_b64, duration_ms = await asyncio.to_thread(
                    _tts.synthesize_with_duration,
                    current_sentence,
                )
                segment_id = motion_channel.advance_segment()
                if layered_turn is not None:
                    layered_payload = await asyncio.to_thread(
                        layered_turn.on_sentence,
                        segment_id,
                        duration_ms,
                        layered_tags_for(planned),
                        base64.b64decode(audio_b64) if layered_turn.rhythm else None,
                    )
                    yield serialize_motion_result(layered_payload)
                if needs_audio and layered_turn is None:
                    requested = motion_channel.drain_actions()
                    candidates = [(t.prompt, t.catalog_id) for t in planned]
                    candidates += list(requested)
                    if candidates:
                        prompt, catalog_id = candidates[0]
                        enqueue_motion_job(
                            motion_channel,
                            prompt,
                            segment_id,
                            catalog_id=catalog_id,
                            position=0.0,
                            audio_wav=base64.b64decode(audio_b64),
                        )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "audio_chunk",
                            "sentence": current_sentence,
                            "audio": audio_b64,
                            "segment_id": segment_id,
                            "duration_ms": round(duration_ms),
                        }
                    )
                    + "\n\n"
                )

        # --- アシスタント応答の保存 ---
        if history_service and full_response.strip():
            try:
                history_service.save_message(
                    user_id=user_id,
                    session_id=session_id,
                    role="assistant",
                    content=full_response.strip(),
                )
            except Exception as e:
                print(f"[WARNING] アシスタント応答保存に失敗: {e}")

        # --- タイミング情報の送信 ---
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"[TIMING] Total response time: {total_ms:.0f}ms")
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "timing",
                    "total_ms": round(total_ms),
                }
            )
            + "\n\n"
        )

        # 会話と音声は先に配信済み。T2Mジョブだけが残っている場合は、
        # SSEを維持しながら完成結果を非同期に回収する。
        motion_deadline = time.monotonic() + MOTION_JOB_WAIT_TIMEOUT
        while not motion_channel.is_idle and time.monotonic() < motion_deadline:
            if await request.is_disconnected():
                break

            motion_data = await asyncio.to_thread(motion_channel.get, 1.0)
            if motion_data is not None:
                yield serialize_motion_result(motion_data)
                continue

            if time.monotonic() - last_keepalive >= 10:
                yield ": keep-alive\n\n"
                last_keepalive = time.monotonic()

        for motion_data in motion_channel.drain():
            yield serialize_motion_result(motion_data)

        if not motion_channel.is_idle:
            yield serialize_motion_result(
                {
                    "status": "error",
                    "prompt": "",
                    "message": "モーション生成の待機時間を超過しました",
                }
            )

        yield ("data: " + json.dumps({"type": "done", "message": "応答完了"}) + "\n\n")

    # 途中のプロキシに圧縮・蓄積させない。後から届く動作(リズムの層の差し替え)が
    # 圧縮バッファに溜まり、数秒遅れて届くのを防ぐ。
    return StreamingResponse(
        _stream_tts_errors(event_stream()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.post("/reset")
async def reset_history(user_id: str = "default", session_id: str = DEFAULT_SESSION_ID):
    """
    指定されたユーザーおよびセッションの対話履歴をリセット（削除）する。
    """
    if history_service:
        try:
            history_service.clear_session(user_id=user_id, session_id=session_id)
            return {"status": "success", "message": "History reset successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to reset history: {e}"}
    return {"status": "ignored", "message": "HistoryService is not available"}
