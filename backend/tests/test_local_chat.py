"""Exercise the real SSE route with an offline Chat Completions transport.

Run in the backend environment: python -m unittest discover -s tests -p test_local_chat.py
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch


def _exercise_route(prompt_path):
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from openai import AsyncOpenAI

    os.environ.update(
        OPENAI_API_KEY="",
        LOCAL_LLM_MODEL="test-local",
        LOCAL_LLM_BASE_URL="http://local-model.invalid/v1",
        LOCAL_LLM_API_KEY="offline",
        LLM_SYSTEM_PROMPT_PATH=prompt_path,
        T2M_PRELOAD_MODEL="false",
        MOTION_BACKEND="custom-t2m",
        TTS_PROVIDER="style-bert-vits2",
        IRODORI_TTS_BASE_URL="",
    )

    # The production router initializes speech, history, and persona modules on import.
    def substitute(name, **symbols):
        module = types.ModuleType(name)
        module.__dict__.update(symbols)
        sys.modules[name] = module

    speech = Mock()
    speech.synthesize_with_duration.return_value = ("YQ==", 1000)
    substitute(
        "app.models.tts.style_bert_vits2_tts",
        StyleBertVITS2_TTS=Mock(return_value=speech),
    )
    substitute("app.rag.retriever", Retriever=Mock())
    substitute("app.services.history_service", HistoryService=Mock(return_value=None))
    substitute("app.prompts.president_persona", get_president_persona=lambda: "学長")

    from app.routers.text_chat_router import router

    models = []

    def completion(request):
        assert str(request.url) == "http://local-model.invalid/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer offline"
        body = json.loads(request.content)
        models.append(body["model"])
        chunk = {
            "id": "test",
            "created": 1,
            "object": "chat.completion.chunk",
            "model": body["model"],
            "choices": [
                {"index": 0, "delta": {"content": "では[act:wave]こんにちは。"}}
            ],
        }
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    offline_http = httpx.AsyncClient(transport=httpx.MockTransport(completion))
    app = FastAPI()
    app.include_router(router)
    with patch(
        "app.chat_models.AsyncOpenAI",
        lambda **kw: AsyncOpenAI(http_client=offline_http, **kw),
    ):
        with TestClient(app) as client:
            for mode in ("general", "ceremony"):
                params = {
                    "mode": mode,
                    "session_id": mode,
                    "text": "こんにちは",
                    "motion_mode": "layered-semantic",
                }
                if mode == "ceremony":
                    params.update(llm_model="other-local", llm_provider="local")
                response = client.get(
                    "/api/text-chat/char-stream-orchestrator", params=params
                )
                response.raise_for_status()
                events = [
                    json.loads(s[6:])
                    for s in response.text.splitlines()
                    if s.startswith("data: ")
                ]
                assert events[-1]["type"] == "done"
                assert any(e["type"] == "audio_chunk" for e in events)
                assert any(e.get("content") == "ではこんにちは。" for e in events)
                assert any(
                    overlay["tag"] == "wave"
                    for e in events
                    if e["type"] == "motion_chunk"
                    for overlay in e.get("layered", {}).get("overlays", [])
                )
            assert models == ["test-local", "other-local"]
            _exercise_tts_selection(client, speech)


def _exercise_tts_selection(client, default_speech):
    import httpx

    route = "/api/text-chat/char-stream-orchestrator"
    settings = "/api/text-chat/model-settings"
    params = {"text": "こんにちは", "motion_mode": "layered-semantic"}
    alternate = {**params, "tts_provider": "irodori-tts"}
    assert client.get(route, params=alternate).status_code == 503
    assert (
        client.get(route, params={**params, "tts_provider": "unknown"}).status_code
        == 400
    )
    assert client.get(route, params={**params, "tts_voice": "other"}).status_code == 400

    def available():
        catalog = client.get(settings).json()["tts"]
        assert catalog["default_provider"] == "style-bert-vits2"
        return {item["id"]: item["available"] for item in catalog["providers"]}

    assert available() == {"style-bert-vits2": True, "irodori-tts": False}

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", ""))
        audio.writeframes(bytes(3200))
    wav = buffer.getvalue()
    fail = False

    def speech_server(request):
        assert request.headers["Authorization"] == "Bearer test-speech-only"
        if str(request.url) == "http://speech.invalid/v1/models":
            return httpx.Response(200, json={"data": [{"id": "irodori-test"}]})
        assert str(request.url) == "http://speech.invalid/v1/audio/speech"
        assert json.loads(request.content) == {
            "model": "irodori-test",
            "input": "ではこんにちは。",
            "voice": "test-voice",
            "response_format": "wav",
        }
        return httpx.Response(200, content=b"invalid" if fail else wav)

    def events(query):
        response = client.get(route, params=query)
        response.raise_for_status()
        return [
            json.loads(s[6:])
            for s in response.text.splitlines()
            if s.startswith("data: ")
        ]

    with patch.dict(
        os.environ,
        {
            "IRODORI_TTS_BASE_URL": "http://speech.invalid/v1",
            "IRODORI_TTS_MODEL": "irodori-test",
            "IRODORI_TTS_VOICE": "test-voice",
            "IRODORI_TTS_API_KEY": "test-speech-only",
        },
    ), patch.object(httpx.HTTPTransport, "handle_request", side_effect=speech_server):
        assert all(available().values())
        with patch.dict(os.environ, IRODORI_TTS_MODEL="unlisted"):
            assert not available()["irodori-tts"]
        old_calls = default_speech.synthesize_with_duration.call_count
        received = events(alternate)
        audio = next(e for e in received if e["type"] == "audio_chunk")
        assert base64.b64decode(audio["audio"]) == wav and audio["duration_ms"] == 100
        assert received[-1]["type"] == "done"
        assert default_speech.synthesize_with_duration.call_count == old_calls

        fail = True
        failed = events(alternate)
        assert failed[-1]["type"] == "error" and "Irodori-TTS" in failed[-1]["message"]
        assert all(e["type"] != "audio_chunk" for e in failed)
        assert default_speech.synthesize_with_duration.call_count == old_calls

        for selection in ({"tts_provider": "style-bert-vits2"}, {}):
            assert events({**params, **selection})[-1]["type"] == "done"
        assert default_speech.synthesize_with_duration.call_count == old_calls + 2


class LocalChatRegressionTest(unittest.TestCase):
    def test_keyless_chat_routes_action_tags_to_motion(self):
        # Isolate import-time application singletons and keep other tests' clients intact.
        with tempfile.TemporaryDirectory() as temp:
            prompt = Path(temp, "prompt.txt")
            prompt.write_text("学長", encoding="utf-8")
            backend = Path(__file__).resolve().parents[1]
            subprocess.run(
                [sys.executable, __file__, str(prompt)],
                check=True,
                timeout=90,
                env={**os.environ, "PYTHONPATH": str(backend)},
                cwd=backend,
            )


if __name__ == "__main__":
    _exercise_route(sys.argv[1])
