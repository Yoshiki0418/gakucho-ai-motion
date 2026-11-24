"""Synthesize speech through a server-owned Irodori-TTS OpenAI-compatible API."""

import base64
import io
import os

import httpx
import soundfile as sf

from .base_tts import BaseTTS


class IrodoriTTSError(RuntimeError):
    """A user-facing error, safe to return without upstream response content."""


class IrodoriTTS(BaseTTS):
    _model_name = "irodori-tts"

    def __init__(self):
        self.base_url = os.getenv("IRODORI_TTS_BASE_URL", "").strip().rstrip("/")
        self.model = os.getenv("IRODORI_TTS_MODEL") or self._model_name
        self.voice = os.getenv("IRODORI_TTS_VOICE") or "none"
        self.api_key = os.getenv("IRODORI_TTS_API_KEY", "").strip()

    def _client(self, timeout: float) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            trust_env=False,
        )

    def is_available(self) -> bool:
        if not self.base_url:
            return False
        try:
            with self._client(timeout=1.0) as client:
                response = client.get("models")
                response.raise_for_status()
                return any(item["id"] == self.model for item in response.json()["data"])
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError, KeyError):
            return False

    def synthesize_with_duration(self, text: str) -> tuple[str, float]:
        try:
            with self._client(timeout=120.0) as client:
                response = client.post(
                    "audio/speech",
                    json={
                        "model": self.model,
                        "input": text,
                        "voice": self.voice,
                        "response_format": "wav",
                    },
                )
                response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise IrodoriTTSError(
                "Irodori-TTSで音声を生成できません。接続先・モデル・音声の設定を確認してください。"
            ) from exc

        try:
            with sf.SoundFile(io.BytesIO(response.content)) as audio:
                if audio.format not in ("WAV", "WAVEX") or audio.frames <= 0:
                    raise ValueError("expected a non-empty WAV")
                duration_ms = 1000 * audio.frames / audio.samplerate
        except (RuntimeError, ValueError) as exc:
            raise IrodoriTTSError(
                "Irodori-TTSから有効なWAV音声を取得できません。"
            ) from exc
        return base64.b64encode(response.content).decode("ascii"), duration_ms

    def synthesize_to_base64(self, text: str) -> str:
        audio, _ = self.synthesize_with_duration(text)
        return audio
