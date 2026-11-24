import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Tuple

from app.chat_models import select_model
from app.models.llm import LocalLLM, OpenAILLM
from app.models.llm.base_llm import BaseLLM
from app.models.tts import StyleBertVITS2_TTS
from app.models.tts.irodori_tts import IrodoriTTS


def read_prompt_from_file(path: str) -> str:
    """システムプロンプトを外部ファイルから読み込む"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found: {path}")


def create_llm(provider: str, system_prompt: str, model_name: str = None) -> BaseLLM:
    selected = select_model(provider, model_name)
    model_class = LocalLLM if selected.provider == "local" else OpenAILLM
    return model_class(system_prompt=system_prompt, model_name=selected.name)


def create_tts(provider: str, model_dir: str):
    if provider == "style-bert-vits2":
        return StyleBertVITS2_TTS(model_dir=model_dir)
    if provider == "irodori-tts":
        return IrodoriTTS()
    raise ValueError("指定した音声合成プロバイダーは利用できません。")


_style_lock = Lock()


@lru_cache(maxsize=1)
def _installed_style_tts(model_dir: str):
    return create_tts("style-bert-vits2", model_dir)


def _style_dir() -> str:
    return (
        os.getenv("TTS_MODEL_DIR") or "/workspace/backend/app/weights/tts/gakucho_ai_v2"
    )


def default_tts_provider() -> str:
    return os.getenv("TTS_PROVIDER") or "style-bert-vits2"


def tts_settings() -> dict:
    provider = default_tts_provider()
    root = Path(_style_dir())
    style_available = provider == "style-bert-vits2" or (
        (root / "config.json").is_file()
        and (root / "style_vectors.npy").is_file()
        and any(root.glob("G_*.pth"))
    )
    return {
        "default_provider": provider,
        "providers": [
            {
                "id": "style-bert-vits2",
                "label": "Style-Bert-VITS2",
                "available": style_available,
            },
            {
                "id": "irodori-tts",
                "label": "Irodori-TTS",
                "available": IrodoriTTS().is_available(),
            },
        ],
    }


def select_tts(provider: str | None = None, voice: str | None = None):
    if not provider or provider == "default":
        provider = default_tts_provider()
    if provider == "irodori-tts":
        engine = IrodoriTTS()
        if not engine.base_url:
            raise RuntimeError("Irodori-TTSの接続先が設定されていません。")
        installed_voice = engine.voice
    elif provider == "style-bert-vits2":
        installed_voice = Path(_style_dir()).name
    else:
        raise ValueError("指定した音声合成プロバイダーは利用できません。")
    if voice and voice not in ("default", installed_voice):
        raise ValueError("指定した音声モデルは設定されていません。")
    if provider == "irodori-tts":
        return engine
    # Multiple first-time requests must not load the same GPU weights in parallel.
    with _style_lock:
        try:
            return _installed_style_tts(_style_dir())
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Style-Bert-VITS2の学習済みモデルが見つかりません。"
            ) from exc


def load_models_from_env() -> Tuple[object, object]:
    selected = select_model()

    prompt_path = os.getenv(
        "LLM_SYSTEM_PROMPT_PATH", "/workspace/backend/app/prompts/system_prompt.txt"
    )
    system_prompt = read_prompt_from_file(prompt_path)

    llm = create_llm(selected.provider, system_prompt, selected.name)
    tts = select_tts()
    return llm, tts
