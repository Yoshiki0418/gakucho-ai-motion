"""Read server-owned connection settings, and resolve a model for one chat turn."""

import os
from dataclasses import dataclass
from functools import lru_cache

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ChatModel:
    provider: str
    name: str


def model_settings() -> dict:
    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return {
        "default_provider": "openai" if has_key else "local",
        "openai_available": has_key,
        "openai_model": os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL_NAME")
        or "gpt-5.2",
        "local_model": os.getenv("LOCAL_LLM_MODEL") or "qwen2.5:7b",
    }


def select_model(provider: str | None = None, name: str | None = None) -> ChatModel:
    settings = model_settings()
    provider = provider or settings["default_provider"]
    if provider not in ("local", "openai"):
        raise ValueError("LLMの接続先は openai または local を指定してください。")
    if provider == "openai" and not settings["openai_available"]:
        raise RuntimeError(
            "OpenAI APIキーが未設定です。ローカルLLMを選択してください。"
        )
    name = (name or "").strip() or settings[f"{provider}_model"]
    if len(name) > 256 or not name.isprintable():
        raise ValueError("モデルIDには256文字以内の表示可能な文字を指定してください。")
    return ChatModel(provider, name)


@lru_cache(maxsize=2)
def chat_client(provider: str) -> AsyncOpenAI:
    if provider == "local":
        # Keep OpenAI credentials away from the separately configured inference server.
        return AsyncOpenAI(
            base_url=os.getenv("LOCAL_LLM_BASE_URL") or "http://127.0.0.1:11434/v1",
            api_key=os.getenv("LOCAL_LLM_API_KEY") or "local",
        )
    if provider != "openai":
        raise ValueError("Unknown LLM provider")
    return AsyncOpenAI()
