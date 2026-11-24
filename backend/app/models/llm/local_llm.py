from .openai_llm import OpenAILLM


class LocalLLM(OpenAILLM):
    """OpenAI-compatible Chat Completions served by the configured local model."""

    provider = "local"

    def __init__(self, model_name: str, system_prompt: str):
        super().__init__(system_prompt=system_prompt, model_name=model_name)
