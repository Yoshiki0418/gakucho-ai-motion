"""Speech and action-tag turns for local servers supporting Chat Completions."""

from typing import AsyncIterator

from app.agent.general_conversation.tools.motion_tags import tag_reference_for_prompt
from app.agent.general_conversation.tools.motion_tool import motion_data_queue_var
from app.models.llm import LocalLLM


class LocalConversationAgent:
    def __init__(self, model: str, persona: str):
        self.model = model
        self.persona = persona

    async def stream_generate(
        self,
        user_id: str,
        message: str,
        history: list[dict] | None = None,
        mode: str = "general",
    ) -> AsyncIterator[str]:
        style = (
            "式典で聴衆に向けた丁寧な挨拶"
            if mode == "ceremony"
            else "相手の言葉に応じた簡潔で自然な対話"
        )
        channel = motion_data_queue_var.get()
        if channel and channel.explicit_prompt:
            gestures = "ユーザーが頼んだ動作は受理済みです。行動タグは追加せず、質問に答えてください。"
        else:
            gestures = (
                "身振りを合わせたい語の直前に次のタグを置いてください。"
                "例：本学の特徴は[act:point]実践的な教育にあります。\n"
                + tag_reference_for_prompt()
            )
        llm = LocalLLM(
            model_name=self.model,
            system_prompt=f"{self.persona}\n{style}として日本語で話してください。\n{gestures}",
        )
        async for part in llm.stream_generate(message, history, max_tokens=512):
            yield part
