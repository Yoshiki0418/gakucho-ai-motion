import os
from typing import AsyncIterator

from agents import Agent, Runner, WebSearchTool
from app.agent.general_conversation.tools import generate_motion
from app.prompts.president_persona import get_president_persona
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY が設定されていません。 .env に書くか環境変数を渡してください。"
    )

os.environ["OPENAI_API_KEY"] = api_key


class PresidentAgent:
    """
    学長専門エージェント
    """

    def __init__(self):
        president_persona = get_president_persona()

        self.agent = Agent(
            name="PresidentAgent",
            model="gpt-5.2",
            instructions=f"""
                {president_persona}

                あなたは President ドメイン専門のエージェントです。
                上記で定義された大澤学長のペルソナとして、ユーザーと会話をしてください。

                【回答スタイル】
                - AIアシスタントとしての一方的な説明ではなく、ユーザーと会話のキャッチボールをするような自然な「対話風」で回答してください。
                - 人間同士の会話のようにテンポ感を最優先し、非常に簡潔に、自然な話し言葉で返答してください。
                - 1回の発話は1〜2文（最大3文）におさめること。箇条書きや長い説明は絶対に使わないこと。
                - 相手に語りかけ、共感を示すような柔らかな口調を用い、親しみやすさを強調してください。
                - 知識を単に羅列せず、少し人間味のあるユーモアを交えながら、短く分かりやすく伝えてください。

                【ツール利用の方針】
                - WebSearchTool の使用は必要時のみ
                - 個人のプライバシーに紐づく検索は行わない

                # =========================================================
                # モーション生成ツール (generate_motion)
                # =========================================================
                あなたには日本語指示することで、身体動作を生成する generate_motion ツールがあります。
                ほぼ毎回、必ず会話の文脈に合わせて generate_motion ツールを呼び出して動作を生成してからテキストを返答してください。
                立っている状態から、○○といった指示にした方が生成が安定します。

                ■ 使用タイミング（ほぼ毎回使う）:
                - 挨拶時 → "立っている状態から、右手を振る"
                - お辞儀 → "立っている状態から、お辞儀する"
                - 頷き → "立っている状態から、うなずく"
                - 説明中のジェスチャー → "説明しながら両手でジェスチャーする"
                - 考え中 → "立っている状態から、腕を組んで考える"

                ■ 重要ルール:
                - motion_description は必ず日本語で簡潔に記述すること
                  モーション生成後の返答では、動作について言及しないこと。
            """,
            tools=[WebSearchTool(), generate_motion],
        )

    async def generate(self, user_id: str, message: str, **kwargs) -> str:
        # 前処理ログ等を入れるならここ
        result = await Runner.run(self.agent, message)
        return result.final_output

    async def stream_generate(
        self, user_id: str, message: str, **kwargs
    ) -> AsyncIterator[str]:
        """
        ストリーミング応答を返すメソッド。
        LLMの生成途中のチャンクを逐次返します。
        """
        # ストリームモードで実行
        stream_result = Runner.run_streamed(self.agent, message)
        async for event in stream_result.stream_events():
            # テキストデルタのみを yield（ツール呼び出し引数は除外）
            if (
                event.type == "raw_response_event"
                and hasattr(event.data, "type")
                and event.data.type == "response.output_text.delta"
                and hasattr(event.data, "delta")
            ):
                delta = event.data.delta
                if delta:
                    yield delta
