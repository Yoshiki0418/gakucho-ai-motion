import os
from functools import lru_cache

from app.agent.general_conversation.agent import GeneralConversationAgent
from app.agent.general_conversation.domains import (
    LifePlanningAgent,
    LocationAgent,
    PresidentAgent,
    ResearchAgent,
)
from app.agent.modules.decision_module import LLMDecisionClassifier
from app.agent.modules.rag_module import RAGModule
from app.agent.response_orchestrator import ResponseOrchestrator
from app.agent.local_conversation import LocalConversationAgent
from app.chat_models import ChatModel, select_model
from app.models.model_registry import create_llm
from app.rag.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()

from app.prompts.president_persona import get_president_persona


@lru_cache(maxsize=1)
def _shared_retriever():
    # The embedding model and database connection are independent of the chat model.
    return Retriever(db_url=os.getenv("DB_URL"))


@lru_cache(maxsize=8)
def build_conversation_orchestrator(selected: ChatModel | None = None):
    selected = selected or select_model()
    president_persona = get_president_persona()
    if selected.provider == "local":
        gc_agent = LocalConversationAgent(selected.name, president_persona)
    else:
        domains = [ResearchAgent().agent, LifePlanningAgent().agent, LocationAgent().agent, PresidentAgent().agent]
        for domain in domains:
            domain.model = selected.name
        gc_agent = GeneralConversationAgent(*domains)
        gc_agent.general_agent.model = gc_agent.ceremony_agent.model = selected.name

    classifier_llm = create_llm(
        selected.provider,
        model_name=selected.name,
        system_prompt="あなたは分類器です。ユーザー発話を候補ラベルから1つ選び、JSONのみ返してください。",
    )
    classifier = LLMDecisionClassifier(
        llm=classifier_llm,
    )

    # --- Filler LLM ---
    filler_llm = create_llm(
        selected.provider,
        model_name=selected.name,
        system_prompt=f"""{president_persona}
            あなたはフィラー（つなぎ会話）専用です。
            ユーザーの発話内容をふまえ、自然な一言を1文だけ返してください。

            ■ 調査・検索を伴う質問（大学の制度、施設、研究、特定の歴史など）の場合は絶対に空文字を返さず、必ずフィラーを生成してください。
            ■ 不要なら空文字を返す（挨拶のみ/即答可能な短い質問/相槌などは空文字でよい）
            ■ 禁止: 結論・事実・提案・箇条書き・長文・「少々お待ちください」などの機械的な定型文
            ■ 丁寧語。毎回違う表現を使い、同じフレーズを繰り返さない。

            ■ フィラーの役割:
            ユーザーの発話を受け止め、「今から考えますよ」「調べてみますね」というニュアンスを、その場限りの自然な言葉で伝えること。

            ■ フィラーの方向性（1つ選択、具体的な言い回しは自由に考えること）
            A) 受け止め — ユーザーの発話内容に触れて軽く共感や理解を示す
            B) 思考の予告 — これから考えて答える旨を、発話内容に絡めて自然に伝える
            C) 確認の予告 — 調査や検索が必要なとき、何について調べるかを具体的に一言添える

            ■ 重要: 定型文ではなく、相手の文脈に沿ったオリジナルな一言を返すこと。
            ■ 出力: 不要→空文字 / 必要→日本語1文のみ""",
    )

    # --- Retriever / RAG Module ---
    rag_engine = RAGModule(
        retriever=_shared_retriever(),
        llm=create_llm(
            selected.provider,
            model_name=selected.name,
            system_prompt=f"{president_persona}\n\nあなたはシステム内部で情報検索と要約を担当するRAGモジュールです。与えられたコンテキストに基づいて、回答を生成してください。",
        ),
    )

    # --- Orchestrator ---
    orchestrator = ResponseOrchestrator(
        rag_engine=rag_engine,
        daily_agent=gc_agent,
        classifier_llm=classifier,
        filler_llm=filler_llm,
        filler_timeout=0.5,
        use_external_filler=False,  # ★実験中★ 外部フィラーを止め、親エージェント自身の自己判断フィラーに切り替え
    )

    return orchestrator
