"""G7-3: the conversation brain (the cloud LLM of the production 学長 agent) hands house instructions to the executor.

    .runtime/world-model/brain-venv/bin/python -m world_model.house_brain "冷蔵庫の前まで来てくれる？"

The brain is the same OpenAI Agents SDK agent shape as the backend (model gpt-5.2) with one tool,
``house_task``: an instruction to move in the house goes to the executor service
(``house_executor_server``, 127.0.0.1:4013) as the user said it, not decomposed; the tool waits for the
episode and returns its result, and the brain answers the user from it. Nothing is trained.
The API key is read from backend/.env (OPENAI_API_KEY) and never printed.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = os.environ.get("HOUSE_EXECUTOR_URL", "http://127.0.0.1:4013")
MODEL = os.environ.get("HOUSE_BRAIN_MODEL", "gpt-5.2")
INSTRUCTIONS = """あなたは学長アバターの対話を担当します。学長の体は家（2階建て）の1階にいて、家の中を歩けます。
ユーザーが家の中での移動（「〜に行って」「〜の前まで来て」など）を頼んだら、言い換えたり分けたりせず、
ユーザーの言った指示をそのまま house_task に渡してください。体を動かすのは実行係で、見て判断しながら動きます。
結果が返ったら、着いたならそう伝え、着けなかったりできなかったりしたら理由を短く伝えてください。
移動の依頼でない話には普通に答えてください。答えは日本語で、短く話し言葉で。"""


def _api_key():
    for line in (ROOT / "backend/.env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("backend/.env に OPENAI_API_KEY がない")


def _call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(EXECUTOR + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def build_agent():
    from agents import Agent, function_tool

    @function_tool
    async def house_task(instruction: str) -> str:
        """家の中で体を動かす指示を実行係に渡し、終わるまで待って結果を返す。

        Args:
            instruction: ユーザーが言った移動の指示（言い換えない）。例：「冷蔵庫の前まで来て」
        """
        started = _call("POST", "/runs", {"instruction": instruction})
        deadline = time.time() + 1200
        while time.time() < deadline:
            await asyncio.sleep(5)
            run = _call("GET", f"/runs/{started['id']}")
            if run["status"] in ("done", "error"):
                return json.dumps({"id": started["id"], "status": run["status"], "result": run.get("result"),
                                   "error": (run.get("error") or "")[:300]}, ensure_ascii=False)
        return json.dumps({"id": started["id"], "status": "timeout"}, ensure_ascii=False)

    return Agent(name="学長（家）", instructions=INSTRUCTIONS, model=MODEL, tools=[house_task])


async def main(text):
    os.environ.setdefault("OPENAI_API_KEY", _api_key())
    from agents import Runner
    result = await Runner.run(build_agent(), text)
    calls = [item for item in result.new_items if item.type == "tool_call_item"]
    for item in calls:
        print("実行係へ:", getattr(item.raw_item, "arguments", ""))
    print("学長:", result.final_output)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "冷蔵庫の前まで来てくれる？"))
