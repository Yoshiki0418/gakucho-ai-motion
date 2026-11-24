# 対話中の動作生成を開発する

学長AIの会話画面 `/` で、返答を作るLLMと、身振りを作る方式を個別に選べます。手元にOpenAI APIキーがなくても、ローカルのOpenAI互換Chat Completionsサーバーを使えます。モデルの接続先を設定する場所は **FastAPIが動くマシン** です。フロントエンドだけを自分のPCで動かす場合は、管理者からローカルLLMを設定済みの開発用バックエンドを受け取ってください。

## Claude Codeへの依頼文

> この資料を使い、OpenAI APIキーなしで対話中の動作生成を開発できるようにしてください。まず接続先の開発用バックエンドとローカルモデルのIDを確認してください。モデルを固定したまま立位で同じ挨拶を3方式で試し、各ターンの実効方式、動作の出所、音声とのタイミングを報告してください。変更は最新mainからIssue専用ブランチを作り、PRで提出してください。認証情報はサーバー側で管理してください。

## ローカルLLMを接続する

一例としてOllamaの `qwen2.5:7b` を使います。既に動いている互換サーバーがあれば、その `/v1` URLと配布済みのモデルIDを使えます。[Ollamaのローカル互換API](https://docs.ollama.com/api/openai-compatibility)・[モデル情報](https://ollama.com/library/qwen2.5:7b)

**FastAPIとOllamaを同じホストで直接実行する場合**は、Ollamaを起動し（サービスがなければ別ターミナルで `ollama serve`）、以下を実行します。

```bash
ollama pull qwen2.5:7b
curl -fsS http://127.0.0.1:11434/v1/models
```

**Docker Composeで開発する場合**は、専用の開発環境にある `.runtime/compose/local.yml` の `services` に次のサービスを追加します。`volumes` はファイルの最上位に置きます。モデルの推論はこの例ではCPU上で行います。Ollamaのポートをホストへ公開する必要はありません。

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - local_llm_models:/root/.ollama
volumes:
  local_llm_models:
```

プロジェクトのルートで実行します。以下は共同開発者向けセットアップ手順のCPU構成です。GPU構成を既に使用している場合は、普段と同じComposeファイルの組み合わせを使います。

```bash
compose=(docker compose --project-name gakucho-ai-motion
  -f docker-compose.prod.yml -f docker-compose.startup.yml
  -f .runtime/compose/local.yml -f docker-compose.cpu.yml)
"${compose[@]}" up -d ollama
"${compose[@]}" exec ollama ollama pull qwen2.5:7b
```

次の設定を **`backend/.env`** に入れ、同じシェルで `"${compose[@]}" up -d --no-build --force-recreate backend` を実行します。ホスト直接実行のときはURLを `http://127.0.0.1:11434/v1` に置き換えてFastAPIを再起動します。

```dotenv
OPENAI_API_KEY=
LOCAL_LLM_BASE_URL=http://ollama:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b
```

認証付きのローカル推論サーバーでは `LOCAL_LLM_API_KEY` にそのサーバー用の鍵を設定します。OpenAI側のモデルの既定値は `OPENAI_MODEL` で指定できます。UIはモデルIDをリクエストごとに渡すため、他の利用者のモデル選択は独立しています。ローカルでの通常会話と式典モードはChat Completionsを使います。OpenAIのホスト型検索や専門エージェントへの引き継ぎはOpenAIモードの機能です。

開発者にバックエンドイメージを渡す際は、この版の `backend/.dockerignore` で `.env` を除外してビルドしたものを使います。過去のDockerfileはバックエンドのディレクトリ全体をコピーしており、旧イメージに環境ファイルが残っている場合があります。

1. 開発用バックエンドの `GET /api/text-chat/model-settings` で `default_provider` が `local`、`openai_available` が `false`、`local_model` が配備したモデルIDになっていることを確認します。サーバーにOpenAI APIキーを設定した場合、自動選択は `openai` になります。
2. フロントエンドを開発モードで動かす場合は `frontend/` で `npm ci` の後、例えば `DEV_API_PROXY_TARGET=http://127.0.0.1:8087 npm run dev` を実行し、`http://127.0.0.1:4000/` を開きます。
3. 「対話モデル」を開き、「自動」「OpenAI API」「ローカルLLM」から選びます。明示的に接続先を選んだときはモデルIDを入力できます。ローカルのモデルIDは `/v1/models` が返すID（Ollamaならタグを含む）に合わせてください。変更は次の送信から適用されます。選択はブラウザに保存されるので、キーの有無に再び追従したいときは「自動」に戻します。

LLMの設定だけを切り替えても、音声合成、会話DB、各方式の動作サービスは従来どおり必要です。新規クローンでは `backend/app/prompts/`（`president_persona.py`、`system_prompt.txt`、`rag/default.md` を含む）と学長TTSの重みを管理者から受け取ります。初期セットアップは [共同開発者向けの手順](https://github.com/Yoshiki0418/gakucho-ai-motion/pull/5) を参照してください。

## 動作方式の比較

会話画面の行動アバターで「動作生成」を開きます。方式は次のターンから反映されます。比較を再現するには、URLに `?motion_mode=layered` などのIDを指定できます。

| UIの方式 | ID | 発話中の動きと必要な資産 |
| --- | --- | --- |
| T2M生成（従来方式） | `current` | 行動タグをもとに専用T2Mサービスで動作を生成。3方式の初期設定。 |
| クリップ＋音声（新方式） | `layered` | 事前作成の意味クリップを待機姿勢に重ねて先に配信。研究評価用SentiAvatarのリズムが届くと、同じタグを維持したまま音声に合わせて更新。 |
| クリップ＋待機（新方式・リズムなし） | `layered-semantic` | 同じ意味クリップを待機姿勢に重ねる。SentiAvatarを用意せずにタグ、身振りの山場、部位ごとの合成を開発できる。 |

最初は「クリップ＋待機」で「手を振って挨拶してください」を試し、次に同じLLMと入力のまま「クリップ＋音声」に切り替えます。ブラウザのNetworkで `char-stream-orchestrator` の `motion_mode` イベントを見れば、要求した方式（`requested`）、実際の方式（`effective`）、切り替え理由（`reason`）を確認できます。座位やクリップ不足で従来方式に戻る場合があります。音声リズム生成だけに失敗した場合は、先に配信した意味動作が残ります。[層ごとの構成と既知の制約](conversation-motion-layers-2026-10-15.md)

| 変更したい箇所 | 実装の入口 |
| --- | --- |
| UIで方式を選び、送信する | [会話画面](../frontend/app/page.tsx)・[SSEの送受信](../frontend/features/text-chat/hooks/useTextChat.ts) |
| 返答の行動タグを解釈する | [タグとカタログIDの対応](../backend/app/agent/general_conversation/tools/motion_tags.py) |
| タグの時刻を決め、音声の動きと合成する | [クリップの配置](../backend/app/agent/general_conversation/tools/conversation_motion/session.py)・[部位別の重ね方](../backend/app/agent/general_conversation/tools/conversation_motion/composer.py) |
| 層を段階的に配信する | [リズムの生成と再配信](../backend/app/agent/general_conversation/tools/conversation_motion/pipeline.py)・[対話ルーター](../backend/app/routers/text_chat_router.py) |

動作合成の回帰テストはNumPyとSciPyのある環境で、リポジトリのルートから `python3 -m unittest discover -s backend/tests -p test_conversation_motion.py` を実行します。LLMの選択と動作方式は独立なので、動作だけを比較する実験ではモデルID・入力・姿勢を固定してください。
