# 共同開発者向けセットアップ

学長AIの会話画面を手元で開くには **Git・Node.js 20・npm** があれば始められます。音声と身体動作を含む対話には、別途バックエンドと動作生成サービス（T2M）が必要です。

## Claude Codeへの最初の依頼

リポジトリを開いたClaude Codeに、次を渡してください。

> `handoff/collaborator-setup.md` を読み、私の端末で学長AIの開発環境を用意してください。まずローカルで会話画面を開いてください。対話も試す場合は、利用できる開発用バックエンドを確認し、下の該当手順へ進んでください。最後に起動したURL、通った確認、不足している資産の名前を教えてください。APIキーや秘密鍵の値は表示しないでください。

## 1. 手元で会話画面を開く

初めて取得する場合は以下を実行します。既にクローンしている場合は、そのルートから `frontend` に移ります。

~~~bash
git clone https://github.com/Yoshiki0418/gakucho-ai-motion.git
cd gakucho-ai-motion
cd frontend
node --version
npm ci
npm run dev -- --hostname 127.0.0.1
~~~

`http://127.0.0.1:4000/` を開き、チャット入力欄と動作生成モードの選択が表示されればUIの準備は完了です。初期設定はタグごとにT2M-GPTで生成する「現行」です。必要に応じて「新方式」「新方式・リズムなし」を選べます。[モードの定義](../frontend/app/page.tsx)・[開発用コマンド](../frontend/package.json)

## 2. 提供済みの開発用バックエンドで対話する

管理者から受け取るのは、開発用バックエンドの接続先です。APIキーとモデルはサーバー側で管理できます。例えば、接続を許可されたSSHホストでバックエンドが `127.0.0.1:8087` を利用している場合、Next.jsを動かす環境（WSLなら同じWSL）の別ターミナルで転送します。`user@dev-host` は自分の接続先に置き換えます。

~~~bash
ssh -N -L 127.0.0.1:18087:127.0.0.1:8087 user@dev-host
~~~

次を `frontend` で実行します。手順1の開発サーバーが動いていれば、そのターミナルで Ctrl+C を押して起動し直します。直接接続できる環境では、下のURLを提供されたバックエンドのURLに置き換えます。

~~~bash
curl -fsS http://127.0.0.1:18087/health
DEV_API_PROXY_TARGET=http://127.0.0.1:18087 npm run dev -- --hostname 127.0.0.1
~~~

ヘルス応答は `{"status":"ok"}` です。画面から短い挨拶を1件送って、返答テキスト、音声、アバターの身振りを確認します。この確認では接続先のLLM APIを使い、会話履歴は接続先のDBに保存されます。[APIとWebSocketの転送設定](../frontend/next.config.ts)

## 3. 専用マシンでバックエンドも用意する場合

この節は自分用のLinuxマシンまたは専用のDocker環境で実施します。共有ホストで既存サービスを利用する場合は手順2で接続できます。Composeには固定コンテナ名・ポートと共有サーバー用のパスがあるため、同じDockerデーモンではプロジェクト名だけ変えても既存環境と衝突します。

### 必要なもの

| 対象 | 配置と確認事項 |
| --- | --- |
| 実行環境 | Bash、Python 3、Docker Engine、`gpus` と `!override` / `!reset` を解釈できるDocker Compose v2。GPU実行にはNVIDIAドライバとNVIDIA Container Toolkitも必要です。 |
| DBとAPI設定 | ルートの `.env` に開発用の `POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB` を設定。Composeは `HF_USERNAME`、`HF_TOKEN` も参照します。`backend/.env` に利用を許可された `OPENAI_API_KEY` を設定します。`DB_URL` はComposeが組み立てます。DBは空の `data/postgres/` から初期化できます。 |
| 応答・音声モデル | 管理者から `backend/app/prompts/system_prompt.txt`、学長音声用の `backend/app/weights/tts/gakucho_ai_v2/`（`config.json`、`style_vectors.npy`、`G_*.pth`）を受け取ります。検索モデル `cl-nagoya/ruri-v3-310m` はバックエンドから取得できるか、イメージ内にキャッシュが必要です。 |
| 動作生成 | このリポジトリと同じ親ディレクトリに、対応する版の `Custom-T2M-Lab/` を置きます。`t2m_gpt/inference.py`、`web_viewer/app.py` および使用する学習済みモデル一式を管理者から受け取ります。 |
| Dockerイメージ | 専用T2M用の `gakucho-ai_devcontainer-backend` と、対話用の `gakucho-ai-motion-backend:prod` を用意します。管理者提供のイメージアーカイブは `docker load -i ファイル名.tar` で読み込めます。バックエンドを再ビルドする場合、Dockerfileは `COPY . .` を使うため、認証ファイルや非公開資産を除いたビルドコンテキストを使います。 |
| HTTPS | 開発ホスト名に対応した証明書を `nginx/certs/server.crt`、その開発用秘密鍵を `nginx/certs/server.key` に置き、ブラウザで発行元を信頼します。 |

[Composeの基本定義](../docker-compose.prod.yml)・[起動時の上書き](../docker-compose.startup.yml)・[音声モデルの読込](../backend/app/models/tts/style_bert_vits2_tts.py)・[DBの初期化](../db/Dockerfile)

### 専用マシンの設定と起動

以下は、Linuxの専用マシン上で `https://localhost:30930` を使う例です。コマンドはリポジトリのルートから実行してください。`.runtime/compose/local.yml` に以下を保存します（親ディレクトリは `mkdir -p .runtime/compose` で作成）。ベース定義の共有ホスト用T2Mマウントを、同じ親に置いた自分のコピーへ差し替えます。

~~~yaml
services:
  t2m:
    volumes:
      - ../Custom-T2M-Lab:/workspace
  frontend:
    build:
      args:
        NEXT_PUBLIC_API_BASE_URL: https://localhost:30930
    environment:
      NEXT_PUBLIC_API_BASE_URL: https://localhost:30930
  nginx:
    ports: !override
      - "127.0.0.1:30930:30930"
~~~

GPU版の基本定義はバックエンドにGPU番号 `1` を割り当てます。専用マシンの番号が異なる場合は、ローカル上書きの `backend` で `deploy.resources.reservations.devices` と `NVIDIA_VISIBLE_DEVICES` を割当先に合わせます。次はCPUで試す例です。GPUを使う場合は下記の `docker-compose.cpu.yml` の行を外します。

~~~bash
python3 tools/prepare-t2m-runtime.py
compose=(docker compose --project-name gakucho-ai-motion
  -f docker-compose.prod.yml -f docker-compose.startup.yml
  -f .runtime/compose/local.yml
  -f docker-compose.cpu.yml)
docker image inspect gakucho-ai_devcontainer-backend gakucho-ai-motion-backend:prod --format '{{.Id}}'
"${compose[@]}" config --quiet
"${compose[@]}" build db frontend
"${compose[@]}" up -d --no-build --wait
"${compose[@]}" ps
curl -fsS http://127.0.0.1:8087/health
curl -fsS http://127.0.0.1:8100/api/realtime/status
~~~

`ps` でDB・T2M・バックエンドのヘルスとfrontend・nginxの起動を確認し、`https://localhost:30930/` で手順2と同じ対話を試します。終了する場合は同じシェルで `"${compose[@]}" stop` を実行します。既存の [共有サーバー用起動スクリプト](../tools/start-system.sh) はイメージをビルドせず、専用マシン用の `local.yml` も読み込まないため、上の例ではComposeを直接使っています。

### 起動時に詰まったら

- **画面は開くが応答しない** — ターミナルから接続先の `/health` を確認し、`DEV_API_PROXY_TARGET` 設定後にNext.jsを再起動します。
- **T2Mの準備または起動が失敗する** — `../Custom-T2M-Lab` の配置と対応バージョンを確認します。`T2M inference changed` は準備スクリプトが対象版を識別できなかったことを示します。[準備スクリプト](../tools/prepare-t2m-runtime.py)
- **バックエンドがhealthyにならない** — DBのヘルス、システムプロンプト、TTSの3種のファイル、検索モデルへのアクセスを順に確認します。起動時にモデルを読み込みます。[モデル初期化](../backend/app/models/model_registry.py)
- **「新方式」で音声リズムのみ出ない** — SentiAvatarは別配布の研究評価用サービスです。「現行」で基本動作を先に確認できます。必要な場合は管理者から利用できるサービスまたは重みを受け取ってください。[SentiAvatarの手順](../docs/sentiavatar-eval.md)

住宅や階段の物理学習は別の開発環境です。担当する場合は [物理環境のREADME](../world_model/README.md) と [住宅動作の引き継ぎ](claude-course-correction-2026-10-15.md) を参照してください。
