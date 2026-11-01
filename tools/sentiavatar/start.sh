#!/bin/sh
# SentiAvatar 研究用常駐サーバを GPU 0 で起動する(非商用ライセンス・研究評価専用)。
# 学長バックエンドのコンテナから届くよう、Docker ネットワークのゲートウェイにだけ待ち受ける。
# Usage: tools/sentiavatar/start.sh   (ログ: .runtime/sentiavatar/server.log)
set -e
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH="$PWD/.runtime/warp-cuda-shim:$PWD/.runtime/nvidia-595.91.07"
export CUDA_VISIBLE_DEVICES=${SENTIAVATAR_GPU:-0} TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
host=${SENTIAVATAR_HOST:-172.20.0.1}
port=${SENTIAVATAR_PORT:-8110}
nohup .runtime/sentiavatar/env/bin/python -m uvicorn --app-dir tools/sentiavatar server:app \
  --host "$host" --port "$port" > .runtime/sentiavatar/server.log 2>&1 &
echo "sentiavatar server pid $! on http://$host:$port"
