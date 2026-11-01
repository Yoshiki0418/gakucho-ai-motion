#!/bin/sh
# 学習中の重みで学長の身体を実時間で動かして配る(/world-model/live)。CPU のみ・GPU 不使用。
# 学習用の venv(.runtime/world-model/mjx-venv、元の Python は miniforge3/envs/tts)を読み取り専用で借りる。
# フロントエンドは next.config.ts の rewrite で /training-live/ → http://training-live:8120/ を読む。
# Usage: tools/training-live-container.sh [NETWORK] [NAME]
set -e
cd "$(dirname "$0")/.."
repo=$PWD
network=${1:-gakucho-ai-motion_default}
name=${2:-gakucho-motion-training-live}
conda_env=/gpu-server/yamamoto/miniforge3/envs/tts
docker network inspect "$network" >/dev/null 2>&1 || docker network create "$network" >/dev/null
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --network "$network" --network-alias training-live \
  --restart unless-stopped --cpus 6 \
  -e JAX_PLATFORMS=cpu -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  -v "$conda_env:$conda_env:ro" \
  -v "$repo:$repo:ro" \
  -w "$repo" \
  --entrypoint "$repo/.runtime/world-model/mjx-venv/bin/python" \
  gakucho-ai-motion-backend:prod \
  -m world_model.tracker.live_server --host 0.0.0.0 --port 8120
echo "started $name on $network (alias training-live:8120)"
