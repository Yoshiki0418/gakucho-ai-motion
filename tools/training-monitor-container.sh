#!/bin/sh
# 学習の様子(/world-model/training)の書き出しを配る静的配信コンテナ。
# 書き出し係は world_model/tracker/start_monitor.sh(ホストで常駐、GPU 不使用)。
# フロントエンドは next.config.ts の rewrite で /training-monitor/ → http://training-monitor:80/ を読む。
set -e
cd "$(dirname "$0")/.."
network=${1:-gakucho-ai-motion_default}
docker rm -f gakucho-motion-training-monitor >/dev/null 2>&1 || true
docker run -d --name gakucho-motion-training-monitor --network "$network" --network-alias training-monitor \
  --restart unless-stopped \
  -v "$PWD/.runtime/world-model/monitor/public:/usr/share/nginx/html:ro" \
  nginx:latest
echo "training-monitor on $network"
