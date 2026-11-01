#!/bin/sh
# SentiAvatar 研究用サーバをコンテナで起動する(非商用ライセンス・研究評価専用)。
# ホストのファイアウォールはコンテナ→ホストの通信を通さないため、学長バックエンドと
# 同じ Docker ネットワークにコンテナとして置き、`http://sentiavatar:8110` で呼ぶ。
#
# Usage: tools/sentiavatar/run-container.sh [NETWORK] [NAME]
#   NETWORK 既定: gakucho-motion-test(検証用の独立ネットワーク)
#   本番の会話画面で使う場合は gakucho-ai-motion_default を指定する。
set -e
cd "$(dirname "$0")/../.."
repo=$PWD
network=${1:-gakucho-motion-test}
name=${2:-gakucho-motion-sentiavatar}
# BVH の組み直しに使う conversation_motion(反映前は検証用コピーを使う)
cm=$repo/backend/app/agent/general_conversation/tools/conversation_motion
[ -d "$cm" ] || cm=$repo/.runtime/staging/backend-app/agent/general_conversation/tools/conversation_motion
docker network inspect "$network" >/dev/null 2>&1 || docker network create "$network" >/dev/null
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --network "$network" --network-alias sentiavatar \
  --restart unless-stopped \
  --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=${SENTIAVATAR_GPU:-0} \
  -e LD_LIBRARY_PATH="$repo/.runtime/warp-cuda-shim:$repo/.runtime/nvidia-595.91.07" \
  -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TRITON_LIBCUDA_PATH="$repo/.runtime/warp-cuda-shim" \
  -v "$repo/.runtime/nvidia-595.91.07:$repo/.runtime/nvidia-595.91.07:ro" \
  -v "$repo/.runtime/warp-cuda-shim:$repo/.runtime/warp-cuda-shim:ro" \
  -v "$repo/.runtime/sentiavatar:$repo/.runtime/sentiavatar:ro" \
  -v "$repo/tools/sentiavatar:$repo/tools/sentiavatar:ro" \
  -v "$cm:$cm:ro" \
  --entrypoint "$repo/.runtime/sentiavatar/env/bin/python" \
  gakucho-ai-motion-backend:prod \
  -m uvicorn --app-dir "$repo/tools/sentiavatar" server:app --host 0.0.0.0 --port 8110
echo "started $name on network $network (alias sentiavatar:8110)"
