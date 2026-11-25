#!/usr/bin/env bash
# Start only gakucho-ai-motion, including the current app and dedicated T2M.
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$TASK_ROOT"
export GAKUCHO_UE_BRIDGE_DIR
GAKUCHO_UE_BRIDGE_DIR=$(python3 "$TASK_ROOT/tools/ue-body-motion/bridge_path.py")
python3 "$TASK_ROOT/tools/prepare-t2m-runtime.py"
TASK_COMPOSE=(docker compose --project-name gakucho-ai-motion --project-directory "$TASK_ROOT" -f "$TASK_ROOT/docker-compose.prod.yml" -f "$TASK_ROOT/docker-compose.startup.yml")
TASK_MODE=gpu
if [[ ${1:-} == --cpu ]]; then
  TASK_COMPOSE+=(-f "$TASK_ROOT/docker-compose.cpu.yml")
  TASK_MODE=cpu
elif [[ -n ${1:-} ]]; then
  printf 'Usage: %s [--cpu]\n' "$0" >&2
  exit 2
elif ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  TASK_DRIVER_VERSION=$(awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit} }' /proc/driver/nvidia/version)
  export GAKUCHO_NVIDIA_DRIVER_DIR="$TASK_ROOT/.runtime/nvidia-$TASK_DRIVER_VERSION"
  for TASK_LIB in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
    if [[ ! -s "$GAKUCHO_NVIDIA_DRIVER_DIR/$TASK_LIB" ]]; then
      printf 'No matching GPU driver bundle for kernel %s. Fix the host driver or explicitly use --cpu.\n' "$TASK_DRIVER_VERSION" >&2
      exit 1
    fi
  done
  (cd "$GAKUCHO_NVIDIA_DRIVER_DIR" && sha256sum --check --status checksums.sha256)
  TASK_COMPOSE+=(-f "$TASK_ROOT/docker-compose.gpu-compat.yml")
  TASK_MODE=gpu-compat
fi
printf 'Starting gakucho-ai-motion (%s).\n' "$TASK_MODE"
"${TASK_COMPOSE[@]}" up -d --no-build --wait --wait-timeout 180
# Refresh DNS after any backend/frontend container recreation.
"${TASK_COMPOSE[@]}" exec -T nginx nginx -t
"${TASK_COMPOSE[@]}" exec -T nginx nginx -s reload
"${TASK_COMPOSE[@]}" ps
printf 'URL: https://192.168.29.19:30930/\n'
