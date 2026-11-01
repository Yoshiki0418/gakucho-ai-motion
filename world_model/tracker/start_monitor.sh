#!/bin/sh
# 学習の様子の書き出し係を常駐させる(GPU は使わない)。ログ: .runtime/world-model/monitor/monitor.log
# 配信は tools/training-monitor/run-container.sh のコンテナが行う。
set -e
cd "$(dirname "$0")/../.."
mkdir -p .runtime/world-model/monitor/public
nohup env JAX_PLATFORMS=cpu .runtime/world-model/mjx-venv/bin/python -u -m world_model.tracker.monitor \
  > .runtime/world-model/monitor/monitor.log 2>&1 &
echo "monitor pid $!"
