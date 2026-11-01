#!/bin/sh
# Usage: world_model/tracker/launch_train.sh RUN_NAME LIBRARY_DIR [train_tracker args...]
# Starts tracker training on GPU 0 in the background; log at runs/RUN_NAME.log
# (MuJoCo Warp solver-iteration warnings are filtered out of the log).
set -e
cd "$(dirname "$0")/../.."
name=$1; library=$2; shift 2
runs=.runtime/world-model/tracker/runs
export LD_LIBRARY_PATH="$PWD/.runtime/warp-cuda-shim:$PWD/.runtime/nvidia-595.91.07"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.6
nohup sh -c ".runtime/world-model/mjx-venv/bin/python -u -m world_model.tracker.train_tracker \
  --library $library --output $runs/$name $* 2>&1 \
  | grep -v --line-buffered 'iterations limit\|warn_overflow\|^Module ' > $runs/$name.log" \
  > /dev/null 2>&1 &
echo "launched $name"
