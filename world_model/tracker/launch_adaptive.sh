#!/bin/sh
# Usage: world_model/tracker/launch_adaptive.sh RUN LIBRARY RESTORE_PARAMS ROUNDS STEPS_PER_ROUND [train_tracker args...]
# Starts adaptive_train.sh in the background on GPU 0; log at runs/RUN.log
# (MuJoCo Warp solver-iteration warnings are filtered out, as in launch_train.sh).
set -e
cd "$(dirname "$0")/../.."
name=$1
runs=.runtime/world-model/tracker/runs
nohup sh -c "sh world_model/tracker/adaptive_train.sh $* 2>&1 \
  | grep -v --line-buffered 'iterations limit\|warn_overflow\|^Module ' > $runs/$name.log" \
  > /dev/null 2>&1 &
echo "launched $name"
