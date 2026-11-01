#!/bin/sh
# Usage: world_model/tracker/adaptive_train.sh RUN LIBRARY RESTORE_PARAMS ROUNDS STEPS_PER_ROUND [train_tracker args...]
#
# Training that practises the clips the policy fails (PHC hard-negative mining), in rounds:
#   1. sweep_clips plays every training clip once on the GPU with the current weights
#   2. train_tracker trains STEPS_PER_ROUND with half of the episodes drawn from the failed clips
# Each step is its own process, so GPU memory held by brax's evaluations is returned
# every round (seated training ran out of memory at the 8th evaluation in one process).
# Always on GPU 0. Log: runs/RUN.log (launch with launch_adaptive.sh); sweeps: runs/RUN/sweeps/.
set -e
cd "$(dirname "$0")/../.."
name=$1; library=$2; params=$3; rounds=$4; per=$5; shift 5
runs=.runtime/world-model/tracker/runs
out=$runs/$name
export LD_LIBRARY_PATH="$PWD/.runtime/warp-cuda-shim:$PWD/.runtime/nvidia-595.91.07"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.6
python=.runtime/world-model/mjx-venv/bin/python
mkdir -p "$out/sweeps"
round=1
offset=0
while [ "$round" -le "$rounds" ]; do
  echo "[adaptive] round $round/$rounds: sweep from step $offset ($(date +%T))"
  if [ ! -f "$out/sweeps/round-$round.json" ]; then   # kept when a round is restarted
    $python -u -m world_model.tracker.sweep_clips --run "$(dirname "$params")" --params "$params" \
      --library "$library" --include-seated --output "$out/sweeps/round-$round.json"
  fi
  more=--more-rounds
  [ "$round" -eq "$rounds" ] && more=
  echo "[adaptive] round $round/$rounds: train ($(date +%T))"
  $python -u -m world_model.tracker.train_tracker --library "$library" --output "$out" --timesteps "$per" \
    --restore "$params" --clip-weights "$out/sweeps/round-$round.json" --step-offset "$offset" \
    --planned-timesteps $((rounds * per)) --include-seated --evals 2 $more "$@"
  params=$out/final_params.pkl
  # brax rounds the step count up to whole updates; continue from where this round really ended
  offset=$($python -c "import json; print(json.loads(open('$out/progress.jsonl').read().splitlines()[-1])['step'])")
  round=$((round + 1))
done
echo "[adaptive] round $rounds/$rounds: final sweep ($(date +%T))"
$python -u -m world_model.tracker.sweep_clips --run "$out" --params "$params" \
  --library "$library" --include-seated --output "$out/sweeps/final.json"
echo "[adaptive] done ($(date +%T))"
