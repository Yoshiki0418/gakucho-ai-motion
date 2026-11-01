#!/bin/sh
# Usage: world_model/tracker/eval_checkpoint.sh RUN_DIR PARAMS_PKL OUT_DIR
# CPU (native MuJoCo) evaluation of one tracker checkpoint:
#   held-out mocap (64 clips), T2M-generated motions (40), phase-2 dev scenes (32).
set -e
cd "$(dirname "$0")/../.."
run=$1; params=$2; out=$3
py=.runtime/world-model/mjx-venv/bin/python
mkdir -p "$out"
$py -m world_model.tracker.evaluate_tracker --run "$run" --params "$params" \
  --library .runtime/world-model/tracker/motion-library-v1 --split test --limit 64 --workers 24 \
  --output "$out/mocap-test64.json" > "$out/mocap-test64.log" 2>&1
$py -m world_model.tracker.evaluate_tracker --run "$run" --params "$params" \
  --library .runtime/world-model/tracker/t2m-samples --split test --workers 24 \
  --output "$out/t2m40.json" > "$out/t2m40.log" 2>&1
$py -m world_model.tracker.navigation_eval --run "$run" --params "$params" \
  --library .runtime/world-model/tracker/motion-library-v1 --episodes 32 --workers 24 \
  --output "$out/nav32" > "$out/nav32.log" 2>&1
$py -m world_model.tracker.plot_navigation --result "$out/nav32" > /dev/null 2>&1 || true
$py -c "
import json, sys
m = json.load(open('$out/mocap-test64.json'))['summary']; t = json.load(open('$out/t2m40.json'))['summary']
n = json.load(open('$out/nav32/summary.json'))['summary']
print(json.dumps({'params': '$params', 'mocapTest': [m['successes'], m['clips']], 'mocapMeanErr': round(m['meanBodyErrorMeters'], 3),
                  't2m': [t['successes'], t['clips']], 't2mMeanErr': round(t['meanBodyErrorMeters'], 3),
                  'nav': [n['successes'], n['episodes']], 'navContacts': n['contacts'], 'navFalls': n['falls'], 'navBySide': n['bySide']}))
" | tee "$out/summary.json"
