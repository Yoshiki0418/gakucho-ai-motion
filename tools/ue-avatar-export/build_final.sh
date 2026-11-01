#!/bin/bash
# usage: build_final.sh <merged.glb> <out.glb> [hair.glb]
set -e
S=$(cd "$(dirname "$0")" && pwd)
SRC=$1; OUT=$2; HAIR=$3
T=$(mktemp -d)
python3 $S/shrink_textures.py "$SRC" $T/a.glb 1024 85 | tail -1
if [ -n "$HAIR" ]; then python3 $S/merge_glb.py $T/b.glb $T/a.glb "$HAIR" | tail -1; else cp $T/a.glb $T/b.glb; fi
python3 $S/finalize_glb.py $T/b.glb "$OUT"
python3 $S/check_weights.py "$OUT" | head -20
python3 $S/inspect_glb.py "$OUT" | grep -E "bin bytes|mesh\[|skin\["
rm -rf $T
