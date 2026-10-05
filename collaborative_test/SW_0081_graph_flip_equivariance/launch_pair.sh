#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
PREFLIGHT="$ROOT/trained_models/SW0081_preflight_seed1/PREFLIGHT_VALIDATED.json"
test -s "$PREFLIGHT"
for GPU in 0 1; do
 PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
 [[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
done
for OUT in "$ROOT/trained_models/SW0081_graphflip_seed1_w0p1" "$ROOT/trained_models/SW0081_graphflip_seed1_w1x"; do
 [[ ! -e "$OUT" ]] || { echo "refusing existing arm output: $OUT" >&2; exit 3; }
done
"$ROOT/collaborative_test/SW_0081_graph_flip_equivariance/run.sh" 0 w0p1 & PID0=$!
"$ROOT/collaborative_test/SW_0081_graph_flip_equivariance/run.sh" 1 w1x & PID1=$!
RC=0; wait "$PID0" || RC=$?; wait "$PID1" || RC=$?
exit "$RC"
