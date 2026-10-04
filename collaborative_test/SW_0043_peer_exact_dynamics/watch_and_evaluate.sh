#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Pass assigned GPU ID}"
TRAIN_PID="${2:?Pass training launcher PID}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0043_factorized_BIM6_s0"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 15
done

test -s "$OUT/core.pt"
bash "$ROOT/collaborative_test/SW_0043_peer_exact_dynamics/evaluate.sh" "$GPU_ID" 0 short
bash "$ROOT/collaborative_test/SW_0043_peer_exact_dynamics/evaluate.sh" "$GPU_ID" 0 long
date -Is > "$OUT/EVALUATED"
