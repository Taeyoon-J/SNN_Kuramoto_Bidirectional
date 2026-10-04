#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select the GPU assigned to this seed}"
SEED="${2:?Pass seed 0, 1, or 2}"
TRAIN_PID="${3:?Pass the active training process PID}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
if [[ ! "$TRAIN_PID" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_PID must be numeric" >&2
  exit 2
fi
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0041_BIM6_s${SEED}"
TRAIN_LOG="$OUT/training.log"
mkdir -p "$OUT/cache"

echo "Waiting for training PID $TRAIN_PID (seed $SEED) to exit" | tee "$OUT/watch.log"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 20
done

if [[ ! -s "$OUT/core.pt" ]] || ! grep -q "trained S2NetCore: $OUT/core.pt" "$TRAIN_LOG"; then
  echo "Training did not produce a completed checkpoint; see $TRAIN_LOG" | tee -a "$OUT/watch.log" >&2
  exit 1
fi
echo "Training completion marker and checkpoint verified" | tee -a "$OUT/watch.log"

bash "$ROOT/collaborative_test/SW_0041_peer_recipe_reproduction/evaluate.sh" \
  "$GPU_ID" "$SEED" our_validation
echo "Completed our_validation" | tee -a "$OUT/watch.log"
bash "$ROOT/collaborative_test/SW_0041_peer_recipe_reproduction/evaluate.sh" \
  "$GPU_ID" "$SEED" peer_validation
echo "Completed peer_validation" | tee -a "$OUT/watch.log"
bash "$ROOT/collaborative_test/SW_0041_peer_recipe_reproduction/evaluate.sh" \
  "$GPU_ID" "$SEED" peer_long
echo "Completed peer_long" | tee -a "$OUT/watch.log"
