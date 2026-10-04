#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass an assigned GPU ID}"
WEIGHT="${2:?Pass the seed2-selected weight}"
ROOT=/Data0/kevinswk/patch_v2_sw
TAG="${WEIGHT//./p}"
SEED2_OUT="$ROOT/trained_models/SW_0050_sample_diversity_s2_w${TAG}_lr0p001_full"
if [[ ! -s "$SEED2_OUT/EVALUATION_COMPLETED" || \
      ! -s "$SEED2_OUT/validation_short_T256_settle64.json" || \
      ! -s "$SEED2_OUT/validation_long_T1024_settle512.json" ]]; then
  echo "Run seed0 harm-control only after seed2 full short/long evaluations complete for weight $WEIGHT." >&2
  exit 1
fi
DIR="$ROOT/collaborative_test/SW_0050_sample_diversity"
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s0_w${TAG}_lr0p001_full"
LOG="$ROOT/trained_models/SW_0050_seed0_w${TAG}_full_launch.log"
if [[ -e "$OUT" || -e "$LOG" ]]; then echo "Refusing to overwrite seed0 control output/log: $OUT" >&2; exit 1; fi
while true; do
  if ! PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>&1)"; then
    echo "Unable to query GPU $GPU_ID: $PIDS" >&2; exit 2
  fi
  if [[ "$PIDS" =~ [0-9] ]]; then echo "GPU $GPU_ID is occupied; waiting 20 seconds."; sleep 20; else break; fi
done
bash "$DIR/preflight.sh" "$GPU_ID"
bash "$DIR/run.sh" "$GPU_ID" 0 "$WEIGHT" 0.001 full > "$LOG" 2>&1 &
TRAIN_PID=$!
bash "$DIR/watch_and_evaluate.sh" "$GPU_ID" 0 "$WEIGHT" 0.001 full "$TRAIN_PID"
