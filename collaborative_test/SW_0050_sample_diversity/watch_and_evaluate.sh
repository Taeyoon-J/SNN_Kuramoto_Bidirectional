#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass an assigned GPU ID}"
SEED="${2:?Pass seed 0 or 2}"
WEIGHT="${3:?Pass diversity weight}"
LR="${4:?Pass learning rate}"
MODE="${5:?Pass pilot or full}"
TRAIN_PID="${6:?Pass training PID}"
if [[ ! "$TRAIN_PID" =~ ^[0-9]+$ ]]; then echo "TRAIN_PID must be numeric" >&2; exit 2; fi
TAG="${WEIGHT//./p}"
LR_TAG="${LR//./p}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s${SEED}_w${TAG}_lr${LR_TAG}_${MODE}"
if [[ ! "$SEED" =~ ^(0|2)$ || ! "$MODE" =~ ^(pilot|full)$ ]]; then echo "invalid seed/mode" >&2; exit 2; fi
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 20; done
if [[ ! -s "$OUT/core.pt" || ! -s "$OUT/TRAINING_COMPLETED" ]]; then
  echo "Training did not complete successfully; inspect $OUT/training.log" >&2
  exit 1
fi
bash "$ROOT/collaborative_test/SW_0050_sample_diversity/evaluate.sh" "$GPU_ID" "$SEED" "$WEIGHT" "$LR" "$MODE" short
bash "$ROOT/collaborative_test/SW_0050_sample_diversity/evaluate.sh" "$GPU_ID" "$SEED" "$WEIGHT" "$LR" "$MODE" long
printf 'completed\n' > "$OUT/EVALUATION_COMPLETED"
echo "SW0050 training/evaluation complete: $OUT"
