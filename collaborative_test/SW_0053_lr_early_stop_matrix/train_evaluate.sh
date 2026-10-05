#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
SEED="${2:?seed required}"
LR="${3:?learning rate required}"
case "$SEED" in 0|1|2) ;; *) echo "invalid seed" >&2; exit 2 ;; esac
case "$LR" in 0.001|0.0003) ;; *) echo "LR must be 0.001 or 0.0003" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0050_sample_diversity"
TAG="${LR//./p}"
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s${SEED}_w0_lr${TAG}_epoch25"
test ! -e "$OUT"
bash "$DIR/run.sh" "$GPU_ID" "$SEED" 0 "$LR" epoch25
test -s "$OUT/TRAINING_COMPLETED" -a -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_25.pt"
bash "$DIR/evaluate.sh" "$GPU_ID" "$SEED" 0 "$LR" epoch25 short
bash "$DIR/evaluate.sh" "$GPU_ID" "$SEED" 0 "$LR" epoch25 long
printf 'completed\n' > "$OUT/EVALUATION_COMPLETED"
echo "SW0053 completed seed=$SEED lr=$LR"
