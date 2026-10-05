#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"; SEED="${2:?seed required}"
DIR=/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0055_unique_data_scale
OUT="/Data0/kevinswk/patch_v2_sw/trained_models/SW0055_unique2500_s${SEED}_e10_lr0p0003"
bash "$DIR/run.sh" "$GPU_ID" "$SEED"
bash "$DIR/evaluate.sh" "$GPU_ID" "$SEED" short
bash "$DIR/evaluate.sh" "$GPU_ID" "$SEED" long
echo completed > "$OUT/EVALUATION_COMPLETED"
