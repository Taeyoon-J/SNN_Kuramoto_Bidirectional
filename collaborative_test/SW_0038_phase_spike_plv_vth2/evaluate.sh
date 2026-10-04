#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select an idle GPU after inspecting nvidia-smi}"
VARIANT="${2:?Use control or spike5}"
SEED="${3:-0}"
if [[ "$VARIANT" != control && "$VARIANT" != spike5 ]]; then
  echo "variant must be control or spike5" >&2
  exit 2
fi
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0038_${VARIANT}_seed${SEED}"
test -f "$OUT/core.pt"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0036_threshold_calibration/diagnose.py" \
  --checkpoint "$OUT/core.pt" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/stage_validation320.json" --count 320 --batch-size 8 \
  --device cuda --thresholds 2.0 \
  --intervention "matched vth2-trained phase/spike-PLV checkpoint stage diagnostic" \
  --warning "GT is used only for metrics and pair AUC, never prediction." \
  > "$OUT/stage_validation.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0035_adaptive_count_baseline/evaluate.py" \
  --checkpoint "$OUT/core.pt" --membrane-vth 2.0 \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/classifier_validation320.json" --count 320 --batch-size 8 \
  --device cuda --selected-only \
  > "$OUT/classifier_validation.log" 2>&1
