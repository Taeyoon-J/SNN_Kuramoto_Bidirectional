#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select an idle GPU after inspecting nvidia-smi}"
VARIANT="${2:?Use shared_b6, regional_b1, or regional_b6}"
SEED="${3:-0}"
case "$VARIANT" in
  control_b1|shared_b6|regional_b1|regional_b6) ;;
  *) echo "variant must be control_b1, shared_b6, regional_b1, or regional_b6" >&2; exit 2 ;;
esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0039_${VARIANT}_seed${SEED}"
CHECKPOINT="$OUT/core.pt"
PROJECTION=shared
if [[ "$VARIANT" == regional_* ]]; then PROJECTION=per_region; fi
if [[ "$VARIANT" == control_b1 ]]; then
  CHECKPOINT="$ROOT/trained_models/SW_0038_spike5_seed${SEED}/core.pt"
fi
test -f "$CHECKPOINT"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0036_threshold_calibration/diagnose.py" \
  --checkpoint "$CHECKPOINT" --dendritic-projection "$PROJECTION" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/stage_validation320.json" --count 320 --batch-size 8 \
  --device cuda --thresholds 2.0 \
  --intervention "matched vth2 component-spike PLV regional projection factorial" \
  --warning "GT is used only for metrics and pair AUC, never prediction." \
  > "$OUT/stage_validation.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0035_adaptive_count_baseline/evaluate.py" \
  --checkpoint "$CHECKPOINT" --dendritic-projection "$PROJECTION" --membrane-vth 2.0 \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/classifier_validation320.json" --count 320 --batch-size 8 \
  --device cuda --selected-only \
  > "$OUT/classifier_validation.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0039_per_region_transduction/diagnose_affinity.py" \
  --checkpoint "$CHECKPOINT" --dendritic-projection "$PROJECTION" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --output-path "$OUT/phase_spike_affinity.json" --device cuda
