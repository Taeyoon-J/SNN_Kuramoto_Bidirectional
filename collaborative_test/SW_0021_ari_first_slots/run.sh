#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0021_ari_first_slots"
PYTHON=/Data0/kevinswk/envs/snn/bin/python
CHECKPOINT="$ROOT/trained_models/SW_0003_membrane_plv_40ep_seed0/core.pt"
test -s "$CHECKPOINT"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"

for SOURCE in spikes membrane; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
    "$ROOT/collaborative_test/SW_0013_dynamic_spike_slots/evaluate.py" \
    --checkpoint "$CHECKPOINT" \
    --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
    --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
    --output-path "$OUT/${SOURCE}_validation320.json" \
    --start 1320 --count 320 --steps 256 --settle 64 \
    --batch-size 8 --thresholds 0.3 0.5 0.7 0.9 \
    --initial-slots 3 6 --assignment-seeds 0 \
    --source "$SOURCE" --device cuda
done
touch "$OUT/VALIDATION_COMPLETED"
