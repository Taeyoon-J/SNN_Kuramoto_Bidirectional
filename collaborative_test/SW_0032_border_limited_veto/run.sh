#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:-3}"
COUNT="${2:-64}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0032_border_limited_veto"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
CUDA_VISIBLE_DEVICES="$GPU_ID" /Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0032_border_limited_veto/evaluate_veto.py" \
  --checkpoint "$ROOT/trained_models/SW_0003_membrane_plv_40ep_seed0/core.pt" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/validation${COUNT}.json" \
  --start 1320 --count "$COUNT" --batch-size 8 --device cuda
