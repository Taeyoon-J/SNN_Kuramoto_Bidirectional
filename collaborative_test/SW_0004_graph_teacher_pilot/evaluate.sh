#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0004_graph_teacher_pilot_seed0"
test -s "$OUT/core.pt"
CUDA_VISIBLE_DEVICES="$GPU_ID" /Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/evaluate_fixed_split.py" \
  --checkpoint "$OUT/core.pt" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-dir "$OUT/validation_fixed_split" \
  --start 1320 --count 320 --steps 256 --settle 64 --k 8 --device cuda
touch "$OUT/VALIDATION_COMPLETED"
