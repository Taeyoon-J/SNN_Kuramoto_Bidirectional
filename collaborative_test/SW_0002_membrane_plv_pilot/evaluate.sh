#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
ROOT=/Data0/kevinswk/patch_v2_sw
BASE="$ROOT/trained_models/baseline_best_seed0_20261003"
PILOT="$ROOT/trained_models/SW_0002_membrane_plv_pilot_seed0"
PYTHON=/Data0/kevinswk/envs/snn/bin/python
EVAL="$ROOT/collaborative_test/evaluate_fixed_split.py"
GAMMA=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
DATA=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$EVAL" \
  --checkpoint "$BASE/core.pt" --gamma-path "$GAMMA" \
  --dataset-path "$DATA" --output-dir "$BASE/validation_fixed_split" \
  --start 1320 --count 320 --steps 256 --settle 64 --k 8 --device cuda

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$EVAL" \
  --checkpoint "$PILOT/core.pt" --gamma-path "$GAMMA" \
  --dataset-path "$DATA" --output-dir "$PILOT/validation_fixed_split" \
  --start 1320 --count 320 --steps 256 --settle 64 --k 8 --device cuda

touch "$PILOT/VALIDATION_COMPLETED"
