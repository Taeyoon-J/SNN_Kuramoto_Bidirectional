#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0026_peer_low_threshold"
PYTHON=/Data0/kevinswk/envs/snn/bin/python
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"

for TAG in SW_0003_membrane_plv_40ep_seed0 SW_0004_graph_teacher_pilot_seed0 SW_0011_graph_teacher_40ep_seed0; do
  CHECKPOINT="$ROOT/trained_models/$TAG/core.pt"
  test -s "$CHECKPOINT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
    "$ROOT/collaborative_test/SW_0006_peer_spike_components/evaluate.py" \
    --checkpoint "$CHECKPOINT" \
    --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
    --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
    --output-dir "$OUT/$TAG" \
    --start 1320 --count 320 --steps 256 --settle 64 --batch-size 8 \
    --thresholds 0.0005 0.002 0.005 0.01 0.02 0.04 0.08 0.1 0.2 0.5 \
    --device cuda
done
touch "$OUT/VALIDATION_COMPLETED"
