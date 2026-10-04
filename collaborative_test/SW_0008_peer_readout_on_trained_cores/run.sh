#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
ROOT=/Data0/kevinswk/patch_v2_sw
PYTHON=/Data0/kevinswk/envs/snn/bin/python
EVAL="$ROOT/collaborative_test/SW_0006_peer_spike_components/evaluate.py"
GAMMA=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
DATA=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW_0008_peer_readout_on_trained_cores"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"

for EXPERIMENT in SW_0003_membrane_plv_40ep_seed0 SW_0004_graph_teacher_pilot_seed0; do
  CHECKPOINT="$ROOT/trained_models/$EXPERIMENT/core.pt"
  test -s "$CHECKPOINT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$EVAL" \
    --checkpoint "$CHECKPOINT" \
    --gamma-path "$GAMMA" \
    --dataset-path "$DATA" \
    --output-dir "$OUT/$EXPERIMENT" \
    --start 1320 --count 320 --steps 256 --settle 64 --batch-size 8 --device cuda
done
touch "$OUT/COMPLETED"
