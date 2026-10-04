#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0011_graph_teacher_40ep_seed0"
PYTHON=/Data0/kevinswk/envs/snn/bin/python
test -f "$OUT/COMPLETED"
test -s "$OUT/core.pt"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
  "$ROOT/collaborative_test/SW_0006_peer_spike_components/evaluate.py" \
  --checkpoint "$OUT/core.pt" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-dir "$OUT/validation_peer_readout" \
  --start 1320 --count 320 --steps 256 --settle 64 --batch-size 8 --device cuda

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
  "$ROOT/collaborative_test/SW_0011_graph_teacher_40ep/diagnose_signal_flow.py" \
  --checkpoint "$OUT/core.pt" \
  --gamma-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$OUT/signal_flow_40ep.json" \
  --start 1320 --count 16 --batch-size 4 --device cuda
touch "$OUT/VALIDATION_COMPLETED"
