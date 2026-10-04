#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select an idle GPU after inspecting nvidia-smi}"
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
EVAL=/export_home/tkim1/collaborative_test/evaluate_model.py
G=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
T=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
M=/work/USERS/tkim1/clevr/with_masks/split_manifest.json
OUT=/Data0/kevinswk/peer_verify_20261004
mkdir -p "$OUT"
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" TRITON_CACHE_DIR="$OUT/cache_gpu${GPU_ID}" \
    "$PY" "$EVAL" \
    --checkpoint "/work/USERS/tkim1/runs/BIM6_s${SEED}/core.pt" \
    --gamma-seq-path "$G" --targets "$T" --manifest "$M" \
    --split validation --spike-per-component \
    --geodesic-steps 3 --geodesic-contrast 2.0 --graph-spatial-decay 0.35 \
    --synchrony-threshold 0.35 \
    --json-out "$OUT/BIM6_s${SEED}_validation1000_sync035.json" \
    > "$OUT/BIM6_s${SEED}_validation1000_sync035.log" 2>&1
done
