#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Select the GPU assigned to this seed}"
SEED="${2:?Pass seed 0, 1, or 2}"
MODE="${3:?Use our_validation, peer_validation, or peer_long}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
case "$MODE" in our_validation|peer_validation|peer_long) ;; *) echo "mode must be our_validation, peer_validation, or peer_long" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0041_BIM6_s${SEED}"
CHECKPOINT="$OUT/core.pt"
GAMMA=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
PEER_TARGETS=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
PEER_MANIFEST=/work/USERS/tkim1/clevr/with_masks/split_manifest.json
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
test -f "$CHECKPOINT"

if [[ "$MODE" == our_validation ]]; then
  START=1320
  COUNT=320
  THRESHOLDS=(0.05 0.10 0.15 0.20 0.25 0.35 0.50)
  STEPS=256
  SETTLE=64
elif [[ "$MODE" == peer_long ]]; then
  START=6000
  COUNT=1000
  THRESHOLDS=(0.05 0.10 0.15 0.20 0.35)
  STEPS=1024
  SETTLE=512
else
  START=6000
  COUNT=1000
  THRESHOLDS=(0.20 0.35 0.50)
  STEPS=256
  SETTLE=64
fi

/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --gamma-path "$GAMMA" --dataset-path "$HDF5" \
  --peer-targets "$PEER_TARGETS" --peer-manifest "$PEER_MANIFEST" \
  --peer-index-map identity \
  --output-path "$OUT/${MODE}.json" \
  --start "$START" --count "$COUNT" --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --phase-endpoint \
  --dendritic-projection shared \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 \
  --graph-spatial-decay 0.35 \
  --thresholds "${THRESHOLDS[@]}" --device cuda \
  > "$OUT/${MODE}.log" 2>&1
