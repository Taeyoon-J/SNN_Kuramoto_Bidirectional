#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass assigned GPU ID}"
SEED="${2:-0}"
WINDOW="${3:?Use short or long}"
case "$SEED" in 0) ;; *) echo "SW0043 is a seed-0 diagnostic" >&2; exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "window must be short or long" >&2; exit 2 ;; esac

ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0043_factorized_BIM6_s${SEED}"
CHECKPOINT="$OUT/core.pt"
GAMMA=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
PEER_TARGETS=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
PEER_MANIFEST=/work/USERS/tkim1/clevr/with_masks/split_manifest.json
RESULT="$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
test -s "$CHECKPOINT"
test -s "$GAMMA"
test -s "$HDF5"
test -s "$PEER_TARGETS"
test -s "$PEER_MANIFEST"

/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --gamma-path "$GAMMA" --dataset-path "$HDF5" \
  --peer-targets "$PEER_TARGETS" --peer-manifest "$PEER_MANIFEST" \
  --peer-index-map identity \
  --output-path "$RESULT" --start 6000 --count 1000 \
  --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 \
  --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda \
  > "$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.log" 2>&1
