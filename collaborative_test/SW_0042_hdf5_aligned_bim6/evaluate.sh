#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass assigned GPU ID}"
SEED="${2:?Pass seed 0, 1, or 2}"
WINDOW="${3:?Use short or long}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "window must be short or long" >&2; exit 2 ;; esac

ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
CHECKPOINT="$OUT/core.pt"
GAMMA=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt
GAMMA_MANIFEST=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/manifest.json
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
RESULT="$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
test -s "$CHECKPOINT"
test -s "$GAMMA"
test -s "$GAMMA_MANIFEST"
test -s "$HDF5"

/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --gamma-path "$GAMMA" --gamma-global-start 1320 \
  --gamma-manifest "$GAMMA_MANIFEST" \
  --dataset-path "$HDF5" \
  --output-path "$RESULT" \
  --start 1320 --count 320 --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared \
  --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 \
  --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda \
  > "$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.log" 2>&1
