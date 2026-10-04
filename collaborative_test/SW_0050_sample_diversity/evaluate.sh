#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass an assigned GPU ID}"
SEED="${2:?Pass seed 0 or 2}"
WEIGHT="${3:?Pass the training diversity weight}"
LR="${4:?Pass the training learning rate}"
MODE="${5:?Pass pilot or full}"
WINDOW="${6:?Pass short or long}"
case "$SEED" in 0|2) ;; *) echo "SEED must be 0 or 2" >&2; exit 2 ;; esac
case "$MODE" in pilot|full) ;; *) echo "MODE must be pilot or full" >&2; exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "WINDOW must be short or long" >&2; exit 2 ;; esac
TAG="${WEIGHT//./p}"
LR_TAG="${LR//./p}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s${SEED}_w${TAG}_lr${LR_TAG}_${MODE}"
CHECKPOINT="$OUT/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
RESULT="$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
LOG="$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.log"
if [[ -e "$RESULT" || -e "$LOG" ]]; then echo "Refusing to overwrite result/log: $RESULT" >&2; exit 1; fi
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
test -s "$CHECKPOINT"; test -s "$GAMMA"; test -s "$MANIFEST"; test -s "$HDF5"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-global-start 1320 \
  --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" --output-path "$RESULT" \
  --start 1320 --count 320 --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda \
  > "$LOG" 2>&1
test -s "$RESULT"
