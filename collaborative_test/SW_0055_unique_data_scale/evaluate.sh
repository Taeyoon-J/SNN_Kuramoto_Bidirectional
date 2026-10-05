#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"; SEED="${2:?seed required}"; WINDOW="${3:?short or long required}"
case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0055_unique2500_s${SEED}_e10_lr0p0003"
RESULT="$OUT/validation_${WINDOW}_T${STEPS}_settle${SETTLE}.json"; LOG="${RESULT%.json}.log"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing overwrite $RESULT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$OUT/core.pt" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
  --gamma-global-start 1320 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$RESULT" --start 1320 --count 320 --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0052_checkpoint_trajectory/validate_result.py" \
  "$RESULT" "$OUT/core.pt" 320 "$STEPS" "$SETTLE"
