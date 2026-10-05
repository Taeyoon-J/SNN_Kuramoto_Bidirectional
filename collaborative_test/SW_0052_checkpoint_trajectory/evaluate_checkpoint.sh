#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
ARM="${2:?arm must be low_lr or diversity}"
EPOCH="${3:?epoch required}"
WINDOW="${4:?window must be short or long}"
case "$ARM" in
  low_lr) MODEL_DIR=SW_0050_sample_diversity_s2_w0_lr0p0003_full ;;
  diversity) MODEL_DIR=SW_0050_sample_diversity_s2_w2p45126043147_lr0p001_full ;;
  *) echo "arm must be low_lr or diversity" >&2; exit 2 ;;
esac
case "$EPOCH" in 20|25|30|35) ;; *) echo "epoch must be 20, 25, 30, or 35" >&2; exit 2 ;; esac
case "$WINDOW" in
  short) STEPS=256; SETTLE=64 ;;
  long) STEPS=1024; SETTLE=512 ;;
  *) echo "window must be short or long" >&2; exit 2 ;;
esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0052_checkpoint_trajectory"
CHECKPOINT="$ROOT/trained_models/$MODEL_DIR/checkpoints/epoch_${EPOCH}.pt"
OUT="$ROOT/trained_models/SW_0052_checkpoint_trajectory"
RESULT="$OUT/${ARM}_epoch${EPOCH}_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
LOG="$OUT/${ARM}_epoch${EPOCH}_${WINDOW}_T${STEPS}_settle${SETTLE}.log"
test -s "$CHECKPOINT"
mkdir -p "$OUT"
if [[ -e "$RESULT" || -e "$LOG" ]]; then echo "refusing overwrite: $RESULT" >&2; exit 1; fi
bash "$DIR/preflight_checkpoint.sh" "$GPU_ID" "$ARM" "$EPOCH"
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$OUT/cache_gpu${GPU_ID}" TRITON_CACHE_DIR="$OUT/cache_gpu${GPU_ID}"
mkdir -p "$TMPDIR"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" \
  --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
  --gamma-global-start 1320 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$RESULT" --start 1320 --count 320 --steps "$STEPS" --settle "$SETTLE" \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --thresholds 0.05 0.10 0.20 0.35 0.50 --phase-endpoint --device cuda > "$LOG" 2>&1
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_result.py" "$RESULT" "$CHECKPOINT" 320 "$STEPS" "$SETTLE"
echo "completed $RESULT"
