#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
WINDOW="${2:?short or long required}"
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "window must be short or long" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation"
CHECKPOINT="$ROOT/trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW_0054_causal_mechanism_ablation"
RESULT="$OUT/seed0_epoch25_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
LOG="$OUT/seed0_epoch25_${WINDOW}.log"
test -s "$CHECKPOINT" && test -s "$GAMMA" && test -s "$MANIFEST" && test -s "$HDF5" && test -s "$OUT/PREFLIGHT_V1.json"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "Refusing to overwrite SW0054 output: $RESULT" >&2; exit 3; }
if ! PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then echo "Unable to query GPU $GPU_ID: $PIDS" >&2; exit 2; fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU_ID occupied" >&2; exit 2; }
bash "$DIR/preflight.sh" "$GPU_ID"
if ! PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then echo "Unable to query GPU $GPU_ID: $PIDS" >&2; exit 2; fi
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU_ID became occupied after preflight" >&2; exit 2; }
mkdir -p "$OUT/cache_gpu${GPU_ID}"
export CUDA_VISIBLE_DEVICES="$GPU_ID" TMPDIR="$OUT/cache_gpu${GPU_ID}" TRITON_CACHE_DIR="$OUT/cache_gpu${GPU_ID}"
/Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" --checkpoint "$CHECKPOINT" \
  --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" \
  --output-path "$RESULT" --start 1320 --count 320 --steps "$STEPS" --settle "$SETTLE" \
  --batch-size 2 --device cuda > "$LOG" 2>&1
