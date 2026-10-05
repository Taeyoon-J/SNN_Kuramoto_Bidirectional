#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"
case "$GPU" in 0|1) ;; *) echo "GPU0/1 only" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
CHECKPOINT="$ROOT/trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW_0060_signed_mask_gate"
RESULT="$OUT/seed0_signed_mask_short_n32_T256_settle64.json"
LOG="$OUT/seed0_signed_mask_short_n32_T256_settle64.log"
VALIDATED="$OUT/seed0_signed_mask_short_n32_T256_settle64_validated.json"
mkdir -p "$OUT/cache_gpu${GPU}"
test -s "$CHECKPOINT" -a -s "$GAMMA" -a -s "$MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" && ! -e "$VALIDATED" ]] || { echo "Refusing existing/partial result" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache_gpu${GPU}" TRITON_CACHE_DIR="$OUT/cache_gpu${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-global-start 1320 \
  --gamma-manifest "$MANIFEST" --dataset-path "$HDF5" --output-path "$RESULT" \
  --start 1320 --count 32 --steps 256 --settle 64 --membrane-vth 0.06 \
  --min-group-size 2 --background largest_component --thresholds 0.35 \
  --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --gate-mode signed_mask --phase-endpoint --device cuda > "$LOG" 2>&1
/Data0/kevinswk/envs/snn/bin/python \
  "$ROOT/collaborative_test/SW_0059_centered_delayed_gate/validate_result.py" \
  "$RESULT" "$CHECKPOINT" signed_mask > "$VALIDATED"
echo "SW0060 signed-mask validation pilot completed"
