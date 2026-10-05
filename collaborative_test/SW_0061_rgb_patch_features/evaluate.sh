#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"
case "$GPU" in 0|1) ;; *) echo "GPU0/1 only" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0061_rgb_patch_features"
OUT="$ROOT/trained_models/SW_0061_rgb_patch_features"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
CHECKPOINT="$ROOT/trained_models/SW_0050_sample_diversity_s0_w0_lr0p0003_epoch25/checkpoints/epoch_25.pt"
GAMMA="$OUT/rgb_patch_gamma_validation_1320_1639.pt"
MANIFEST="$OUT/rgb_patch_gamma_validation_1320_1639.manifest.json"
RESULT="$OUT/seed0_rgb_patch_short_n32_T256_settle64.json"
LOG="$OUT/seed0_rgb_patch_short_n32_T256_settle64.log"
mkdir -p "$OUT/cache_gpu${GPU}"
test -s "$HDF5" -a -s "$CHECKPOINT"
if [[ ! -e "$GAMMA" && ! -e "$MANIFEST" ]]; then
  /Data0/kevinswk/envs/snn/bin/python "$DIR/generate_gamma.py" \
    --hdf5 "$HDF5" --start 1320 --count 320 --output "$GAMMA" --manifest "$MANIFEST"
fi
test -s "$GAMMA" -a -s "$MANIFEST"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "Refusing existing/partial result" >&2; exit 3; }
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
  --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
echo "SW0061 label-free RGB patch feature pilot completed"
