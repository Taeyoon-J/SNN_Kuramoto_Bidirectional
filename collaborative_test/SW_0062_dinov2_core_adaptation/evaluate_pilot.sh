#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0062_dinov2_s0_e5_lr0p0003"
RESULT="$OUT/validation_short_n32_T256_settle64.json"
LOG="${RESULT%.json}.log"
test -s "$OUT/TRAINING_COMPLETED" -a -s "$OUT/core.pt"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing result" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
  --checkpoint "$OUT/core.pt" \
  --gamma-path "$ROOT/data/SW_0062_dinov2_core_adaptation/dino_gamma_validation_1320_1639.pt" \
  --gamma-global-start 1320 \
  --gamma-manifest "$ROOT/data/SW_0062_dinov2_core_adaptation/manifest.json" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$RESULT" --start 1320 --count 32 --steps 256 --settle 64 \
  --membrane-vth 0.06 --min-group-size 2 --background largest_component \
  --thresholds 0.35 --dendritic-projection shared --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
echo completed > "$OUT/EVALUATION_COMPLETED"
