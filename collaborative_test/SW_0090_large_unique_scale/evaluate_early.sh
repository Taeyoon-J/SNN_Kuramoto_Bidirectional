#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU required}"; SEED="${2:?seed 0 or 1 required}"
case "$SEED" in 0|1) ;; *) exit 2;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0090_early_seed01"
mkdir -p "$OUT/cache${GPU}"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
for EPOCH in 1 3 10; do
  printf -v EP '%02d' "$EPOCH"
  CHECKPOINT="$ROOT/trained_models/SW0090_unique70000_s${SEED}_e10/checkpoints/epoch_${EP}.pt"
  RESULT="$OUT/seed${SEED}_epoch${EPOCH}.json"; LOG="${RESULT%.json}.log"
  test -s "$CHECKPOINT"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3
  /Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
   --checkpoint "$CHECKPOINT" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" --gamma-global-start 1320 \
   --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
   --output-path "$RESULT" --start 1320 --count 320 --steps 1024 --settle 512 --membrane-vth .06 \
   --min-group-size 2 --background largest_component --thresholds .50 \
   --dendritic-projection shared --graph-spatial-decay .35 --geodesic-steps 3 --geodesic-radius 1.5 \
   --geodesic-contrast 2 --geodesic-temperature .5 --geodesic-cap 16 --kuramoto-backend factorized \
   --gate-mode raw --device cuda > "$LOG" 2>&1
  test -s "$RESULT"
done

