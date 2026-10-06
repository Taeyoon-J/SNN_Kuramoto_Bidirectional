#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?gpu}"; SEED="${2:?seed}"; EPOCH="${3:?epoch}"
ROOT=/Data0/kevinswk/patch_v2_sw; printf -v EP '%02d' "$EPOCH"
CHECKPOINT="$ROOT/trained_models/SW0092_our_on_official_s${SEED}/checkpoints/epoch_${EP}.pt"
OUT="$ROOT/trained_models/SW0092_our_on_official_eval"; RESULT="$OUT/seed${SEED}_epoch${EPOCH}.json"
test -s "$CHECKPOINT"; test ! -e "$RESULT"; mkdir -p "$OUT/cache$GPU"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache$GPU" TRITON_CACHE_DIR="$OUT/cache$GPU"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CHECKPOINT" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
 --gamma-global-start 1320 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$RESULT" \
 --start 1320 --count 320 --steps 1024 --settle 512 --membrane-vth .06 --min-group-size 2 \
 --background largest_component --thresholds .50 --dendritic-projection shared --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --device cuda > "${RESULT%.json}.log" 2>&1
test -s "$RESULT"

