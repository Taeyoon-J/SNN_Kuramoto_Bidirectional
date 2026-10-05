#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU required}"; SEED="${2:?seed 1 or 2}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$SEED" in 1|2) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0072_full320_long"; CHECKPOINT="$ROOT/trained_models/SW0072_frozen_seed0_graph_s${SEED}_e10/core.pt"; RESULT="$OUT/seed${SEED}_long.json"; LOG="${RESULT%.json}.log"
test -s "$CHECKPOINT"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CHECKPOINT" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" --gamma-global-start 1320 \
 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
 --output-path "$RESULT" --start 1320 --count 320 --steps 1024 --settle 512 --membrane-vth .06 \
 --min-group-size 2 --background largest_component --thresholds .05 .10 .20 .35 .50 \
 --dendritic-projection shared --graph-spatial-decay .35 --geodesic-steps 3 --geodesic-radius 1.5 \
 --geodesic-contrast 2 --geodesic-temperature .5 --geodesic-cap 16 --kuramoto-backend factorized \
 --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"

