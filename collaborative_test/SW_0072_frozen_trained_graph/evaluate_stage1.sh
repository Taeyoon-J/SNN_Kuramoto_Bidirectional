#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU required}"; SEED="${2:?seed required}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$SEED" in 1|2) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; CHECKPOINT="$ROOT/trained_models/SW0072_frozen_seed0_graph_s${SEED}_e10/core.pt"; OUT="$ROOT/trained_models/SW0072_stage1"; RESULT="$OUT/candidate_seed${SEED}_n32.json"; LOG="${RESULT%.json}.log"
test -s "$CHECKPOINT"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3; PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$PIDS" =~ [0-9] ]]
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CHECKPOINT" --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" \
 --gamma-global-start 1320 --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$RESULT" \
 --start 1320 --count 32 --steps 256 --settle 64 --membrane-vth 0.06 --min-group-size 2 \
 --background largest_component --thresholds 0.35 --dendritic-projection shared --graph-spatial-decay 0.35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature 0.5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"

