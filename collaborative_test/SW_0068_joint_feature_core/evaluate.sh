#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1}"; TAG="${2:?lr3e6 or lr3e5}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$TAG" in lr3e6|lr3e5) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
MODEL="$ROOT/trained_models/SW0068_joint_s1_${TAG}"
OUT="$ROOT/trained_models/SW0068_stage1"
RESULT="$OUT/${TAG}_seed1_n32.json"; LOG="${RESULT%.json}.log"
test -f "$MODEL/TRAINING_COMPLETED" -a -s "$MODEL/model/core.pt" -a -s "$MODEL/gamma_validation.pt"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$PIDS" =~ [0-9] ]]
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$MODEL/model/core.pt" --gamma-path "$MODEL/gamma_validation.pt" \
 --gamma-global-start 1320 --gamma-manifest "$MODEL/gamma_manifest.json" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
 --output-path "$RESULT" --start 1320 --count 32 --steps 256 --settle 64 \
 --membrane-vth 0.06 --min-group-size 2 --background largest_component --thresholds 0.35 \
 --dendritic-projection shared --graph-spatial-decay 0.35 --geodesic-steps 3 \
 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature 0.5 --geodesic-cap 16 \
 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
