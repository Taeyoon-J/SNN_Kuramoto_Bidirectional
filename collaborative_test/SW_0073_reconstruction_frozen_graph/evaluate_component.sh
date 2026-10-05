#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU}"; TAG="${2:?w0p3 or w1}"; MODE="${3:?feature_only or core_only}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$TAG" in w0p3|w1) ;; *) exit 2 ;; esac; case "$MODE" in feature_only|core_only) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; MODEL="$ROOT/trained_models/SW0073_s1_${TAG}"; OUT="$ROOT/trained_models/SW0073_stage1"; RESULT="$OUT/${TAG}_${MODE}_seed1_n32.json"; LOG="${RESULT%.json}.log"
if [[ "$MODE" == feature_only ]]; then CHECKPOINT="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"; GAMMA="$MODEL/gamma_validation.pt"; MANIFEST="$MODEL/gamma_manifest.json"; else CHECKPOINT="$MODEL/model/core.pt"; GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"; MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"; fi
test -s "$CHECKPOINT" -a -s "$GAMMA" -a -s "$MANIFEST"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-global-start 1320 --gamma-manifest "$MANIFEST" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$RESULT" \
 --start 1320 --count 32 --steps 256 --settle 64 --membrane-vth .06 --min-group-size 2 \
 --background largest_component --thresholds .35 --dendritic-projection shared --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"

