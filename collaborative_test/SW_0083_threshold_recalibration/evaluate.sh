#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"; VTAG="${2:?v006, v05, v10, or v20 required}"
case "$GPU" in 0|1) ;; *) echo 'GPU0/1 only' >&2; exit 2 ;; esac
case "$VTAG" in v006) VTH=.06 ;; v05) VTH=.5 ;; v10) VTH=1.0 ;; v20) VTH=2.0 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
GAMMA_MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW0083_threshold_recalibration"
RESULT="$OUT/${VTAG}_seed1_n32.json"; LOG="${RESULT%.json}.log"
test -s "$CORE" -a -s "$GAMMA" -a -s "$GAMMA_MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing result/log: $RESULT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0040_peer_transfer/evaluate.py \
 --checkpoint "$CORE" --gamma-path "$GAMMA" --gamma-global-start 1320 --gamma-manifest "$GAMMA_MANIFEST" \
 --dataset-path "$HDF5" --output-path "$RESULT" --start 1320 --count 32 \
 --steps 256 --settle 64 --membrane-vth "$VTH" --min-group-size 2 --background largest_component \
 --thresholds .35 --affinity-modes spike --dendritic-projection shared --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --event-diagnostics --device cuda \
 > "$LOG" 2>&1
test -s "$RESULT"
