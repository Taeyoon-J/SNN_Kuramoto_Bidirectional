#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"; ARM="${2:?baseline, w0p1 or w1x required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$ARM" in baseline|w0p1|w1x) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
if [[ "$ARM" == baseline ]]; then
 MODEL="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10"
 CORE="$MODEL/core.pt"
else
 MODEL="$ROOT/trained_models/SW0081_graphflip_seed1_${ARM}"
 CORE="$MODEL/model/core.pt"
 test -f "$MODEL/model/TRAINING_COMPLETED"
fi
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"; MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW0081_fixed_pilot"; RESULT="$OUT/${ARM}_seed1_n32.json"; LOG="${RESULT%.json}.log"
test -s "$CORE" -a -s "$GAMMA" -a -s "$MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing result/log" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CORE" --gamma-path "$GAMMA" --gamma-global-start 1320 --gamma-manifest "$MANIFEST" \
 --dataset-path "$HDF5" --output-path "$RESULT" --start 1320 --count 32 \
 --steps 256 --settle 64 --membrane-vth .06 --min-group-size 2 --background largest_component \
 --thresholds .35 --dendritic-projection shared --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
