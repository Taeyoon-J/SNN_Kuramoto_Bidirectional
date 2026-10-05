#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 only}"; WHICH="${2:?baseline or epoch 1..5}"
[[ "$GPU" == 0 ]] || { echo 'SW0082 evaluation is GPU0-only' >&2; exit 2; }
ROOT=/Data0/kevinswk/patch_v2_sw
if [[ "$WHICH" == baseline ]]; then
 CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"; TAG=baseline
elif [[ "$WHICH" == sw0081 ]]; then
 CORE="$ROOT/trained_models/SW0081_graphflip_seed1_w0p1/model/core.pt"; TAG=sw0081
 test -s "$CORE" -a -s "$ROOT/trained_models/SW0081_graphflip_seed1_w0p1/model/TRAINING_COMPLETED"
else
 case "$WHICH" in 1|2|3|4|5) ;; *) exit 2 ;; esac
 CORE="$ROOT/trained_models/SW0082_graphflip_trajectory_seed1/epochs/epoch_0${WHICH}_core.pt"; TAG="epoch${WHICH}"
 test -s "$ROOT/trained_models/SW0082_graphflip_trajectory_seed1/model/TRAINING_COMPLETED"
fi
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"; MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW0082_fixed_pilot"; RESULT="$OUT/${TAG}_seed1_n32.json"; LOG="${RESULT%.json}.log"
test -s "$CORE" -a -s "$GAMMA" -a -s "$MANIFEST" -a -s "$HDF5"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing result/log" >&2; exit 3; }
PIDS="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo 'GPU0 is busy' >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES=0 TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CORE" --gamma-path "$GAMMA" --gamma-global-start 1320 --gamma-manifest "$MANIFEST" \
 --dataset-path "$HDF5" --output-path "$RESULT" --start 1320 --count 32 \
 --steps 256 --settle 64 --membrane-vth .06 --min-group-size 2 --background largest_component \
 --thresholds .35 --dendritic-projection shared --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 \
 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
