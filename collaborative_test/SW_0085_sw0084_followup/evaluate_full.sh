#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0-3 required}"; ARM="${2:?A/B/C/D required}"; EPOCH="${3:?05/10 required}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
case "$ARM" in A|B|C|D) ;; *) exit 2 ;; esac
case "$EPOCH" in 05|10) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
MODEL="$ROOT/trained_models/SW0084_${ARM}_seed1/epochs/epoch_${EPOCH}"
OUT="$ROOT/trained_models/SW0085_followup"
RESULT="$OUT/${ARM}_epoch${EPOCH}_seed1_full320_long.json"; LOG="${RESULT%.json}.log"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
test -s "$MODEL/core.pt" -a -s "$MODEL/gamma_validation.pt" -a -s "$MODEL/gamma_manifest.json"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing output" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$MODEL/core.pt" --gamma-path "$MODEL/gamma_validation.pt" --gamma-global-start 1320 \
 --gamma-manifest "$MODEL/gamma_manifest.json" --dataset-path "$HDF5" --output-path "$RESULT" \
 --start 1320 --count 320 --steps 1024 --settle 512 --membrane-vth .06 --min-group-size 2 \
 --background largest_component --thresholds .35 --affinity-modes spike --dendritic-projection shared \
 --graph-spatial-decay .35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature .5 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw \
 --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"

