#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0-3 required}"; ARM="${2:?baseline or H/I/J required}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0089_fixed_pilot"
if [[ "$ARM" == baseline ]]; then
 CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
 GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
 GMAN="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"; TAG=baseline
else
 case "$ARM" in H|I|J) ;; *) exit 2 ;; esac
 EP="$ROOT/trained_models/SW0089_${ARM}_seed1/epochs/epoch_05"
 test -s "$ROOT/trained_models/SW0089_${ARM}_seed1/TRAINING_COMPLETED"
 CORE="$EP/core.pt"; GAMMA="$EP/gamma_validation.pt"; GMAN="$EP/gamma_manifest.json"; TAG="$ARM"
fi
RESULT="$OUT/${TAG}_seed1_n32.json"; LOG="${RESULT%.json}.log"
test -s "$CORE" -a -s "$GAMMA" -a -s "$GMAN"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$PIDS" =~ [0-9] ]] || exit 4
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$CORE" --gamma-path "$GAMMA" --gamma-global-start 1320 --gamma-manifest "$GMAN" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$RESULT" \
 --start 1320 --count 32 --steps 256 --settle 64 --membrane-vth .06 --min-group-size 2 \
 --background largest_component --thresholds .35 --affinity-modes spike --dendritic-projection shared \
 --graph-spatial-decay .35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature .5 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw \
 --phase-endpoint --device cuda > "$LOG" 2>&1
test -s "$RESULT"
