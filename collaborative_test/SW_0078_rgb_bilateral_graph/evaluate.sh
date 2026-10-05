#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"
case "$GPU" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0078_stage1"
RESULT="$OUT/seed1_n32_rgb_graph.json"
LOG="${RESULT%.json}.log"
CHECKPOINT="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
test -s "$CHECKPOINT" -a -s "$GAMMA" -a -s "$MANIFEST"
[[ ! -e "$RESULT" && ! -e "$LOG" ]] || { echo "refusing existing output" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache${GPU}"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0078_rgb_bilateral_graph/evaluate.py \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" \
  --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
  --output-path "$RESULT" --start 1320 --count 32 --steps 256 --settle 64 \
  --threshold .35 --device cuda > "$LOG" 2>&1
test -s "$RESULT"
