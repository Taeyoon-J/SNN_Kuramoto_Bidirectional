#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU}"; SEED="${2:?seed 0,1,2}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0071_hybrid"; RESULT="$OUT/seed${SEED}_long.json"; LOG="${RESULT%.json}.log"
if [[ "$SEED" == 0 ]]; then CKPT="$ROOT/trained_models/SW0055_unique2500_s0_e10_lr0p0003/core.pt"; else CKPT="$ROOT/trained_models/SW0066_graphinit0_s${SEED}_e10/core.pt"; fi
test -s "$CKPT"; [[ ! -e "$RESULT" && ! -e "$LOG" ]] || exit 3; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache${GPU}"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache${GPU}" TRITON_CACHE_DIR="$OUT/cache${GPU}"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0051_frozen_cc_spectral_hybrid/evaluate.py" --checkpoint "$CKPT" \
 --gamma-path "$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt" --gamma-manifest "$ROOT/data/SW_0042_hdf5_aligned/manifest.json" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$RESULT" --global-start 1320 \
 --count 320 --steps 1024 --settle 512 --batch-size 4 --spike-threshold .50 --device cuda > "$LOG" 2>&1
test -s "$RESULT"; printf 'completed\n' > "$OUT/seed${SEED}_COMPLETED"
