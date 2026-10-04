#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
SEED="${2:?seed 0, 1, or 2 required}"
WINDOW="${3:?short or long required}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
case "$WINDOW" in short) STEPS=256; SETTLE=64 ;; long) STEPS=1024; SETTLE=512 ;; *) echo "window must be short or long" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
CHECKPOINT="$OUT/core.pt"
GAMMA=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt
MANIFEST=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/manifest.json
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
RESULT="$OUT/frozen_cc_hybrid_${WINDOW}_T${STEPS}_settle${SETTLE}.json"
test -s "$CHECKPOINT" && test -s "$GAMMA" && test -s "$MANIFEST" && test -s "$HDF5"
test -s "$OUT/SPATIAL_EVALUATED"
if [[ -e "$RESULT" || -e "$OUT/frozen_cc_hybrid_${WINDOW}.log" ]]; then
  echo "Refusing to overwrite prior SW0051 output: $RESULT" >&2; exit 3
fi
check_gpu_idle() {
  local pids
  if ! pids="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
    echo "Unable to query GPU $GPU_ID: $pids" >&2; return 2
  fi
  if [[ "$pids" =~ [0-9] ]]; then echo "GPU $GPU_ID is occupied; refusing overlapping inference" >&2; return 1; fi
}
check_gpu_idle
"$(dirname "$0")/preflight.sh" "$GPU_ID" "$SEED"
check_gpu_idle
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache" CUDA_VISIBLE_DEVICES="$GPU_ID"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0051_frozen_cc_spectral_hybrid/evaluate.py" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-manifest "$MANIFEST" \
  --dataset-path "$HDF5" --output-path "$RESULT" --global-start 1320 --count 320 \
  --steps "$STEPS" --settle "$SETTLE" --batch-size 4 --device cuda \
  > "$OUT/frozen_cc_hybrid_${WINDOW}.log" 2>&1
