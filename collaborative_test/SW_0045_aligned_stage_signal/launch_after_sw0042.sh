#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Pass a GPU ID for the diagnostic}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s0"
CHECKPOINT="$OUT/core.pt"
SHORT_PREREQ="$OUT/validation_short_T256_settle64.json"
LONG_PREREQ="$OUT/validation_long_T1024_settle512.json"
SPATIAL_PREREQ="$OUT/SPATIAL_EVALUATED"
RESULT="$OUT/stage_signal_validation320.json"
MARKER="$OUT/STAGE_SIGNAL_DIAGNOSED"
GAMMA="$ROOT/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt"
GAMMA_MANIFEST="$ROOT/data/SW_0042_hdf5_aligned/manifest.json"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
DIAGNOSE="$ROOT/collaborative_test/SW_0034_stage_signal_diagnostic/diagnose.py"

if [[ -e "$RESULT" || -e "$MARKER" ]]; then
  echo "Refusing to overwrite existing stage diagnostic output or marker: $OUT" >&2
  exit 1
fi

gpu_has_compute_process() {
  local pids
  if ! pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>&1)"; then
    echo "Unable to query GPU $GPU_ID: $pids" >&2
    return 2
  fi
  [[ "$pids" =~ [0-9] ]]
}

while true; do
  if [[ -e "$RESULT" || -e "$MARKER" ]]; then
    echo "Refusing to overwrite existing stage diagnostic output or marker: $OUT" >&2
    exit 1
  fi
  if [[ -s "$CHECKPOINT" && -s "$SHORT_PREREQ" && -s "$LONG_PREREQ" \
      && -s "$SPATIAL_PREREQ" \
      && -s "$GAMMA" && -s "$GAMMA_MANIFEST" && -s "$HDF5" ]]; then
    if gpu_has_compute_process; then
      echo "GPU $GPU_ID has compute processes; waiting 20 seconds."
    else
      status=$?
      if [[ "$status" -eq 2 ]]; then exit 2; fi
      if [[ -s "$CHECKPOINT" && -s "$SHORT_PREREQ" && -s "$LONG_PREREQ" \
          && -s "$SPATIAL_PREREQ" \
          && -s "$GAMMA" && -s "$GAMMA_MANIFEST" && -s "$HDF5" \
          && ! -e "$RESULT" && ! -e "$MARKER" ]]; then
        if gpu_has_compute_process; then
          echo "GPU $GPU_ID became busy before launch; waiting 20 seconds."
        else
          status=$?
          if [[ "$status" -eq 2 ]]; then exit 2; fi
          break
        fi
      fi
    fi
  else
    echo "Waiting for SW0042 seed0 core, both base evaluations, SW0044 completion, and aligned inputs."
  fi
  sleep 20
done

mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache"
export TRITON_CACHE_DIR="$OUT/cache"
CUDA_VISIBLE_DEVICES="$GPU_ID" /Data0/kevinswk/envs/snn/bin/python -u "$DIAGNOSE" \
  --checkpoint "$CHECKPOINT" --gamma-path "$GAMMA" --gamma-global-start 1320 \
  --gamma-manifest "$GAMMA_MANIFEST" --dataset-path "$HDF5" \
  --output-path "$RESULT" --start 1320 --count 320 --batch-size 4 \
  --steps 256 --settle 64 --membrane-vth 0.06 --dendritic-projection shared \
  --graph-spatial-decay 0.35 --geodesic-steps 3 --geodesic-radius 1.5 \
  --geodesic-contrast 2.0 --geodesic-temperature 0.5 --geodesic-cap 16 \
  --kuramoto-backend factorized --device cuda \
  > "$OUT/stage_signal_validation320.log" 2>&1
test -s "$RESULT"
printf 'completed\n' > "$MARKER"
echo "SW0045 stage diagnostic completed: $RESULT"
