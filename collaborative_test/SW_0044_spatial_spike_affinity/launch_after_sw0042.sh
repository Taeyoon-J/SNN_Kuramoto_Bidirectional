#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Pass the GPU ID reserved for this evaluation}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s0"
CHECKPOINT="$OUT/core.pt"
SHORT_PREREQ="$OUT/validation_short_T256_settle64.json"
LONG_PREREQ="$OUT/validation_long_T1024_settle512.json"
SHORT_RESULT="$OUT/spatial_affinity_short.json"
LONG_RESULT="$OUT/spatial_affinity_long.json"
MARKER="$OUT/SPATIAL_EVALUATED"
EVALUATE="$ROOT/collaborative_test/SW_0044_spatial_spike_affinity/evaluate.sh"

if [[ -e "$SHORT_RESULT" || -e "$LONG_RESULT" || -e "$MARKER" ]]; then
  echo "Refusing to overwrite existing SW0044 output or marker under $OUT" >&2
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
  if [[ -e "$SHORT_RESULT" || -e "$LONG_RESULT" || -e "$MARKER" ]]; then
    echo "Refusing to overwrite existing SW0044 output or marker under $OUT" >&2
    exit 1
  fi
  if [[ -s "$CHECKPOINT" && -s "$SHORT_PREREQ" && -s "$LONG_PREREQ" ]]; then
    if gpu_has_compute_process; then
      echo "GPU $GPU_ID has compute processes; waiting 20 seconds."
    else
      status=$?
      if [[ "$status" -eq 2 ]]; then exit 2; fi
      if [[ -s "$CHECKPOINT" && -s "$SHORT_PREREQ" && -s "$LONG_PREREQ" \
          && ! -e "$SHORT_RESULT" && ! -e "$LONG_RESULT" && ! -e "$MARKER" ]]; then
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
    echo "Waiting for SW0042 seed0 checkpoint and both validation results."
  fi
  sleep 20
done

bash "$EVALUATE" "$GPU_ID" "$CHECKPOINT" "$SHORT_RESULT" short
bash "$EVALUATE" "$GPU_ID" "$CHECKPOINT" "$LONG_RESULT" long
printf 'completed\n' > "$MARKER"
echo "SW0044 spatial affinity evaluations completed: $SHORT_RESULT, $LONG_RESULT"
