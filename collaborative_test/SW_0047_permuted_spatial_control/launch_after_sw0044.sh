#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Pass the GPU reserved for this evaluation}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s0"
CHECKPOINT="$OUT/core.pt"
SHORT_BASE="$OUT/validation_short_T256_settle64.json"
LONG_BASE="$OUT/validation_long_T1024_settle512.json"
SPATIAL_MARKER="$OUT/SPATIAL_EVALUATED"
SHORT_RESULT="$OUT/spike_spatial_permuted_short.json"
LONG_RESULT="$OUT/spike_spatial_permuted_long.json"
MARKER="$OUT/PERMUTED_SPATIAL_EVALUATED"
EVALUATE="$ROOT/collaborative_test/SW_0047_permuted_spatial_control/evaluate.sh"

refuse_existing() {
  if [[ -e "$SHORT_RESULT" || -e "$LONG_RESULT" || -e "$MARKER" ]]; then
    echo "Refusing to overwrite SW0047 result or marker under $OUT" >&2
    exit 1
  fi
}
gpu_has_compute_process() {
  local pids
  if ! pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>&1)"; then
    echo "Unable to query GPU $GPU_ID: $pids" >&2
    return 2
  fi
  [[ "$pids" =~ [0-9] ]]
}

while true; do
  refuse_existing
  if [[ -s "$CHECKPOINT" && -s "$SHORT_BASE" && -s "$LONG_BASE" && -s "$SPATIAL_MARKER" ]]; then
    if gpu_has_compute_process; then
      echo "GPU $GPU_ID has compute processes; waiting 20 seconds."
    else
      status=$?
      if [[ "$status" -eq 2 ]]; then exit 2; fi
      refuse_existing
      if gpu_has_compute_process; then
        echo "GPU $GPU_ID became busy before launch; waiting 20 seconds."
      else
        status=$?
        if [[ "$status" -eq 2 ]]; then exit 2; fi
        break
      fi
    fi
  else
    echo "Waiting for SW0042 seed0 short/long baselines and SW0044 spatial marker."
  fi
  sleep 20
done

bash "$EVALUATE" "$GPU_ID" "$CHECKPOINT" "$SHORT_RESULT" short
bash "$EVALUATE" "$GPU_ID" "$CHECKPOINT" "$LONG_RESULT" long
printf 'completed\n' > "$MARKER"
echo "SW0047 completed: $SHORT_RESULT, $LONG_RESULT"
