#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass a GPU to queue on}"
TARGET_RATIO="${2:?Choose 0.1 or 1.0 baseline-gradient target}"
case "$TARGET_RATIO" in 0.1) ;; *) echo "Only the 0.1x measured-gradient diversity arm is scheduled here; 1x is held." >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0050_sample_diversity"
BASE_OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s2"
PROBE="$BASE_OUT/sample_diversity_gradient_probe.json"
PROBE_LOCK="$BASE_OUT/.sample_diversity_probe_lock"
QUEUE_LOG="$ROOT/trained_models/SW_0050_seed2_ratio${TARGET_RATIO}_full_queue.log"
if [[ -e "$QUEUE_LOG" ]]; then echo "Refusing duplicate queue log: $QUEUE_LOG" >&2; exit 1; fi
while [[ ! -s "$BASE_OUT/core.pt" || ! -s /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt ]]; do
  echo "Waiting for SW0042 seed2 checkpoint and aligned training gamma."
  sleep 20
done
gpu_has_compute_process() {
  local pids
  if ! pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>&1)"; then
    echo "Unable to query GPU $GPU_ID: $pids" >&2; return 2
  fi
  [[ "$pids" =~ [0-9] ]]
}
while true; do
  if gpu_has_compute_process; then
    echo "GPU $GPU_ID is occupied; waiting 20 seconds."
  else
    status=$?
    if [[ "$status" -eq 2 ]]; then exit 2; fi
    if gpu_has_compute_process; then
      echo "GPU $GPU_ID became busy; waiting 20 seconds."
    else
      status=$?
      if [[ "$status" -eq 2 ]]; then exit 2; fi
      break
    fi
  fi
  sleep 20
done
bash "$DIR/preflight.sh" "$GPU_ID"
if gpu_has_compute_process; then echo "GPU $GPU_ID became busy after preflight; refusing to overlap training" >&2; exit 2; else status=$?; [[ "$status" -eq 1 ]] || exit 2; fi
while [[ ! -s "$PROBE" ]]; do
  if mkdir "$PROBE_LOCK" 2>/dev/null; then
    trap 'rmdir "$PROBE_LOCK" 2>/dev/null || true' EXIT
    if [[ ! -s "$PROBE" ]]; then
      bash "$DIR/probe.sh" "$GPU_ID"
    fi
    rmdir "$PROBE_LOCK"
    trap - EXIT
  else
    echo "Another queued seed2 job is measuring baseline gradient scale; waiting 20 seconds."
    sleep 20
  fi
done
WEIGHT="$(/Data0/kevinswk/envs/snn/bin/python - "$PROBE" "$TARGET_RATIO" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
key = "target_0.1x_baseline_gradient" if sys.argv[2] == "0.1" else "target_1.0x_baseline_gradient"
print(format(float(data["suggested_weights"][key]), ".12g"))
PY
)"
TAG="${WEIGHT//./p}"
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s2_w${TAG}_lr0p001_full"
TRAIN_LOG="$ROOT/trained_models/SW_0050_seed2_w${TAG}_full_queue.log"
if [[ -e "$OUT" || -e "$TRAIN_LOG" ]]; then
  echo "Refusing to overwrite existing seed2 full output/log for weight $WEIGHT" >&2
  exit 1
fi
bash "$DIR/run.sh" "$GPU_ID" 2 "$WEIGHT" 0.001 full > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "Started seed2 full run PID=$TRAIN_PID weight=$WEIGHT target_ratio=$TARGET_RATIO"
bash "$DIR/watch_and_evaluate.sh" "$GPU_ID" 2 "$WEIGHT" 0.001 full "$TRAIN_PID"
