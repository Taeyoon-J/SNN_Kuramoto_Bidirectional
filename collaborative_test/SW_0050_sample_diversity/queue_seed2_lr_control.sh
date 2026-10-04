#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?Pass a GPU to queue on}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0050_sample_diversity"
OUT="$ROOT/trained_models/SW_0050_sample_diversity_s2_w0_lr0p0003_full"
QUEUE_LOG="$ROOT/trained_models/SW_0050_seed2_lr3e-4_control_queue.log"
if [[ -e "$OUT" || -e "$QUEUE_LOG" ]]; then echo "Refusing to overwrite/no duplicate LR-control output: $OUT" >&2; exit 1; fi
while [[ ! -s /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt ]]; do
  echo "Waiting for aligned training gamma."
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
bash "$DIR/run.sh" "$GPU_ID" 2 0 0.0003 full > "$QUEUE_LOG" 2>&1 &
TRAIN_PID=$!
bash "$DIR/watch_and_evaluate.sh" "$GPU_ID" 2 0 0.0003 full "$TRAIN_PID"
