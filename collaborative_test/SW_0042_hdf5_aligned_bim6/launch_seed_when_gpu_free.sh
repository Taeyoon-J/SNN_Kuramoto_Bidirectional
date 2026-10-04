#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?Pass assigned GPU ID}"
SEED="${2:?Pass seed 0, 1, or 2}"
PREREQUISITE="${3:?Pass prerequisite marker path}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac

ROOT=/Data0/kevinswk/patch_v2_sw
EXP="$ROOT/collaborative_test/SW_0042_hdf5_aligned_bim6"
OUT="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${SEED}"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', ' -v gpu="$GPU_ID" '$1 == gpu {print $2}')"
test -n "$GPU_UUID"

while true; do
  if [[ -s "$PREREQUISITE" ]] && ! nvidia-smi \
      --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$GPU_UUID"; then
    break
  fi
  sleep 20
done

if [[ -e "$OUT/core.pt" || -e "$OUT/training.log" ]]; then
  echo "Refusing to overwrite existing SW0042 output: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
nohup bash "$EXP/run.sh" "$GPU_ID" "$SEED" > "$OUT/launcher.log" 2>&1 < /dev/null &
TRAIN_PID=$!
nohup bash "$EXP/watch_and_evaluate.sh" "$GPU_ID" "$SEED" "$TRAIN_PID" \
  > "$OUT/watcher_launcher.log" 2>&1 < /dev/null &
printf 'Started seed %s on GPU %s (training PID %s, watcher PID %s)\n' \
  "$SEED" "$GPU_ID" "$TRAIN_PID" "$!"
