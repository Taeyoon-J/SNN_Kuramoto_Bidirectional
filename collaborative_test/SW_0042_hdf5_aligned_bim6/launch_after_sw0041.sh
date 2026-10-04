#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
EXP="$ROOT/collaborative_test/SW_0042_hdf5_aligned_bim6"
mkdir -p "$ROOT/trained_models"
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for duplicate-launch protection." >&2
  exit 1
fi
exec 9>"$ROOT/trained_models/.SW_0042_HDF5_aligned_BIM6.launch.lock"
if ! flock -n 9; then
  echo "Another SW0042 launcher is active; refusing duplicate launch/wait." >&2
  exit 1
fi
DRY_RUN=0
WAIT_MODE=0
WAIT_TIMEOUT=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --wait) WAIT_MODE=1; shift ;;
    --wait-timeout-seconds)
      [[ "$#" -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        echo "--wait-timeout-seconds requires nonnegative seconds" >&2; exit 2;
      }
      WAIT_TIMEOUT="$2"; shift 2 ;;
    *) echo "Usage: bash launch_after_sw0041.sh [--dry-run] [--wait] [--wait-timeout-seconds N]" >&2; exit 2 ;;
  esac
done
if [[ "$WAIT_TIMEOUT" -gt 0 && "$WAIT_MODE" -ne 1 ]]; then
  echo "A timeout is valid only with --wait" >&2
  exit 2
fi

TRAIN_GAMMA=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/gamma_train.pt
VAL_GAMMA=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/gamma_validation_1320_1639.pt
VAL_MANIFEST=/Data0/kevinswk/patch_v2_sw/data/SW_0042_hdf5_aligned/manifest.json
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
for input in "$TRAIN_GAMMA" "$VAL_GAMMA" "$VAL_MANIFEST" "$HDF5"; do
  if [[ ! -s "$input" ]]; then echo "Required input is missing: $input" >&2; exit 1; fi
done

check_outputs_available() {
  for seed in 0 1 2; do
    out="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${seed}"
    if [[ -e "$out/core.pt" || -e "$out/training.log" || \
          -e "$out/training.pid" || -e "$out/watcher.pid" ]]; then
      echo "Refusing to overwrite existing SW0042 output: $out" >&2
      return 1
    fi
  done
}

check_ready() {
  for seed in 0 1 2; do
    prior="$ROOT/trained_models/SW_0041_BIM6_s${seed}"
    if [[ ! -s "$prior/our_validation.json" || \
          ! -s "$prior/peer_validation.json" || \
          ! -s "$prior/peer_long.json" ]] || \
          ! grep -Fq "Completed peer_long" "$prior/watch.log" 2>/dev/null; then
      READY_REASON="SW0041 seed $seed watcher/evaluations are incomplete"
      return 1
    fi
  done
  for gpu in 0 1 2; do
    if ! gpu_processes="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"; then
      READY_REASON="Could not inspect GPU $gpu"
      return 1
    fi
    if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
      READY_REASON="GPU $gpu has compute process(es): ${gpu_processes//$'\n'/, }"
      return 1
    fi
  done
  READY_REASON=""
  return 0
}

check_outputs_available
elapsed=0
while ! check_ready; do
  if [[ "$WAIT_MODE" -ne 1 ]]; then
    echo "$READY_REASON; use --wait to poll safely." >&2
    exit 1
  fi
  if [[ "$WAIT_TIMEOUT" -gt 0 && "$elapsed" -ge "$WAIT_TIMEOUT" ]]; then
    echo "Timed out after ${elapsed}s: $READY_REASON. Nothing was launched." >&2
    exit 1
  fi
  echo "Waiting (20s poll): $READY_REASON"
  sleep 20
  elapsed=$((elapsed + 20))
  check_outputs_available
done

# Recheck all gates immediately before launch; a timeout or failed check never
# falls through as ready, and existing processes are never signalled.
check_outputs_available
if ! check_ready; then
  echo "$READY_REASON; no jobs were started." >&2
  exit 1
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Ready to launch SW0042 seeds 0/1/2 on GPUs 0/1/2 using %s\n' "$TRAIN_GAMMA"
  exit 0
fi

for seed in 0 1 2; do
  gpu="$seed"
  out="$ROOT/trained_models/SW_0042_HDF5_aligned_BIM6_s${seed}"
  mkdir -p "$out"
  nohup bash "$EXP/run.sh" "$gpu" "$seed" > "$out/launcher.log" 2>&1 < /dev/null &
  train_pid=$!
  printf '%s\n' "$train_pid" > "$out/training.pid"
  nohup bash "$EXP/watch_and_evaluate.sh" "$gpu" "$seed" "$train_pid" \
    > "$out/watch_launcher.log" 2>&1 < /dev/null &
  watcher_pid=$!
  printf '%s\n' "$watcher_pid" > "$out/watcher.pid"
  printf 'Started seed %s on GPU %s (training PID %s, watcher PID %s)\n' \
    "$seed" "$gpu" "$train_pid" "$watcher_pid"
done
