#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
if [[ "$MODE" != full && "$MODE" != smoke ]]; then
  echo "MODE must be full or smoke" >&2; exit 2
fi
ROOT=/Data0/kevinswk/patch_v2_sw
RUN_DIR="$ROOT/collaborative_test/SW_0048_matched_slot_attention"
LOG_DIR="$ROOT/trained_models/SW_0048_parallel_logs"
mkdir -p "$LOG_DIR"
LOCK_DIR="$LOG_DIR/.launch_lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another SW0048 parallel launcher owns $LOCK_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
for seed in 0 1 2; do
  out="$ROOT/trained_models/SW_0048_slot_attention_seed${seed}"
  [[ "$MODE" == full ]] || out="$out-smoke"
  if [[ -e "$out" || -e "$LOG_DIR/seed${seed}.log" ]]; then
    echo "Refusing parallel launch: output/log already exists for seed $seed" >&2
    exit 1
  fi
done
pids=()
for seed in 0 1 2; do
  bash "$RUN_DIR/run_seed.sh" "$seed" "$MODE" > "$LOG_DIR/seed${seed}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
