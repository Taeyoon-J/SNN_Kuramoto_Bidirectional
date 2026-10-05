#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"
DIR=/Data0/kevinswk/patch_v2_sw/collaborative_test/SW_0056_matched_slot_2500
case "$MODE" in
  full) exec bash "$DIR/wait_for_low_load_and_run.sh" ;;
  dry-run) exec bash "$DIR/wait_for_low_load_and_run.sh" --dry-run ;;
  *) echo "Only the guarded full runner or dry-run is supported here; use run_seed.sh SEED smoke for isolated plumbing checks." >&2; exit 2 ;;
esac
