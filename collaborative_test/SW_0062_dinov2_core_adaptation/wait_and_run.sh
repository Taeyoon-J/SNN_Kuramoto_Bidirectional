#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0062_dinov2_core_adaptation"
DATA="$ROOT/data/SW_0062_dinov2_core_adaptation"
STATE="$ROOT/trained_models/SW0062_dinov2_scheduler"
PRIOR_DONE="$ROOT/trained_models/SW0057_fixed_multireadout_scheduler/LAUNCH_COMPLETED"
mkdir -p "$STATE"
[[ ! -e "$STATE/LAUNCH_COMPLETED" && ! -e "$STATE/RUNNING" ]] || { echo "existing scheduler state" >&2; exit 3; }
printf 'pid=%s\n' "$$" > "$STATE/RUNNING"
trap 'rm -f "$STATE/RUNNING"' EXIT
until [[ -s "$DATA/manifest.json" && -s "$DATA/dino_gamma_train_0_999.pt" && -s "$DATA/dino_gamma_validation_1320_1639.pt" ]]; do sleep 30; done
until [[ -s "$PRIOR_DONE" ]]; do sleep 30; done
gpu_idle() {
  local pids
  pids="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" || return 2
  [[ ! "$pids" =~ [0-9] ]]
}
until gpu_idle; do rc=$?; [[ $rc -eq 1 ]] || exit 4; sleep 30; done
bash "$DIR/preflight.sh" 0
until gpu_idle; do rc=$?; [[ $rc -eq 1 ]] || exit 4; sleep 30; done
bash "$DIR/run_pilot.sh" 0
until gpu_idle; do rc=$?; [[ $rc -eq 1 ]] || exit 4; sleep 30; done
bash "$DIR/evaluate_pilot.sh" 0
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/LAUNCH_COMPLETED"
echo "SW0062 direction pilot complete"
