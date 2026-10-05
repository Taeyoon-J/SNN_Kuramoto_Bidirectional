#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0055_unique_data_scale"
SW54="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation"
STATE="$ROOT/trained_models/SW0055_unique2500_autolaunch"
LOCK="$STATE/LOCK"
DONE="$STATE/LAUNCH_COMPLETED"
LOG="$STATE/$(date -u +%Y%m%dT%H%M%SZ)_wait_and_launch.log"
PILOT_RESULT="$ROOT/trained_models/SW_0054_causal_mechanism_ablation/seed0_epoch25_short_n32_pilot_T256_settle64.json"
PILOT_LOG="$ROOT/trained_models/SW_0054_causal_mechanism_ablation/seed0_epoch25_short_n32_pilot.log"

mkdir -p "$STATE"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "Another SW0055 wait/launch wrapper holds $LOCK" >&2
  exit 2
fi
printf '%s\n' "pid=$$ started_utc=$(date -u +%FT%TZ)" > "$LOCK/owner"
cleanup() { rm -rf -- "$LOCK"; }
trap cleanup EXIT INT TERM
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$DONE" ]]; then
  echo "Completion marker already exists; refusing relaunch: $DONE"
  exit 3
fi
for seed in 0 1 2; do
  out="$ROOT/trained_models/SW0055_unique2500_s${seed}_e10_lr0p0003"
  if [[ -e "$out" ]]; then
    echo "Existing SW0055 seed output found; refusing restart: $out"
    exit 3
  fi
done

gpu_idle() {
  local gpu pids
  gpu="$1"
  if ! pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then
    echo "GPU $gpu query failed: $pids" >&2
    return 2
  fi
  [[ ! "$pids" =~ [0-9] ]]
}

all_gpus_idle() {
  gpu_idle 1 && gpu_idle 2 && gpu_idle 3
}

validate_pilot() {
  /Data0/kevinswk/envs/snn/bin/python - "$PILOT_RESULT" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1], encoding='utf-8'))
assert r['target']['ground_truth_used_for_prediction'] is False
assert r['count']==32
conditions=r['conditions']
assert set(conditions)=={'normal','gate_perm_s0','gate_perm_s1','gate_perm_s2',
 'carrier_perm_s0','carrier_perm_s1','carrier_perm_s2','gate_mean','carrier_mean','K0'}
for condition in conditions.values():
    for metrics in condition['fixed_readouts'].values():
        assert all(math.isfinite(float(v)) for v in metrics['metrics'].values())
PY
}

if [[ -e "$PILOT_RESULT" ]]; then
  [[ -s "$PILOT_LOG" ]] || { echo "Pilot result exists without its log; refusing to infer success"; exit 4; }
  validate_pilot || { echo "Existing SW0054 pilot failed validation; refusing overwrite/retry"; exit 4; }
  echo "Validated existing SW0054 pilot; skipping it."
elif [[ -e "$PILOT_LOG" ]]; then
  echo "SW0054 pilot log exists without a result; treating as failed run, no automatic retry: $PILOT_LOG"
  exit 4
else
  pilot_done=0
  while (( pilot_done == 0 )); do
    for gpu in 1 2 3; do
      if gpu_idle "$gpu"; then
        echo "$(date -u +%FT%TZ) GPU $gpu idle; starting SW0054 preflight + count32 pilot"
        if bash "$SW54/preflight.sh" "$gpu"; then
          if bash "$SW54/evaluate.sh" "$gpu" short 32 pilot; then
            validate_pilot || { echo "Pilot output failed validation; stop without retry"; exit 4; }
            pilot_done=1
            break
          else
            rc=$?
            if [[ -e "$PILOT_RESULT" || -e "$PILOT_LOG" ]]; then
              echo "SW0054 evaluator failed (status $rc) and left result/log artifacts; stop without retry"
              exit 4
            fi
            if gpu_idle "$gpu"; then
              echo "Pilot command exited $rc while GPU $gpu is idle and left no artifacts; treating as a command failure"
              exit "$rc"
            fi
            echo "GPU $gpu became occupied or unavailable before pilot launch; safe to retry after 30 seconds"
          fi
        else
          rc=$?
          if [[ -e "$PILOT_RESULT" || -e "$PILOT_LOG" ]]; then
            echo "SW0054 preflight failed after evaluator artifacts appeared; stop without retry"
            exit 4
          fi
          if gpu_idle "$gpu"; then
            echo "Preflight failed with status $rc while GPU $gpu is idle; treating as a real preflight error"
            exit "$rc"
          fi
          echo "GPU $gpu became occupied or unavailable during preflight; safe to retry after 30 seconds"
        fi
      fi
    done
    (( pilot_done == 1 )) || sleep 30
  done
fi

echo "$(date -u +%FT%TZ) SW0054 pilot validated; waiting for GPUs 1,2,3 to be idle"
until all_gpus_idle; do sleep 30; done
# Recheck after the final poll immediately before the launcher performs its own
# preflight and launch-time occupancy checks.
all_gpus_idle || { echo "GPU occupancy changed at launch boundary; retrying wait"; sleep 30; until all_gpus_idle; do sleep 30; done; }
for seed in 0 1 2; do
  out="$ROOT/trained_models/SW0055_unique2500_s${seed}_e10_lr0p0003"
  [[ ! -e "$out" ]] || { echo "SW0055 output appeared while waiting; refusing launch: $out"; exit 3; }
done
if bash "$DIR/launch_parallel.sh" 1 2 3; then
  printf '%s\n' "launched_utc=$(date -u +%FT%TZ)" "pilot=$PILOT_RESULT" > "$DONE"
  echo "SW0055 seeds 0/1/2 launched; marker written: $DONE"
else
  rc=$?
  for seed in 0 1 2; do
    out="$ROOT/trained_models/SW0055_unique2500_s${seed}_e10_lr0p0003"
    if [[ -e "$out" ]]; then
      echo "Launcher failed with status $rc after output appeared; refusing automatic retry"
      exit "$rc"
    fi
  done
  echo "Launcher failed with status $rc before outputs; stop for manual review (no retry)"
  exit "$rc"
fi
