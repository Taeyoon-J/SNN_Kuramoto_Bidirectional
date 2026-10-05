#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0055_unique_data_scale"
SW54="$ROOT/collaborative_test/SW_0054_causal_mechanism_ablation"
STATE="$ROOT/trained_models/SW0055_unique2500_autolaunch"
LOCK="$STATE/LOCK"
PID_FILE="$STATE/wrapper.pid"
DONE="$STATE/LAUNCH_COMPLETED"
START_TIME="$(date -u +%FT%TZ)"
LOG="$STATE/$(date -u +%Y%m%dT%H%M%SZ)_wait_and_launch.log"
PILOT_RESULT="$ROOT/trained_models/SW_0054_causal_mechanism_ablation/seed0_epoch25_short_n32_pilot_T256_settle64.json"
PILOT_LOG="$ROOT/trained_models/SW_0054_causal_mechanism_ablation/seed0_epoch25_short_n32_pilot.log"

mkdir -p "$STATE"

# Treat a live or ambiguous PID as active. Remove stale state only after
# confirming the recorded numeric PID is dead and LOCK is a real directory.
pid_check() {
  local pid cmd
  pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 2
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/cmdline" ]]; then
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$cmd" == *wait_for_idle_and_launch.sh* ]] || return 2
  fi
  return 0
}
if [[ -e "$LOCK" ]]; then
  [[ -d "$LOCK" && ! -L "$LOCK" ]] || { echo "Unsafe lock path: $LOCK" >&2; exit 2; }
  prior=""
  if [[ -s "$LOCK/owner" ]]; then
    prior="$(sed -n 's/^pid=\([0-9][0-9]*\).*/\1/p' "$LOCK/owner" | head -n 1)"
  elif [[ -s "$PID_FILE" ]]; then
    prior="$(cat "$PID_FILE")"
  fi
  [[ "$prior" =~ ^[0-9]+$ ]] || { echo "Ambiguous lock owner; refusing cleanup" >&2; exit 2; }
  if pid_check "$prior"; then
    echo "Scheduler PID $prior is active; refusing duplicate" >&2; exit 2
  else
    rc=$?
    [[ $rc -eq 1 ]] || { echo "Cannot prove lock owner stale; refusing cleanup" >&2; exit 2; }
  fi
  echo "Removing verified stale lock for dead PID $prior"
  rm -rf -- "$LOCK"
  [[ ! -f "$PID_FILE" || "$(cat "$PID_FILE")" != "$prior" ]] || rm -f -- "$PID_FILE"
elif [[ -e "$PID_FILE" ]]; then
  prior="$(cat "$PID_FILE")"
  if pid_check "$prior"; then
    echo "wrapper.pid is active but LOCK is absent; refusing" >&2; exit 2
  else
    rc=$?
    [[ $rc -eq 1 ]] || { echo "Malformed/ambiguous wrapper.pid; refusing cleanup" >&2; exit 2; }
    rm -f -- "$PID_FILE"
  fi
fi
mkdir "$LOCK" 2>/dev/null || { echo "Could not acquire scheduler lock" >&2; exit 2; }
printf 'pid=%s started_utc=%s\n' "$$" "$START_TIME" > "$LOCK/owner"
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
  if [[ -f "$LOCK/owner" ]] && grep -Fxq "pid=$$ started_utc=$START_TIME" "$LOCK/owner"; then rm -rf -- "$LOCK"; fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$$" ]]; then rm -f -- "$PID_FILE"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec > >(tee -a "$LOG") 2>&1

if [[ -e "$DONE" ]]; then echo "Completion marker exists; refusing relaunch"; exit 3; fi
for marker in PILOT_STARTED PILOT_COMPLETED SEEDS01_STARTED SEEDS01_COMPLETED SEED2_STARTED; do
  [[ ! -e "$STATE/$marker" ]] || { echo "Prior partial state $marker; manual review required"; exit 3; }
done
seed_out() { printf '%s/trained_models/SW0055_unique2500_s%s_e10_lr0p0003' "$ROOT" "$1"; }
for seed in 0 1 2; do out="$(seed_out "$seed")"; [[ ! -e "$out" ]] || { echo "Existing output; refusing restart: $out"; exit 3; }; done

gpu_idle() {
  local gpu pids
  gpu="$1"
  [[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "Only GPU 0/1 permitted" >&2; return 2; }
  if ! pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1)"; then echo "GPU $gpu query failed: $pids" >&2; return 2; fi
  [[ ! "$pids" =~ [0-9] ]]
}
both_idle() { gpu_idle 0 && gpu_idle 1; }
validate_pilot() {
  /Data0/kevinswk/envs/snn/bin/python - "$PILOT_RESULT" <<'PY'
import json, math, sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r['target']['ground_truth_used_for_prediction'] is False
assert r['count']==32 and r['ids']==[1320,1351]
assert r['core']['steps']==256 and r['core']['settle']==64
assert r['interventions']['condition_set']=='pilot'
c=r['conditions']
assert set(c)=={'normal','gate_perm_s0','carrier_perm_s0','gate_mean','carrier_mean','K0'}
for name,row in c.items():
 for readout in row['fixed_readouts'].values():
  assert all(math.isfinite(float(v)) for v in readout['metrics'].values())
 if name.startswith(('gate_','carrier_')):
  assert row['gate_invariants'] and all(x['graph_bitwise_equal'] and x['theta_allclose'] for x in row['gate_invariants'])
assert c['K0']['kuramoto_K_during_rollout']==0.0
PY
}

if [[ -e "$PILOT_RESULT" ]]; then
  [[ -s "$PILOT_LOG" ]] || { echo "Pilot result lacks log; refusing"; exit 4; }
  validate_pilot || { echo "Existing pilot invalid; refusing retry"; exit 4; }
  echo "Existing SW0054 pilot validated; skipping."
else
  [[ ! -e "$PILOT_LOG" ]] || { echo "Pilot log without result; manual review required"; exit 4; }
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/PILOT_STARTED"
  while true; do
    gpu=""
    for candidate in 0 1; do if gpu_idle "$candidate"; then gpu="$candidate"; break; fi; done
    if [[ -z "$gpu" ]]; then sleep 30; continue; fi
    echo "$(date -u +%FT%TZ) GPU $gpu free; SW0054 preflight/pilot"
    if bash "$SW54/preflight.sh" "$gpu"; then
      if gpu_idle "$gpu" && bash "$SW54/evaluate.sh" "$gpu" short 32 pilot; then
        validate_pilot || { echo "Pilot validation failed; no retry"; exit 4; }
        break
      fi
      if [[ -e "$PILOT_RESULT" || -e "$PILOT_LOG" ]]; then echo "Pilot evaluator artifact indicates failure; no retry"; exit 4; fi
      if gpu_idle "$gpu"; then echo "Pilot failed while GPU idle; abort"; exit 4; fi
      echo "GPU claimed during pilot start; retrying after 30s"
    else
      rc=$?
      if gpu_idle "$gpu"; then echo "Preflight failed on idle GPU (status $rc); abort"; exit "$rc"; fi
      echo "GPU claimed during preflight; retrying after 30s"
    fi
    sleep 30
  done
fi
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/PILOT_COMPLETED"

echo "$(date -u +%FT%TZ) Pilot passed; waiting for both allowed GPUs"
until both_idle; do sleep 30; done
for seed in 0 1; do out="$(seed_out "$seed")"; [[ ! -e "$out" ]] || { echo "Seed $seed output appeared; refusing"; exit 3; }; done
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEEDS01_STARTED"

# The SW0055 preflight uses GPU0. Recheck both devices afterwards and before
# starting either seed so occupancy changes never silently create a partial set.
while true; do
  until both_idle; do sleep 30; done
  if bash "$DIR/preflight_training.sh" 0; then
    if both_idle; then break; fi
  else
    rc=$?
    if both_idle; then echo "SW0055 preflight failed while both GPUs are idle; abort"; exit "$rc"; fi
  fi
  echo "An allowed GPU was claimed during preflight; waiting to retry safely"
  sleep 30
done
both_idle || { echo "GPU occupancy changed at launch boundary; aborting"; exit 5; }
for seed in 0 1; do out="$(seed_out "$seed")"; [[ ! -e "$out" ]] || { echo "Existing seed output; refusing: $out"; exit 3; }; done

launch_seed() {
  local gpu seed log pidfile pid
  gpu="$1"; seed="$2"
  log="$ROOT/trained_models/SW0055_unique2500_s${seed}_launcher.log"
  pidfile="$ROOT/trained_models/SW0055_unique2500_s${seed}_launcher.pid"
  [[ ! -e "$log" && ! -e "$pidfile" ]] || { echo "Existing seed launcher log/PID for seed $seed; refusing"; return 3; }
  nohup bash "$DIR/train_evaluate.sh" "$gpu" "$seed" > "$log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" > "$pidfile"
  printf '%s\n' "$pid" > "$STATE/seed${seed}.pid"
  echo "$(date -u +%FT%TZ) launched seed=$seed gpu=$gpu pid=$pid log=$log"
}

launch_seed 0 0
launch_seed 1 1
pid0="$(cat "$STATE/seed0.pid")"
pid1="$(cat "$STATE/seed1.pid")"
rc0=0; rc1=0
if wait "$pid0"; then :; else rc0=$?; fi
if wait "$pid1"; then :; else rc1=$?; fi
for seed in 0 1; do
  out="$(seed_out "$seed")"
  if [[ ! -s "$out/EVALUATION_COMPLETED" || ! -s "$out/core.pt" || \
        ! -s "$out/validation_short_T256_settle64.json" || \
        ! -s "$out/validation_long_T1024_settle512.json" ]]; then
    echo "Seed $seed incomplete; retaining partial output and refusing restart"; exit 6
  fi
done
[[ $rc0 -eq 0 && $rc1 -eq 0 ]] || { echo "Seed exit statuses $rc0/$rc1; seed2 will not start"; exit 6; }
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEEDS01_COMPLETED"

echo "$(date -u +%FT%TZ) Seeds 0/1 passed; waiting for GPU0 for seed2"
until gpu_idle 0; do sleep 30; done
out2="$(seed_out 2)"
[[ ! -e "$out2" ]] || { echo "Seed2 output appeared; refusing: $out2"; exit 3; }
printf 'started_utc=%s gpu=0\n' "$(date -u +%FT%TZ)" > "$STATE/SEED2_STARTED"
log2="$ROOT/trained_models/SW0055_unique2500_s2_launcher.log"
pidfile2="$ROOT/trained_models/SW0055_unique2500_s2_launcher.pid"
[[ ! -e "$log2" && ! -e "$pidfile2" ]] || { echo "Existing seed2 launcher log/PID; refusing"; exit 3; }
nohup bash "$DIR/train_evaluate.sh" 0 2 > "$log2" 2>&1 &
pid2=$!
printf '%s\n' "$pid2" > "$pidfile2"
printf '%s\n' "$pid2" > "$STATE/seed2.pid"
echo "$(date -u +%FT%TZ) launched seed=2 gpu=0 pid=$pid2 log=$log2"
rc2=0
if wait "$pid2"; then :; else rc2=$?; fi
if [[ $rc2 -ne 0 || ! -s "$out2/EVALUATION_COMPLETED" || ! -s "$out2/core.pt" || \
      ! -s "$out2/validation_short_T256_settle64.json" || \
      ! -s "$out2/validation_long_T1024_settle512.json" ]]; then
  echo "Seed2 incomplete (exit $rc2); retaining artifacts, no automatic restart"; exit 7
fi
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$DONE"
echo "All SW0055 training/evaluations passed; wrote $DONE"
