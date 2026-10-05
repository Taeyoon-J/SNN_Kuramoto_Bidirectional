#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0057_fixed_multireadout"
STATE="$ROOT/trained_models/SW0057_fixed_multireadout_scheduler"
SW55_DONE="$ROOT/trained_models/SW0055_unique2500_autolaunch/LAUNCH_COMPLETED"
LOCK="$STATE/LOCK"; PID_FILE="$STATE/wrapper.pid"
STARTED="$(date -u +%FT%TZ)"
PY=/Data0/kevinswk/envs/snn/bin/python
SUMMARY="$DIR/results/sw0057_long_3seed.json"; SUMMARY_MD="${SUMMARY%.json}.md"
if [[ "${1:-}" == --dry-run ]]; then "$PY" "$DIR/scheduler_contract.py" --dry-run; exit 0; fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
mkdir -p "$STATE"
pid_status() {
  local pid cmd; pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 2
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 2
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmd" == *wait_for_sw0055_and_evaluate.sh* ]] || return 2
  return 0
}
if [[ -e "$LOCK" ]]; then
  [[ -d "$LOCK" && ! -L "$LOCK" && -s "$LOCK/owner" ]] || { echo "Unsafe/ambiguous lock" >&2; exit 2; }
  prior="$(sed -n 's/^pid=\([0-9][0-9]*\).*/\1/p' "$LOCK/owner" | head -n 1)"
  if pid_status "$prior"; then echo "Scheduler active under PID $prior" >&2; exit 2
  else rc=$?; [[ $rc -eq 1 ]] || { echo "Cannot prove lock stale" >&2; exit 2; }
  fi
  rm -rf -- "$LOCK"
  [[ ! -f "$PID_FILE" || "$(cat "$PID_FILE")" != "$prior" ]] || rm -f -- "$PID_FILE"
elif [[ -e "$PID_FILE" ]]; then
  prior="$(cat "$PID_FILE")"
  if pid_status "$prior"; then echo "Live wrapper PID without lock" >&2; exit 2
  else rc=$?; [[ $rc -eq 1 ]] || { echo "Ambiguous wrapper PID" >&2; exit 2; }; rm -f -- "$PID_FILE"
  fi
fi
mkdir "$LOCK" 2>/dev/null || { echo "Could not acquire lock" >&2; exit 2; }
printf 'pid=%s started_utc=%s\n' "$$" "$STARTED" > "$LOCK/owner"
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
  if [[ -f "$LOCK/owner" ]] && grep -Fxq "pid=$$ started_utc=$STARTED" "$LOCK/owner"; then rm -rf -- "$LOCK"; fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$$" ]]; then rm -f -- "$PID_FILE"; fi
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
LOG="$STATE/$(date -u +%Y%m%dT%H%M%SZ)_sw0057.log"
[[ ! -e "$LOG" ]] || { echo "Log exists: $LOG" >&2; exit 2; }
exec > >(tee -a "$LOG") 2>&1
MARKERS=(WAITING_FOR_SW0055 SW0055_READY SEEDS01_STARTED SEEDS01_COMPLETED SEED2_STARTED SEED2_COMPLETED SUMMARY_COMPLETED LAUNCH_COMPLETED)
for marker in "${MARKERS[@]}"; do [[ ! -e "$STATE/$marker" ]] || { echo "Partial prior state $marker; refusing restart"; exit 3; }; done
seed_out() { printf '%s/trained_models/SW0055_unique2500_s%s_e10_lr0p0003' "$ROOT" "$1"; }
guard_outputs() {
  [[ ! -e "$SUMMARY" && ! -e "$SUMMARY_MD" ]] || return 1
  for seed in 0 1 2; do
    local out; out="$(seed_out "$seed")"
    for f in "$out/SW0057_PREFLIGHT_V1.json" "$out/sw0057_fixed_multireadout_long.json" "$out/sw0057_fixed_multireadout_long.log"; do [[ ! -e "$f" ]] || { echo "Existing/partial output $f"; return 1; }; done
  done
}
guard_outputs || { echo "Refusing existing/partial SW0057 outputs"; exit 3; }
printf 'started_utc=%s\n' "$STARTED" > "$STATE/WAITING_FOR_SW0055"
echo "Waiting for SW0055 marker $SW55_DONE"
until [[ -s "$SW55_DONE" ]]; do sleep 30; done
for seed in 0 1 2; do
  out="$(seed_out "$seed")"
  for f in "$out/EVALUATION_COMPLETED" "$out/core.pt" "$out/validation_short_T256_settle64.json" "$out/validation_long_T1024_settle512.json"; do [[ -s "$f" ]] || { echo "SW0055 incomplete: $f"; exit 4; }; done
done
guard_outputs || { echo "SW0057 outputs appeared during wait"; exit 3; }
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SW0055_READY"
gpu_idle() {
  local gpu pids; gpu="$1"
  [[ "$gpu" == 0 || "$gpu" == 1 ]] || return 2
  pids="$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" || return 2
  [[ ! "$pids" =~ [0-9] ]]
}
run_seed() {
  local gpu seed out marker result log rc busyrc phase; gpu="$1"; seed="$2"; out="$(seed_out "$seed")"
  marker="$out/SW0057_PREFLIGHT_V1.json"; result="$out/sw0057_fixed_multireadout_long.json"; log="${result%.json}.log"
  for phase in preflight evaluation; do
    while true; do
      if [[ "$phase" == preflight ]]; then
        [[ ! -e "$marker" ]] || { echo "Existing preflight marker seed$seed; no restart"; return 11; }
        if gpu_idle "$gpu"; then :; else rc=$?; [[ $rc -eq 1 ]] || return 12; sleep 30; continue; fi
        if bash "$DIR/preflight.sh" "$gpu" "$seed"; then [[ -s "$marker" ]] || return 13; break
        else rc=$?; [[ ! -e "$marker" ]] || return 13
          if gpu_idle "$gpu"; then :; else busyrc=$?; [[ $busyrc -eq 1 ]] || return 12; sleep 30; continue; fi
          echo "Preflight failed on idle GPU"; return "$rc"
        fi
      else
        [[ ! -e "$result" && ! -e "$log" ]] || { echo "Partial evaluator output seed$seed; no restart"; return 14; }
        if gpu_idle "$gpu"; then :; else rc=$?; [[ $rc -eq 1 ]] || return 12; sleep 30; continue; fi
        if bash "$DIR/evaluate.sh" "$gpu" "$seed"; then "$PY" "$DIR/validate_result.py" "$result" "$out/core.pt" "$seed"; break
        else rc=$?; [[ ! -e "$result" && ! -e "$log" ]] || return 14
          if gpu_idle "$gpu"; then :; else busyrc=$?; [[ $busyrc -eq 1 ]] || return 12; sleep 30; continue; fi
          echo "Evaluation failed on idle GPU"; return "$rc"
        fi
      fi
    done
  done
}
printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEEDS01_STARTED"
run_seed 0 0 & p0=$!; printf '%s\n' "$p0" > "$STATE/seed0.pid"
run_seed 1 1 & p1=$!; printf '%s\n' "$p1" > "$STATE/seed1.pid"
r0=0; r1=0; if wait "$p0"; then :; else r0=$?; fi; if wait "$p1"; then :; else r1=$?; fi
[[ $r0 -eq 0 && $r1 -eq 0 ]] || { echo "Seed0/1 failed ($r0/$r1); artifacts retained"; exit 5; }
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEEDS01_COMPLETED"
printf 'started_utc=%s gpu=0\n' "$(date -u +%FT%TZ)" > "$STATE/SEED2_STARTED"
run_seed 0 2 || { echo "Seed2 failed; artifacts retained"; exit 6; }
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEED2_COMPLETED"
[[ ! -e "$SUMMARY" && ! -e "$SUMMARY_MD" ]] || { echo "Summary appeared; refusing overwrite"; exit 7; }
"$PY" "$DIR/summarize.py" --model-root "$ROOT/trained_models" --output "$SUMMARY"
"$PY" - "$SUMMARY" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r['experiment']=='SW0057 fixed multi-readout three-seed summary'
assert r['seeds']==[0,1,2] and r['ids']==[1320,1639] and r['window']=={'steps':1024,'settle':512}
assert set(r['readouts'])=={'spike_cc_threshold_0p50','membrane_spatial_sigma1p5_k10'}
for row in r['readouts'].values():
 for name in ('fg_ari','foreground_iou','matched_object_iou'):
  assert len(row[name]['per_seed'])==3 and math.isfinite(float(row[name]['mean']))
PY
[[ -s "$SUMMARY_MD" ]] || { echo "Missing Markdown summary"; exit 8; }
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SUMMARY_COMPLETED"
"$PY" "$DIR/scheduler_contract.py" --validate-state "$STATE"
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/LAUNCH_COMPLETED"
"$PY" "$DIR/scheduler_contract.py" --validate-state "$STATE" --require-complete
echo "SW0057 complete: $SUMMARY"
