#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
STATE="$ROOT/trained_models/SW0056_low_load_scheduler"
LOCK="$STATE/LOCK"; PID_FILE="$STATE/wrapper.pid"
PY=/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python
SNN_PY=/Data0/kevinswk/envs/snn/bin/python
STARTED="$(date -u +%FT%TZ)"
SUMMARY="$DIR/results/SW0056_three_seed_validation.json"
SUMMARY_MD="${SUMMARY%.json}.md"
if [[ "${1:-}" == --dry-run ]]; then
  echo "DRY RUN: concurrency=1; ten consecutive 60s samples load1<=32 load5<=36 load15<=40 MemAvailable>=8GiB; smoke then seeds0,1,2; gate before every training/inference/scoring phase"
  exit 0
fi
[[ $# -eq 0 ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }
mkdir -p "$STATE"
pid_status() {
  local pid cmd; pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 2
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 2
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmd" == *wait_for_low_load_and_run.sh* ]] || return 2
  return 0
}
if [[ -e "$LOCK" ]]; then
  [[ -d "$LOCK" && ! -L "$LOCK" && -s "$LOCK/owner" ]] || { echo "Unsafe/ambiguous low-load lock" >&2; exit 2; }
  prior="$(sed -n 's/^pid=\([0-9][0-9]*\).*/\1/p' "$LOCK/owner" | head -n 1)"
  if pid_status "$prior"; then echo "Low-load scheduler active under PID $prior" >&2; exit 2
  else rc=$?; [[ $rc -eq 1 ]] || { echo "Cannot prove low-load lock stale" >&2; exit 2; }
  fi
  rm -rf -- "$LOCK"
  [[ ! -f "$PID_FILE" || "$(cat "$PID_FILE")" != "$prior" ]] || rm -f -- "$PID_FILE"
elif [[ -e "$PID_FILE" ]]; then
  prior="$(cat "$PID_FILE")"
  if pid_status "$prior"; then echo "Live wrapper PID without lock" >&2; exit 2
  else rc=$?; [[ $rc -eq 1 ]] || { echo "Ambiguous low-load wrapper PID" >&2; exit 2; }; rm -f -- "$PID_FILE"
  fi
fi
mkdir "$LOCK" 2>/dev/null || { echo "Could not acquire low-load lock" >&2; exit 2; }
printf 'pid=%s started_utc=%s\n' "$$" "$STARTED" > "$LOCK/owner"
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
  if [[ -f "$LOCK/owner" ]] && grep -Fxq "pid=$$ started_utc=$STARTED" "$LOCK/owner"; then rm -rf -- "$LOCK"; fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$$" ]]; then rm -f -- "$PID_FILE"; fi
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
LOG="$STATE/$(date -u +%Y%m%dT%H%M%SZ)_low_load_run.log"
[[ ! -e "$LOG" ]] || { echo "Timestamp log exists: $LOG" >&2; exit 2; }
exec > >(tee -a "$LOG") 2>&1
MARKERS=(WAITING_FOR_LOW_LOAD SMOKE_VALIDATED SEED0_COMPLETED SEED1_COMPLETED SEED2_COMPLETED SUMMARY_COMPLETED COMPLETED)
for marker in "${MARKERS[@]}"; do
  [[ ! -e "$STATE/$marker" ]] || { echo "Prior/partial scheduler marker $marker; refusing restart"; exit 3; }
done
[[ ! -e "$STATE/FAILED" ]] || { echo "Prior failed scheduler run; manual review required"; exit 3; }
[[ ! -e "$STATE/FAILED" ]] || { echo "Prior failed scheduler run; manual review required"; exit 3; }
[[ ! -e "$SUMMARY" && ! -e "$SUMMARY_MD" ]] || { echo "Existing summary output; refusing overwrite"; exit 3; }
mark_failure() {
  local phase rc; phase="$1"; rc="$2"
  [[ ! -e "$STATE/FAILED" ]] || return 0
  (set -o noclobber; printf 'phase=%s\nrc=%s\ntimestamp_utc=%s\n' "$phase" "$rc" "$(date -u +%FT%TZ)") > "$STATE/FAILED" 2>/dev/null || true
}
printf 'started_utc=%s\n' "$STARTED" > "$STATE/WAITING_FOR_LOW_LOAD"
echo "run_seed.sh enforces the ten-sample load gate before each compute phase; starting smoke controller"
if SW0056_REQUIRE_LOW_LOAD=1 bash "$DIR/run_seed.sh" 0 smoke; then
  if "$PY" "$DIR/validate_phase.py" smoke "$ROOT/trained_models/SW0056_matched_slot_2500_seed0-smoke" 0; then :
  else rc=$?; mark_failure smoke-validation "$rc"; exit "$rc"; fi
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SMOKE_VALIDATED"
else
  rc=$?; mark_failure smoke "$rc"; exit "$rc"
fi
for seed in 0 1 2; do
  echo "Starting seed$seed controller; run_seed.sh will recheck load before each compute phase"
  if SW0056_REQUIRE_LOW_LOAD=1 bash "$DIR/run_seed.sh" "$seed" full; then
    out="$ROOT/trained_models/SW0056_matched_slot_2500_seed${seed}"
    if "$SNN_PY" "$DIR/validate_phase.py" complete "$out/validation1320_1639" "$seed" --training-protocol "$out/training_protocol.json"; then :
    else rc=$?; mark_failure "seed${seed}_validation" "$rc"; exit "$rc"; fi
    [[ -s "$out/COMPLETED" ]] || { echo "seed$seed COMPLETED marker missing" >&2; mark_failure "seed${seed}_validation" 9; exit 9; }
    printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SEED${seed}_COMPLETED"
  else
    rc=$?; mark_failure "seed${seed}" "$rc"; exit "$rc"
  fi
done
summary_phase() {
if "$PY" "$DIR/summarize.py" --model-root "$ROOT/trained_models" --output "$SUMMARY"; then :
else return $?; fi
if "$PY" - "$SUMMARY" <<'PY'
import json,math,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
assert r['experiment']=='SW0056 matched-data Slot Attention three-seed summary'
assert r['seeds']==[0,1,2] and r['image_ids_inclusive']==[1320,1639] and r['count']==320
for name in ('fg_ari','foreground_iou','matched_object_iou'):
 row=r['metrics'][name]
 assert len(row['per_seed'])==3 and math.isfinite(float(row['mean'])) and math.isfinite(float(row['std_sample']))
PY
then :; else return $?; fi
[[ -s "$SUMMARY_MD" ]] || { echo "Summary Markdown missing" >&2; return 10; }
}
if summary_phase; then :; else
  rc=$?; mark_failure summary "$rc"; echo "Summary phase failed; preserving any partial summary" >&2; exit "$rc"
fi
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/SUMMARY_COMPLETED"
if "$PY" "$DIR/scheduler_contract.py" --validate-state "$STATE"; then :
else rc=$?; mark_failure state-validation "$rc"; exit "$rc"; fi
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" > "$STATE/COMPLETED"
if "$PY" "$DIR/scheduler_contract.py" --validate-state "$STATE" --require-complete; then :
else rc=$?; mark_failure final-state-validation "$rc"; exit "$rc"; fi
echo "SW0056 smoke and seeds0/1/2 completed sequentially under the fixed low-load gate."
