#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
STATE="$ROOT/trained_models/SW0056_GPU_SEED12_QUEUE.json"
SUMMARY="$DIR/results/SW0056_three_seed_validation.json"
SUMMARY_ROOT="$ROOT/trained_models/SW0056_gpu_summary_root"

status=$(python3 - "$STATE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", "missing"))
PY
)
[[ "$status" == training ]] || { echo "unexpected queue status: $status" >&2; exit 3; }

# Preserve the two live GPU jobs and replace only the much slower CPU seed0.
recover_validator_failure() {
  local seed="$1" out val failed
  out="$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed${seed}"
  val="$out/validation1320_1639"; failed="$out/FAILED"
  [[ -e "$out/COMPLETED" ]] && return 0
  [[ -s "$failed" && -s "$val/SCORING_COMPLETED" ]] || return 0
  grep -qx 'phase=scoring-validation' "$failed"
  grep -qx 'rc=2' "$failed"
  /Data0/kevinswk/envs/snn/bin/python "$DIR/validate_phase.py" complete \
    "$val" "$seed" --training-protocol "$out/training_protocol.json"
  python3 - "$out/VALIDATOR_RECOVERY.json" "$failed" <<'PY'
import json, pathlib, sys, time
failed = pathlib.Path(sys.argv[2]).read_text()
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "reason": "runner omitted required --training-protocol from scoring validation",
    "original_failed_marker": failed, "recovered_unix_time": time.time(),
    "artifacts_recomputed": False,
}, indent=2) + "\n")
PY
  rm -- "$failed"
  printf 'seed=%s completed_utc=%s validator_recovery=true\n' \
    "$seed" "$(date -u +%FT%TZ)" > "$out/COMPLETED"
}
recover_validator_failure 1
recover_validator_failure 2
for seed in 1 2; do
  while [[ ! -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed${seed}/COMPLETED" ]]; do
    sleep 30
  done
done
bash "$DIR/run_gpu_seed.sh" 0 0 full

for seed in 0 1 2; do
  test -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed${seed}/COMPLETED"
done
[[ ! -e "$SUMMARY_ROOT" ]] || { echo "summary root exists" >&2; exit 4; }
mkdir "$SUMMARY_ROOT"
for seed in 0 1 2; do
  ln -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed${seed}" \
    "$SUMMARY_ROOT/SW0056_matched_slot_2500_seed${seed}"
  out="$SUMMARY_ROOT/SW0056_matched_slot_2500_seed${seed}"
  /Data0/kevinswk/envs/snn/bin/python "$DIR/validate_phase.py" complete \
    "$out/validation1320_1639" "$seed" --training-protocol "$out/training_protocol.json"
done
[[ ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.md" ]] || { echo "summary already exists" >&2; exit 5; }
/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python "$DIR/summarize.py" \
  --model-root "$SUMMARY_ROOT" --output "$SUMMARY"
test -s "$SUMMARY" -a -s "${SUMMARY%.json}.md"
python3 - "$STATE" <<'PY'
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1]); x = json.loads(p.read_text())
if x.get("status") != "training": raise RuntimeError("queue state changed")
x.update(status="complete", completed_unix_time=time.time(),
         seed0_source="GPU accelerated matched run", seeds=[0, 1, 2])
p.write_text(json.dumps(x, indent=2) + "\n")
PY
