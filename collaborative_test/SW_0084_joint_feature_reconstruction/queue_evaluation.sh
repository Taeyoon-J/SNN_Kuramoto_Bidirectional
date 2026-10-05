#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0084_joint_feature_reconstruction"
OUT="$ROOT/trained_models/SW0084_fixed_pilot"
STATE="$ROOT/trained_models/SW0084_EVALUATION_QUEUE.json"

for arm in A B C D; do
  test -s "$ROOT/trained_models/SW0084_${arm}_seed1/TRAINING_COMPLETED"
done

python3 - "$STATE" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
if path.exists():
    raise FileExistsError(f"refusing existing queue state: {path}")
path.write_text(json.dumps({
    "experiment": "SW0084 fixed seed1 pilot evaluation queue",
    "status": "waiting_for_gpu",
    "registered_unix_time": time.time(),
    "jobs": ["baseline_epoch10"] + [
        f"{arm}_epoch{epoch}" for arm in "ABCD" for epoch in ("05", "10")
    ],
    "gpu_policy": "use any idle GPU0-3; if a tkim1 GPU process exists, use only GPU0-1",
    "training_restart_allowed": False,
}, indent=2) + "\n", encoding="utf-8")
PY

while true; do
  taeyoon=0
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == tkim1 ]] && taeyoon=1
  done
  if [[ "$taeyoon" == 1 ]]; then candidates=(0 1); else candidates=(0 1 2 3); fi
  chosen=""
  for gpu in "${candidates[@]}"; do
    pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1)
    if [[ ! "$pids" =~ [0-9] ]]; then chosen="$gpu"; break; fi
  done
  [[ -n "$chosen" ]] && break
  sleep 30
done

python3 - "$STATE" "$chosen" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1]); state = json.loads(path.read_text())
state.update(status="evaluating", selected_gpu=int(sys.argv[2]), started_unix_time=time.time())
path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
PY

run_one() {
  local tag="$1" arm="$2" epoch="$3"
  local result="$OUT/${tag}_seed1_n32.json" log="$OUT/${tag}_seed1_n32.log"
  if [[ -s "$result" ]]; then return 0; fi
  [[ ! -e "$log" ]] || { echo "partial evaluation requires audit: $log" >&2; exit 5; }
  bash "$DIR/evaluate.sh" "$chosen" "$arm" "$epoch"
}

run_one baseline_epoch10 baseline 10
for arm in A B C D; do
  for epoch in 05 10; do run_one "${arm}_epoch${epoch}" "$arm" "$epoch"; done
done

if [[ ! -e "$OUT/summary.json" && ! -e "$OUT/summary.md" ]]; then
  /Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" \
    --results-dir "$OUT" --output-json "$OUT/summary.json" --output-md "$OUT/summary.md"
fi
test -s "$OUT/summary.json" -a -s "$OUT/summary.md"

python3 - "$STATE" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1]); state = json.loads(path.read_text())
state.update(status="complete", completed_unix_time=time.time())
path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
PY
