#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0084_joint_feature_reconstruction"
OUT="$ROOT/trained_models/SW0084_fixed_pilot"
STATE="$ROOT/trained_models/SW0084_EVALUATION_QUEUE.json"
UPSTREAM="$ROOT/trained_models/SW0086_QUEUE.json"

status=$(python3 - "$STATE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
[[ "$status" == waiting_for_gpu ]] || { echo "SW0084 state is not resumable: $status" >&2; exit 3; }

# SW0086 is the user's newer cross-contract request. Serialize the two queues so
# they cannot both claim the same newly idle GPU and leave one partial.
while true; do
  upstream=$(python3 - "$UPSTREAM" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status", "missing"))
except Exception: print("missing")
PY
  )
  [[ "$upstream" == complete ]] && break
  sleep 30
done

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
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text())
if x.get("status") != "waiting_for_gpu": raise RuntimeError("state changed before resume")
x.update(status="evaluating", selected_gpu=int(sys.argv[2]), started_unix_time=time.time(),
         serialized_after="SW0086")
p.write_text(json.dumps(x,indent=2)+"\n")
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
  /Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" --results-dir "$OUT" \
   --output-json "$OUT/summary.json" --output-md "$OUT/summary.md"
fi
test -s "$OUT/summary.json" -a -s "$OUT/summary.md"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY

