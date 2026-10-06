#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0084_joint_feature_reconstruction"
OUT="$ROOT/trained_models/SW0084_fixed_pilot"
STATE="$ROOT/trained_models/SW0084_EVALUATION_QUEUE.json"
SLOT_UPSTREAM="$ROOT/trained_models/SW0056_GPU_SEED12_QUEUE.json"
TRANSFER_UPSTREAM="$ROOT/trained_models/SW0086_QUEUE.json"

status=$(python3 - "$STATE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
[[ "$status" == waiting_for_gpu ]] || { echo "SW0084 state is not resumable: $status" >&2; exit 3; }

# Keep the matched Slot GPU run ahead of this evaluation. Also require the
# already-finished SW0086 transfer state to be durably summarized.
while true; do
  upstream=$(python3 - "$SLOT_UPSTREAM" "$TRANSFER_UPSTREAM" <<'PY'
import json,sys
def status(path):
    try: return json.load(open(path)).get("status", "missing")
    except Exception: return "missing"
print(status(sys.argv[1]) + ":" + status(sys.argv[2]))
PY
  )
  [[ "$upstream" == complete:complete ]] && break
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
         serialized_after="SW0056_GPU_SEED12_and_SW0086")
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

