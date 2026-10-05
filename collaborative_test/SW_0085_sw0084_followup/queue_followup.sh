#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0085_sw0084_followup"
OUT="$ROOT/trained_models/SW0085_followup"
STATE="$ROOT/trained_models/SW0085_FOLLOWUP_QUEUE.json"
mkdir -p "$OUT"
[[ ! -e "$STATE" ]] || { echo "refusing existing queue state" >&2; exit 3; }
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0085 automatic SW0084 follow-up", "status":"waiting_for_sw0084",
 "registered_unix_time":time.time(), "gpu_policy":"any idle GPU0-3; GPU0-1 only while tkim1 is active",
 "actions":["maximin candidate selection","baseline/candidate activation-flow probe","full320 long evaluation only after all-three pilot advance"]
},indent=2)+"\n")
PY
while true; do
  status=$(python3 - "$ROOT/trained_models/SW0084_EVALUATION_QUEUE.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
  )
  [[ "$status" == complete ]] && break
  sleep 30
done
test -s "$ROOT/trained_models/SW0084_fixed_pilot/summary.json"
/Data0/kevinswk/envs/snn/bin/python "$DIR/select_candidate.py" \
 --summary "$ROOT/trained_models/SW0084_fixed_pilot/summary.json" --output "$OUT/selection.json"
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
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="diagnosing",selected_gpu=int(sys.argv[2]),started_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
bash "$DIR/probe.sh" "$chosen" baseline
bash "$DIR/probe.sh" "$chosen" candidate
read -r ARM EPOCH PROMOTE < <(/Data0/kevinswk/envs/snn/bin/python - "$OUT/selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x["chosen"]["arm"],x["chosen"]["epoch"],str(x["full320_promotion"]).lower())
PY
 )
/Data0/kevinswk/envs/snn/bin/python "$DIR/compare_flow.py" --baseline "$OUT/flow_baseline.json" \
 --candidate "$OUT/flow_${ARM}_epoch${EPOCH}.json" --selection "$OUT/selection.json" --output "$OUT/flow_comparison.json"
if [[ "$PROMOTE" == true ]]; then
  python3 - "$STATE" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x["status"]="full320_evaluation"; p.write_text(json.dumps(x,indent=2)+"\n")
PY
  bash "$DIR/evaluate_full.sh" "$chosen" "$ARM" "$EPOCH"
fi
python3 - "$STATE" "$PROMOTE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",full320_promoted=sys.argv[2]=="true",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY

