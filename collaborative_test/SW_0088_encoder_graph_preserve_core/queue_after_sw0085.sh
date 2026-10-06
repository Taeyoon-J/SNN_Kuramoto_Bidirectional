#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0088_encoder_graph_preserve_core"
STATE="$ROOT/trained_models/SW0088_QUEUE.json"; UPSTREAM="$ROOT/trained_models/SW0085_FOLLOWUP_QUEUE.json"
[[ ! -e "$STATE" ]] || { echo "refusing existing SW0088 state" >&2; exit 3; }
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0088 encoder+graph with frozen downstream core",
 "status":"waiting_for_sw0085","registered_unix_time":time.time(),"arms":["E","F","G"],
 "conditional":"run only if SW0084 has no all-three pilot advance",
 "gpu_policy":"GPU0-1 while tkim1 is active; otherwise any idle GPU0-3"
},indent=2)+"\n")
PY
while true; do
 status=$(python3 - "$UPSTREAM" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
 )
 [[ "$status" == complete ]] && break
 sleep 30
done
promoted=$(python3 - "$UPSTREAM" <<'PY'
import json,sys
print(str(bool(json.load(open(sys.argv[1])).get("full320_promoted",False))).lower())
PY
)
if [[ "$promoted" == true ]]; then
 python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="skipped_for_sw0084_advance",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
 exit 0
fi

while true; do
 taeyoon=0
 for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == tkim1 ]] && taeyoon=1
 done
 if [[ "$taeyoon" == 1 ]]; then candidates=(0 1); need=2; else candidates=(0 1 2 3); need=3; fi
 idle=()
 for gpu in "${candidates[@]}"; do
  pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1)
  [[ ! "$pids" =~ [0-9] ]] && idle+=("$gpu")
 done
 [[ ${#idle[@]} -ge $need ]] && break
 sleep 30
done
python3 - "$STATE" "${idle[*]}" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="running",selected_gpus=[int(v) for v in sys.argv[2].split()],started_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
mkdir -p "$ROOT/trained_models/SW0088_fixed_pilot"
bash "$DIR/evaluate.sh" "${idle[0]}" baseline
run_arm() {
 local gpu="$1" arm="$2"
 bash "$DIR/preflight.sh" "$gpu" "$arm"
 bash "$DIR/run.sh" "$gpu" "$arm"
 bash "$DIR/evaluate.sh" "$gpu" "$arm"
}
run_arm "${idle[0]}" E & p0=$!
run_arm "${idle[1]}" F & p1=$!
if [[ "$need" == 3 ]]; then run_arm "${idle[2]}" G & p2=$!; fi
rc=0; wait "$p0" || rc=$?; wait "$p1" || rc=$?
if [[ "$need" == 3 ]]; then wait "$p2" || rc=$?; else run_arm "${idle[0]}" G || rc=$?; fi
[[ "$rc" == 0 ]] || exit "$rc"
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" \
 --results-dir "$ROOT/trained_models/SW0088_fixed_pilot" \
 --output "$ROOT/trained_models/SW0088_fixed_pilot/summary.json" \
 > "$ROOT/trained_models/SW0088_fixed_pilot/summary.log" 2>&1
test -s "$ROOT/trained_models/SW0088_fixed_pilot/summary.json"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
