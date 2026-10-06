#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0089_gate_affinity_anchor"
STATE="$ROOT/trained_models/SW0089_QUEUE.json"; UPSTREAM="$ROOT/trained_models/SW0088_QUEUE.json"
[[ ! -e "$STATE" ]] || { echo "refusing existing SW0089 state" >&2; exit 3; }
python3 - "$STATE" <<'PY'
import json, pathlib, sys, time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0089 classifier-aligned gate-affinity anchor", "status":"waiting_for_sw0088",
 "registered_unix_time":time.time(), "arms":["H","I","J"],
 "weights":[1.0,10.0,100.0], "reconstruction_weight":0.3,
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
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(
 status="running",selected_gpus=[int(v) for v in sys.argv[2].split()],started_unix_time=time.time())
p.write_text(json.dumps(x,indent=2)+"\n")
PY
mkdir -p "$ROOT/trained_models/SW0089_fixed_pilot"
bash "$DIR/evaluate.sh" "${idle[0]}" baseline
run_arm() {
 local gpu="$1" arm="$2"
 bash "$DIR/preflight.sh" "$gpu" "$arm"
 bash "$DIR/run.sh" "$gpu" "$arm"
 bash "$DIR/evaluate.sh" "$gpu" "$arm"
}
run_arm "${idle[0]}" H & p0=$!
run_arm "${idle[1]}" I & p1=$!
if [[ "$need" == 3 ]]; then run_arm "${idle[2]}" J & p2=$!; fi
rc=0; wait "$p0" || rc=$?; wait "$p1" || rc=$?
if [[ "$need" == 3 ]]; then wait "$p2" || rc=$?; else run_arm "${idle[0]}" J || rc=$?; fi
[[ "$rc" == 0 ]] || exit "$rc"
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" \
 --results-dir "$ROOT/trained_models/SW0089_fixed_pilot" \
 --output "$ROOT/trained_models/SW0089_fixed_pilot/summary.json" \
 > "$ROOT/trained_models/SW0089_fixed_pilot/summary.log" 2>&1
test -s "$ROOT/trained_models/SW0089_fixed_pilot/summary.json"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time())
p.write_text(json.dumps(x,indent=2)+"\n")
PY
