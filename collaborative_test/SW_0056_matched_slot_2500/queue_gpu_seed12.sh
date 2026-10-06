#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
STATE="$ROOT/trained_models/SW0056_GPU_SEED12_QUEUE.json"
SUMMARY="$DIR/results/SW0056_three_seed_validation.json"
SUMMARY_ROOT="$ROOT/trained_models/SW0056_gpu_summary_root"
[[ ! -e "$STATE" ]] || { echo "refusing existing GPU queue state" >&2; exit 3; }
for path in "$ROOT/trained_models/SW0056_matched_slot_2500_seed0" \
 "$ROOT/trained_models/SW0056_matched_slot_2500_seed1" "$ROOT/trained_models/SW0056_matched_slot_2500_seed2"; do test -d "$path"; done
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0056 CUDA seed1/2 acceleration",
 "status":"waiting_for_two_gpus","registered_unix_time":time.time(),"seeds":[1,2],
 "seed0_source":"existing CPU run","gpu_environment":"tensorflow[and-cuda]==2.15.1",
 "protocol_unchanged":"2500 unique scenes x10 exposures, 1563 updates"
},indent=2)+"\n")
PY
while true; do
 taeyoon=0
 for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  [[ "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" == tkim1 ]] && taeyoon=1
 done
 if [[ "$taeyoon" == 1 ]]; then candidates=(0 1); else candidates=(0 1 2 3); fi
 idle=()
 for gpu in "${candidates[@]}"; do
  pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1)
  [[ ! "$pids" =~ [0-9] ]] && idle+=("$gpu")
 done
 [[ ${#idle[@]} -ge 2 ]] && break
 sleep 30
done
python3 - "$STATE" "${idle[0]}" "${idle[1]}" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="smoke",selected_gpus=[int(sys.argv[2]),int(sys.argv[3])],started_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
bash "$DIR/run_gpu_seed.sh" 1 "${idle[0]}" smoke & s1=$!
bash "$DIR/run_gpu_seed.sh" 2 "${idle[1]}" smoke & s2=$!
rc=0; wait "$s1" || rc=$?; wait "$s2" || rc=$?; [[ "$rc" == 0 ]] || exit "$rc"
python3 - "$STATE" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x["status"]="training"; p.write_text(json.dumps(x,indent=2)+"\n")
PY
bash "$DIR/run_gpu_seed.sh" 1 "${idle[0]}" full & p1=$!
bash "$DIR/run_gpu_seed.sh" 2 "${idle[1]}" full & p2=$!
rc=0; wait "$p1" || rc=$?; wait "$p2" || rc=$?; [[ "$rc" == 0 ]] || exit "$rc"
while [[ ! -s "$ROOT/trained_models/SW0056_matched_slot_2500_seed0/COMPLETED" ]]; do sleep 30; done
for seed in 1 2; do test -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed${seed}/COMPLETED"; done
[[ ! -e "$SUMMARY_ROOT" ]] || { echo "summary root exists" >&2; exit 4; }
mkdir "$SUMMARY_ROOT"
ln -s "$ROOT/trained_models/SW0056_matched_slot_2500_seed0" "$SUMMARY_ROOT/SW0056_matched_slot_2500_seed0"
ln -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed1" "$SUMMARY_ROOT/SW0056_matched_slot_2500_seed1"
ln -s "$ROOT/trained_models/SW0056_matched_slot_2500_gpu_seed2" "$SUMMARY_ROOT/SW0056_matched_slot_2500_seed2"
for seed in 0 1 2; do
 out="$SUMMARY_ROOT/SW0056_matched_slot_2500_seed${seed}"
 /Data0/kevinswk/envs/snn/bin/python "$DIR/validate_phase.py" complete \
  "$out/validation1320_1639" "$seed" --training-protocol "$out/training_protocol.json"
done
[[ ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.md" ]] || { echo "summary already exists" >&2; exit 5; }
/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python "$DIR/summarize.py" \
 --model-root "$SUMMARY_ROOT" --output "$SUMMARY"
test -s "$SUMMARY" -a -s "${SUMMARY%.json}.md"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
