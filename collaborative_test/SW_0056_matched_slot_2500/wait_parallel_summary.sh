#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
STATE="$ROOT/trained_models/SW0056_parallel_completion_queue.json"
SUMMARY="$DIR/results/SW0056_three_seed_validation.json"
[[ ! -e "$STATE" ]] || { echo "refusing existing parallel queue state" >&2; exit 3; }
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0056 matched Slot parallel completion queue",
 "status":"waiting_for_seeds", "registered_unix_time":time.time(),
 "seeds":[0,1,2], "training_concurrency":3,
 "reason":"user requested immediate matched-data Slot training; 48 CPUs and >80 GiB available",
 "protocol_unchanged":"2500 unique HDF5 scenes x10 exposures, 1563 updates, seed-specific order"
},indent=2)+"\n")
PY
while true; do
  ready=1
  for seed in 0 1 2; do
    test -s "$ROOT/trained_models/SW0056_matched_slot_2500_seed${seed}/COMPLETED" || ready=0
  done
  [[ "$ready" == 1 ]] && break
  sleep 60
done
for seed in 0 1 2; do
  out="$ROOT/trained_models/SW0056_matched_slot_2500_seed${seed}"
  /Data0/kevinswk/envs/snn/bin/python "$DIR/validate_phase.py" complete \
   "$out/validation1320_1639" "$seed" --training-protocol "$out/training_protocol.json"
done
[[ ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.md" ]] || { echo "summary already exists" >&2; exit 4; }
/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python "$DIR/summarize.py" \
 --model-root "$ROOT/trained_models" --output "$SUMMARY"
test -s "$SUMMARY" -a -s "${SUMMARY%.json}.md"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY

