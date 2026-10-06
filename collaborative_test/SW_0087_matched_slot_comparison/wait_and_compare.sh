#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0087_matched_slot_comparison"
SLOT="$ROOT/collaborative_test/SW_0056_matched_slot_2500/results/SW0056_three_seed_validation.json"
MODEL="$ROOT/collaborative_test/SW_0072_frozen_trained_graph/results/long/summary.json"
OUT="$DIR/results/SW0087_goal_comparison.json"
STATE="$ROOT/trained_models/SW0087_GOAL_COMPARISON_QUEUE.json"
[[ ! -e "$STATE" ]] || { echo "refusing existing SW0087 state" >&2; exit 3; }
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({
 "experiment":"SW0087 fixed-contract matched Slot goal comparison",
 "status":"waiting_for_sw0056", "registered_unix_time":time.time(),
 "model":"SW0072", "slot":"SW0056", "metrics":["fg_ari","foreground_iou","matched_object_iou"]
},indent=2)+"\n")
PY
while [[ ! -s "$SLOT" ]]; do sleep 60; done
test -s "$MODEL"
[[ ! -e "$OUT" && ! -e "${OUT%.json}.md" ]] || { echo "comparison output already exists" >&2; exit 4; }
/Data0/kevinswk/envs/snn/bin/python "$DIR/compare.py" --model "$MODEL" --slot "$SLOT" --output "$OUT" \
 > "$DIR/results/SW0087_goal_comparison.log" 2>&1
test -s "$OUT" -a -s "${OUT%.json}.md"
python3 - "$STATE" "$OUT" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); result=json.load(open(sys.argv[2])); x=json.loads(p.read_text())
x.update(status="complete",goal_achieved=result["goal_achieved"],completed_unix_time=time.time())
p.write_text(json.dumps(x,indent=2)+"\n")
PY
