#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0087_matched_slot_comparison"
SLOT="$ROOT/collaborative_test/SW_0056_matched_slot_2500/results/SW0056_three_seed_validation.json"
MODEL="$ROOT/trained_models/SW0072_full320_long/summary.json"
OUT="$DIR/results/SW0087_goal_comparison.json"
STATE="$ROOT/trained_models/SW0087_GOAL_COMPARISON_QUEUE.json"

status=$(python3 - "$STATE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("status", "missing"))
PY
)
[[ "$status" == waiting_for_sw0056 ]] || { echo "unexpected state: $status" >&2; exit 3; }
test -s "$SLOT" -a -s "$MODEL"
[[ ! -e "$OUT" && ! -e "${OUT%.json}.md" ]] || { echo "comparison output exists" >&2; exit 4; }
/Data0/kevinswk/envs/snn/bin/python "$DIR/compare.py" \
  --model "$MODEL" --slot "$SLOT" --output "$OUT"
test -s "$OUT" -a -s "${OUT%.json}.md"
python3 - "$STATE" "$OUT" <<'PY'
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1]); result = json.load(open(sys.argv[2])); x = json.loads(p.read_text())
if x.get("status") != "waiting_for_sw0056": raise RuntimeError("state changed")
x.update(status="complete", goal_achieved=result["goal_achieved"],
         completed_unix_time=time.time(), recovery="watcher was absent; comparison resumed from complete inputs")
p.write_text(json.dumps(x, indent=2) + "\n")
PY
