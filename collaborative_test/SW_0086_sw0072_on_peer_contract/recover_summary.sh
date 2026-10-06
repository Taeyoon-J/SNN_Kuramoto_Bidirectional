#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0086_sw0072_on_peer_contract"
OUT="$ROOT/trained_models/SW0086_peer_contract"
STATE="$ROOT/trained_models/SW0086_QUEUE.json"

for seed in 0 1 2; do test -s "$OUT/seed${seed}.json"; done
test ! -e "$OUT/summary.json"
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" \
  --results "$OUT" --output "$OUT/summary.json"
test -s "$OUT/summary.json"
python3 - "$STATE" <<'PY'
import json, pathlib, sys, time
p = pathlib.Path(sys.argv[1])
x = json.loads(p.read_text())
if x.get("status") != "evaluating":
    raise RuntimeError(f"unexpected state: {x.get('status')}")
x.update(status="complete", completed_unix_time=time.time(),
         recovery="summary-only after metric-key contract correction")
p.write_text(json.dumps(x, indent=2) + "\n")
PY
