#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0086_sw0072_on_peer_contract"
OUT="$ROOT/trained_models/SW0086_peer_contract"; STATE="$ROOT/trained_models/SW0086_QUEUE.json"
UPSTREAM="$ROOT/trained_models/SW0056_GPU_SEED12_QUEUE.json"
GAMMA=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
TARGETS=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
MANIFEST=/work/USERS/tkim1/clevr/with_masks/split_manifest.json
declare -a CKPTS=(
 "$ROOT/trained_models/SW0055_unique2500_s0_e10_lr0p0003/core.pt"
 "$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
 "$ROOT/trained_models/SW0072_frozen_seed0_graph_s2_e10/core.pt")
status=$(python3 - "$STATE" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
[[ "$status" == waiting_for_gpu ]] || { echo "SW0086 state is not resumable: $status" >&2; exit 3; }
while true; do
 upstream=$(python3 - "$UPSTREAM" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get("status","missing"))
except Exception: print("missing")
PY
 )
 [[ "$upstream" == complete ]] && break
 sleep 30
done
for path in "$GAMMA" "$TARGETS" "$MANIFEST" "${CKPTS[@]}"; do test -s "$path"; done
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
if x.get("status")!="waiting_for_gpu": raise RuntimeError("SW0086 state changed")
x.update(status="evaluating",selected_gpu=int(sys.argv[2]),started_unix_time=time.time(),serialized_after="SW0056_GPU_SEED12")
p.write_text(json.dumps(x,indent=2)+"\n")
PY
mkdir -p "$OUT/cache${chosen}"; export CUDA_VISIBLE_DEVICES="$chosen" TMPDIR="$OUT/cache${chosen}" TRITON_CACHE_DIR="$OUT/cache${chosen}"
for seed in 0 1 2; do
 result="$OUT/seed${seed}.json"; log="$OUT/seed${seed}.log"
 if [[ -s "$result" ]]; then continue; fi
 [[ ! -e "$log" ]] || { echo "partial seed $seed requires audit" >&2; exit 5; }
 /Data0/kevinswk/envs/snn/bin/python "$DIR/evaluate.py" --checkpoint "${CKPTS[$seed]}" \
  --seed "$seed" --gamma "$GAMMA" --targets "$TARGETS" --manifest "$MANIFEST" \
  --output "$result" --device cuda > "$log" 2>&1
 test -s "$result"
done
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize.py" --results "$OUT" --output "$OUT/summary.json" > "$OUT/summary.log" 2>&1
test -s "$OUT/summary.json"
python3 - "$STATE" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); x.update(status="complete",completed_unix_time=time.time()); p.write_text(json.dumps(x,indent=2)+"\n")
PY
