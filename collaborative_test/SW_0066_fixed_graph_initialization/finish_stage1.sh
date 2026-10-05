#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
EXP="$ROOT/collaborative_test/SW_0066_fixed_graph_initialization"
OUT="$ROOT/trained_models/SW0066_stage1"
LOCK="$ROOT/trained_models/SW0066_stage1_controller.lock"

mkdir "$LOCK"
mkdir -p "$OUT"
trap 'printf "failed\n" > "$OUT/STAGE1_FAILED"' ERR

deadline=$((SECONDS + 14400))
while [[ ! -f "$ROOT/trained_models/SW0066_graphinit0_s1_e10/TRAINING_COMPLETED" ||
         ! -f "$ROOT/trained_models/SW0066_graphinit0_s2_e10/TRAINING_COMPLETED" ]]; do
  (( SECONDS < deadline )) || { echo "training wait timed out" >&2; exit 4; }
  sleep 20
done

for gpu in 0 1; do
  while nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -Eq '[0-9]'; do
    (( SECONDS < deadline )) || { echo "GPU wait timed out" >&2; exit 5; }
    sleep 5
  done
done

bash "$EXP/evaluate_stage1.sh" 0 1 candidate & p1=$!
bash "$EXP/evaluate_stage1.sh" 1 2 candidate & p2=$!
wait "$p1"; wait "$p2"
bash "$EXP/evaluate_stage1.sh" 0 1 baseline & p1=$!
bash "$EXP/evaluate_stage1.sh" 1 2 baseline & p2=$!
wait "$p1"; wait "$p2"

/Data0/kevinswk/envs/snn/bin/python "$EXP/summarize_stage1.py" \
  --result-dir "$OUT" --output "$OUT/summary.json" > "$OUT/summary.log"
test -s "$OUT/summary.json"
printf 'completed\n' > "$OUT/STAGE1_COMPLETED"
