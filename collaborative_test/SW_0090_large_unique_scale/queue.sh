#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0090_large_unique_scale"
STATE="$ROOT/trained_models/SW0090_QUEUE.json"
PY=/Data0/kevinswk/envs/snn/bin/python
mkdir -p "$ROOT/trained_models" "$ROOT/data/SW_0090_large_unique_scale"
printf '{"status":"building_gamma","started":%s}\n' "$(date +%s)" > "$STATE"
if [[ ! -s "$ROOT/data/SW_0090_large_unique_scale/gamma_train_70000.pt" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" "$DIR/build_gamma.py" \
    --count 70000 --batch-size 128 --device cuda \
    --output "$ROOT/data/SW_0090_large_unique_scale/gamma_train_70000.pt" \
    --manifest "$ROOT/data/SW_0090_large_unique_scale/manifest.json" \
    > "$ROOT/data/SW_0090_large_unique_scale/build.log" 2>&1
fi
printf '{"status":"training_seeds_0_1","started":%s}\n' "$(date +%s)" > "$STATE"
bash "$DIR/run.sh" 0 0 > "$ROOT/trained_models/SW0090_seed0_runner.log" 2>&1 & p0=$!
bash "$DIR/run.sh" 1 1 > "$ROOT/trained_models/SW0090_seed1_runner.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
printf '{"status":"training_seed_2","started":%s}\n' "$(date +%s)" > "$STATE"
bash "$DIR/run.sh" 0 2 > "$ROOT/trained_models/SW0090_seed2_runner.log" 2>&1
printf '{"status":"evaluating","started":%s}\n' "$(date +%s)" > "$STATE"
for epoch in 1 3 10; do
  bash "$DIR/evaluate.sh" 0 0 "$epoch" & p0=$!
  bash "$DIR/evaluate.sh" 1 1 "$epoch" & p1=$!
  wait "$p0"; wait "$p1"
  bash "$DIR/evaluate.sh" 0 2 "$epoch"
done
"$PY" "$DIR/summarize.py" --results "$ROOT/trained_models/SW0090_full320_long" \
  --output "$ROOT/trained_models/SW0090_full320_long/summary.json" \
  > "$ROOT/trained_models/SW0090_full320_long/summary.log" 2>&1
printf '{"status":"complete","completed":%s}\n' "$(date +%s)" > "$STATE"

