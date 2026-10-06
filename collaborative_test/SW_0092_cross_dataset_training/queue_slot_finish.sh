#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0092_cross_dataset_training"
STATE="$ROOT/trained_models/SW0092_SLOT_OUR70000_QUEUE.json"
printf '{"status":"training_seeds_1_2"}\n' > "$STATE"
while [[ ! -s "$ROOT/trained_models/SW0092_slot_our70000_s1/TRAINING_COMPLETED" ]]; do sleep 60; done
printf '{"status":"training_seed_0_and_evaluating_seeds_1_2"}\n' > "$STATE"
(
 while [[ ! -s "$ROOT/trained_models/SW0092_slot_our70000_s2/TRAINING_COMPLETED" ]]; do sleep 60; done
 for epoch in 1 3 10; do
  bash "$DIR/evaluate_slot_our70000.sh" 3 1 "$epoch"
  bash "$DIR/evaluate_slot_our70000.sh" 3 2 "$epoch"
 done
) & evaluation_pid=$!
bash "$DIR/run_slot_seed0_final.sh" 2
wait "$evaluation_pid"
printf '{"status":"evaluating_seed_0"}\n' > "$STATE"
for epoch in 1 3 10; do
  bash "$DIR/evaluate_slot_our70000.sh" 2 0 "$epoch"
done
/Data0/kevinswk/envs/snn/bin/python "$DIR/summarize_slot_our70000.py" \
 "$ROOT/trained_models/SW0092_slot_our70000_eval" "$ROOT/trained_models/SW0092_slot_our70000_eval/summary.json"
printf '{"status":"complete"}\n' > "$STATE"

