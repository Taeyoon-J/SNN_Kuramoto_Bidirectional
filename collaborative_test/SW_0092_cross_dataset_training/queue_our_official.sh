#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0092_cross_dataset_training"
STATE="$ROOT/trained_models/SW0092_OUR_OFFICIAL_QUEUE.json"; PY=/Data0/kevinswk/envs/snn/bin/python
while [[ ! -s "$ROOT/trained_models/SW0090_VISUALIZATION_COMPLETE" ]]; do sleep 60; done
printf '{"status":"building_gamma"}\n' > "$STATE"
if [[ ! -s "$ROOT/data/SW_0092_cross_dataset/official_gamma_manifest.json" ]]; then
CUDA_VISIBLE_DEVICES=0 "$PY" "$DIR/build_official_gamma.py" \
 --dataset "$ROOT/data/SW_0092_cross_dataset/official_clevr6_train.hdf5" \
 --output "$ROOT/data/SW_0092_cross_dataset/official_clevr6_gamma.pt" \
 --manifest "$ROOT/data/SW_0092_cross_dataset/official_gamma_manifest.json" \
 > "$ROOT/trained_models/SW0092/build_official_gamma.log" 2>&1
fi
printf '{"status":"training_seeds_0_1"}\n' > "$STATE"
bash "$DIR/run_our_official.sh" 0 0 & p0=$!; bash "$DIR/run_our_official.sh" 1 1 & p1=$!; wait "$p0"; wait "$p1"
printf '{"status":"training_seed_2"}\n' > "$STATE"; bash "$DIR/run_our_official.sh" 0 2
printf '{"status":"evaluating"}\n' > "$STATE"
for epoch in 1 3 10; do
  bash "$DIR/evaluate_our_official.sh" 0 0 "$epoch" & p0=$!; bash "$DIR/evaluate_our_official.sh" 1 1 "$epoch" & p1=$!; wait "$p0"; wait "$p1"
  bash "$DIR/evaluate_our_official.sh" 0 2 "$epoch"
done
"$PY" "$DIR/summarize_our_official.py" "$ROOT/trained_models/SW0092_our_on_official_eval" \
 "$ROOT/trained_models/SW0092_our_on_official_eval/summary.json"
printf '{"status":"complete"}\n' > "$STATE"

