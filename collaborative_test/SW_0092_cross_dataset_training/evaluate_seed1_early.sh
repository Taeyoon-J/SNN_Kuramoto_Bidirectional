#!/usr/bin/env bash
set -euo pipefail
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0092_cross_dataset_training"
test -s "$ROOT/trained_models/SW0092_our_on_official_s1/TRAINING_COMPLETED"
for epoch in 1 3 10; do
  while nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do sleep 30; done
  bash "$DIR/evaluate_our_official.sh" 1 1 "$epoch"
done
echo complete > "$ROOT/trained_models/SW0092_OUR_SEED1_EARLY_EVAL_COMPLETED"
