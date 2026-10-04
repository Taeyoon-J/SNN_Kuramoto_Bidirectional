#!/usr/bin/env bash
set -euo pipefail

ROOT=/Data0/kevinswk/patch_v2_sw
RUN_DIR="$ROOT/collaborative_test/SW_0046_aligned_slot_audit"
ASSET_DIR=/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002
CHECKPOINT_DIR="$ASSET_DIR/checkpoint"
MODEL_PY="$ASSET_DIR/model.py"
DATASET=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
START=1320
COUNT=320
OUTPUT_DIR="$RUN_DIR/results/seed0_validation1320_1639"
TF_PY=/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python
SNN_PY=/Data0/kevinswk/envs/snn/bin/python

for artifact in predictions.npz protocol.json evaluation_summary.json per_image.csv patch_masks.pt INFERENCE_COMPLETED SCORING_COMPLETED inference.log scoring.log RUNNING FAILED COMPLETED; do
  if [[ -e "$OUTPUT_DIR/$artifact" ]]; then
    echo "Refusing to overwrite existing SW0046 artifact: $OUTPUT_DIR/$artifact" >&2
    exit 1
  fi
done
test -s "$MODEL_PY"
test -s "$CHECKPOINT_DIR/ckpt-500.index"
test -s "$DATASET"
mkdir -p "$OUTPUT_DIR"
printf 'running\n' > "$OUTPUT_DIR/RUNNING"
trap 'status=$?; if [[ "$status" -ne 0 ]]; then printf "failed\n" > "$OUTPUT_DIR/FAILED"; fi; rm -f "$OUTPUT_DIR/RUNNING"' EXIT

"$TF_PY" -u "$RUN_DIR/slot_attention_checkpoint_predict.py" \
  --checkpoint-dir "$CHECKPOINT_DIR" --checkpoint-prefix ckpt-500 \
  --model-py "$MODEL_PY" --dataset "$DATASET" \
  --start "$START" --count "$COUNT" --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/inference.log" 2>&1

"$SNN_PY" -u "$RUN_DIR/score_predictions.py" \
  --predictions "$OUTPUT_DIR/predictions.npz" \
  --protocol "$OUTPUT_DIR/protocol.json" --dataset "$DATASET" \
  --start "$START" --count "$COUNT" --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/scoring.log" 2>&1
test -s "$OUTPUT_DIR/predictions.npz"
test -s "$OUTPUT_DIR/protocol.json"
test -s "$OUTPUT_DIR/evaluation_summary.json"
test -s "$OUTPUT_DIR/per_image.csv"
test -s "$OUTPUT_DIR/patch_masks.pt"
printf 'completed\n' > "$OUTPUT_DIR/COMPLETED"
echo "SW0046 completed: $OUTPUT_DIR"
