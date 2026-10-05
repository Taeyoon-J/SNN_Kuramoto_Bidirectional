#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?Pass seed 0, 1, or 2}"
MODE="${2:-full}"
if [[ ! "$SEED" =~ ^[0-2]$ ]]; then echo "SEED must be 0, 1, or 2" >&2; exit 2; fi
if [[ "$MODE" != full && "$MODE" != smoke && "$MODE" != dry-run ]]; then
  echo "MODE must be full, smoke, or dry-run" >&2; exit 2
fi
ROOT=/Data0/kevinswk/patch_v2_sw
RUN_DIR="$ROOT/collaborative_test/SW_0048_matched_slot_attention"
ASSET_DIR=/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002
MODEL_PY="$ASSET_DIR/model.py"
DATASET=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW_0048_slot_attention_seed${SEED}"
SNN_PY=/Data0/kevinswk/envs/snn/bin/python
TF_PY=/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python

if [[ "$MODE" == dry-run ]]; then
  printf 'DRY RUN: seed=%s mode=%s output=%s train_ids=0-999 steps=2500 batch=16 passes=40 validation_ids=1320-1639 CPU-only\n' "$SEED" "$MODE" "$OUT"
  exit 0
fi
if [[ "$MODE" == smoke ]]; then
  OUT="$OUT-smoke"
  TRAIN_ARGS=(--smoke)
else
  TRAIN_ARGS=()
fi
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing SW0048 seed output: $OUT" >&2
  exit 1
fi
test -s "$MODEL_PY"
test -s "$DATASET"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
mkdir "$OUT"
"$TF_PY" -u "$RUN_DIR/train_seed.py" \
  --dataset "$DATASET" --model-py "$MODEL_PY" --output-dir "$OUT" \
  --seed "$SEED" "${TRAIN_ARGS[@]}" > "$OUT/training.log" 2>&1

if [[ "$MODE" == smoke ]]; then
  echo "SW0048 smoke training completed; full validation is intentionally skipped: $OUT"
  exit 0
fi

VAL_OUT="$OUT/validation1320_1639"
for artifact in predictions.npz protocol.json evaluation_summary.json per_image.csv patch_masks.pt INFERENCE_COMPLETED SCORING_COMPLETED FAILED; do
  if [[ -e "$VAL_OUT/$artifact" ]]; then echo "Refusing to overwrite validation artifact: $VAL_OUT/$artifact" >&2; exit 1; fi
done
mkdir -p "$VAL_OUT"
"$TF_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/slot_attention_checkpoint_predict.py" \
  --checkpoint-dir "$OUT/checkpoint" --checkpoint-prefix "ckpt-2500" \
  --checkpoint-source "SW0048 scratch matched-data/pass-budget candidate" \
  --training-seed "$SEED" --inference-seed 0 --training-protocol "$OUT/training_protocol.json" \
  --model-py "$MODEL_PY" --dataset "$DATASET" \
  --start 1320 --count 320 --output-dir "$VAL_OUT" \
  > "$VAL_OUT/inference.log" 2>&1
"$SNN_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/score_predictions.py" \
  --predictions "$VAL_OUT/predictions.npz" --protocol "$VAL_OUT/protocol.json" \
  --dataset "$DATASET" --start 1320 --count 320 --output-dir "$VAL_OUT" \
  > "$VAL_OUT/scoring.log" 2>&1
test -s "$VAL_OUT/evaluation_summary.json"
printf 'completed\n' > "$OUT/COMPLETED"
echo "SW0048 seed $SEED training and validation completed: $OUT"
