#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?Pass seed 0, 1, or 2}"
MODE="${2:-full}"
[[ "$SEED" =~ ^[0-2]$ ]] || { echo "seed must be 0, 1, or 2" >&2; exit 2; }
[[ "$MODE" == full || "$MODE" == smoke || "$MODE" == dry-run ]] || { echo "mode must be full, smoke, or dry-run" >&2; exit 2; }

ROOT=/Data0/kevinswk/patch_v2_sw
RUN_DIR="$ROOT/collaborative_test/SW_0056_matched_slot_2500"
MODEL_PY=/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py
DATASET=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
TF_PY=/Data0/kevinswk/envs/slot_attention_eval_tf215/bin/python
SNN_PY=/Data0/kevinswk/envs/snn/bin/python
OUT="$ROOT/trained_models/SW0056_matched_slot_2500_seed${SEED}"
if [[ "$MODE" == smoke ]]; then OUT="$OUT-smoke"; fi
if [[ "$MODE" == dry-run ]]; then
  printf 'DRY RUN: seed=%s train_ids=0-999,1640-3139 val_ids=1320-1639 exposures=25000 batch=16 updates=1563 final_batch=8 CPU-only\n' "$SEED"
  exit 0
fi
[[ ! -e "$OUT" ]] || { echo "refusing existing output $OUT" >&2; exit 1; }
test -s "$MODEL_PY" && test -s "$DATASET" && test -x "$TF_PY" && test -x "$SNN_PY"

export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
if [[ "$MODE" == smoke ]]; then
  nice -n 10 "$TF_PY" -u "$RUN_DIR/train_seed.py" --dataset "$DATASET" \
    --model-py "$MODEL_PY" --output-dir "$OUT" --seed "$SEED" --smoke \
    > "$OUT/training.log" 2>&1
  echo "SW0056 CPU smoke complete: $OUT"
  exit 0
fi

nice -n 10 "$TF_PY" -u "$RUN_DIR/train_seed.py" --dataset "$DATASET" \
  --model-py "$MODEL_PY" --output-dir "$OUT" --seed "$SEED" \
  > "$OUT/training.log" 2>&1
test -s "$OUT/checkpoint/ckpt-1563.index"
test -s "$OUT/TRAINING_COMPLETED"

VAL_OUT="$OUT/validation1320_1639"
for artifact in predictions.npz protocol.json evaluation_summary.json per_image.csv patch_masks.pt INFERENCE_COMPLETED SCORING_COMPLETED FAILED; do
  [[ ! -e "$VAL_OUT/$artifact" ]] || { echo "refusing existing validation artifact $VAL_OUT/$artifact" >&2; exit 1; }
done
mkdir -p "$VAL_OUT"
nice -n 10 "$TF_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/slot_attention_checkpoint_predict.py" \
  --checkpoint-dir "$OUT/checkpoint" --checkpoint-prefix ckpt-1563 \
  --checkpoint-source "SW0056 scratch matched-data 10-pass budget" \
  --training-seed "$SEED" --inference-seed 0 --training-protocol "$OUT/training_protocol.json" \
  --tf-intra-threads 2 --tf-inter-threads 1 \
  --model-py "$MODEL_PY" --dataset "$DATASET" --start 1320 --count 320 \
  --output-dir "$VAL_OUT" > "$VAL_OUT/inference.log" 2>&1
nice -n 10 "$SNN_PY" -u "$ROOT/collaborative_test/SW_0046_aligned_slot_audit/score_predictions.py" \
  --predictions "$VAL_OUT/predictions.npz" --protocol "$VAL_OUT/protocol.json" \
  --dataset "$DATASET" --start 1320 --count 320 --output-dir "$VAL_OUT" \
  > "$VAL_OUT/scoring.log" 2>&1
test -s "$VAL_OUT/evaluation_summary.json"
printf 'completed\n' > "$OUT/COMPLETED"
echo "SW0056 seed $SEED complete: $OUT"
