#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?gpu}"; SEED="${2:?seed}"; EPOCH="${3:?epoch}"; ROOT=/Data0/kevinswk/patch_v2_sw
case "$SEED" in 0) TRAIN="$ROOT/trained_models/SW0092_slot_our70000_s0_final";; 1|2) TRAIN="$ROOT/trained_models/SW0092_slot_our70000_s${SEED}";; *) exit 2;; esac
STEP=$((2188 * EPOCH)); OUT="$ROOT/trained_models/SW0092_slot_our70000_eval/seed${SEED}_epoch${EPOCH}"
test -s "$TRAIN/checkpoint/ckpt-${STEP}.index"; test ! -e "$OUT"; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU" SW0092_ALLOW_GPU=1 TF_FORCE_GPU_ALLOW_GROWTH=true
/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python -u "$ROOT/collaborative_test/SW_0092_cross_dataset_training/slot_predict_gpu.py" \
 --checkpoint-dir "$TRAIN/checkpoint" --checkpoint-prefix "ckpt-${STEP}" \
 --checkpoint-source "SW0092 Slot scratch on our 70000 unique scenes epoch ${EPOCH}" \
 --training-seed "$SEED" --inference-seed 0 --training-protocol "$TRAIN/training_protocol.json" \
 --tf-intra-threads 2 --tf-inter-threads 1 \
 --model-py /Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --start 1320 --count 320 \
 --output-dir "$OUT" > "$OUT/inference.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python -u "$ROOT/collaborative_test/SW_0092_cross_dataset_training/score_predictions.py" \
 --predictions "$OUT/predictions.npz" --protocol "$OUT/protocol.json" \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --start 1320 --count 320 \
 --output-dir "$OUT" > "$OUT/scoring.log" 2>&1
test -s "$OUT/evaluation_summary.json" -a -s "$OUT/SCORING_COMPLETED"

