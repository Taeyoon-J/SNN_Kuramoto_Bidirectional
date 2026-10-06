#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?gpu}"; SEED="${2:?seed}"
case "$GPU" in 1|2|3) ;; *) exit 2;; esac; case "$SEED" in 0|1|2) ;; *) exit 2;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0092_slot_our70000_s${SEED}"
test ! -e "$OUT"
p="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$p" =~ [0-9] ]]
export CUDA_VISIBLE_DEVICES="$GPU" TF_FORCE_GPU_ALLOW_GROWTH=true SW0056_ALLOW_GPU=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
/Data0/kevinswk/envs/slot_attention_gpu_tf215/bin/python -u \
 "$ROOT/collaborative_test/SW_0092_cross_dataset_training/train_slot_hdf5_70000.py" \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
 --model-py /Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/model.py \
 --output-dir "$OUT" --seed "$SEED" --epochs 10 --batch-size 32 \
 > "$ROOT/trained_models/SW0092_slot_our70000_s${SEED}.log" 2>&1
test -s "$OUT/TRAINING_COMPLETED"

