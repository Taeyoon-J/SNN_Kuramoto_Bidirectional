#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1}"; TAG="${2:?w0p3 or w1}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$TAG" in w0p3) WEIGHT=0.3 ;; w1) WEIGHT=1.0 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0073_s1_${TAG}"; SOURCE="$ROOT/trained_models/SW0055_unique2500_s0_e10_lr0p0003/core.pt"
test -s "$SOURCE"; [[ ! -e "$OUT" ]] || exit 3; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"; cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0068_joint_feature_core/train_joint.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
 --anchor-gamma "$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt" \
 --encoder /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --output-dir "$OUT/model" --seed 1 --graph-checkpoint "$SOURCE" --freeze-graph \
 --epochs 10 --warmup-epochs 2 --batch-size 16 --core-lr 0.0003 --encoder-lr 0.00003 \
 --slot-reconstruction-weight "$WEIGHT" --slot-num-slots 7 --slot-temperature 0.3 --device cuda > "$OUT/training.log" 2>&1
test -s "$OUT/model/core.pt" -a -s "$OUT/model/encoder.pt" -a -s "$OUT/model/manifest.json"
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0068_joint_feature_core/export_gamma.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --encoder "$OUT/model/encoder.pt" \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --start 1320 --count 320 --output "$OUT/gamma_validation.pt" --manifest "$OUT/gamma_manifest.json" \
 --device cuda > "$OUT/export.log" 2>&1
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"

