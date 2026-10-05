#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU}"; TAG="${2:?a10 or a100}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac; case "$TAG" in a10) ANCHOR=10 ;; a100) ANCHOR=100 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0074_s1_${TAG}"; CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
test -s "$CORE"; [[ ! -e "$OUT" ]] || exit 3; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"; cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0068_joint_feature_core/train_joint.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --anchor-gamma "$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt" \
 --encoder /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --output-dir "$OUT/model" --seed 1 --core-checkpoint "$CORE" --freeze-core --epochs 5 --warmup-epochs 0 \
 --batch-size 16 --core-lr .0003 --encoder-lr .00003 --anchor-weight "$ANCHOR" \
 --slot-reconstruction-weight 1 --slot-num-slots 7 --slot-temperature .3 --device cuda > "$OUT/training.log" 2>&1
test -s "$OUT/model/core.pt" -a -s "$OUT/model/encoder.pt" -a -s "$OUT/model/manifest.json"
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0068_joint_feature_core/export_gamma.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --encoder "$OUT/model/encoder.pt" \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --start 1320 --count 320 --output "$OUT/gamma_validation.pt" --manifest "$OUT/gamma_manifest.json" --device cuda > "$OUT/export.log" 2>&1
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"

