#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1}"; TAG="${2:?anchor0 or anchor100}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$TAG" in anchor0) AW=0 ;; anchor100) AW=100 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; OUT="$ROOT/trained_models/SW0069_frozen_s1_${TAG}"
test ! -e "$OUT"; p=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1); [[ ! "$p" =~ [0-9] ]]
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"; cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0068_joint_feature_core/train_joint.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 \
 --anchor-gamma "$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt" \
 --encoder /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --core-checkpoint "$ROOT/trained_models/SW0066_graphinit0_s1_e10/core.pt" --freeze-core \
 --output-dir "$OUT/model" --seed 1 --graph-init-seed 0 --epochs 5 --warmup-epochs 0 \
 --batch-size 16 --core-lr .0003 --encoder-lr .00003 --anchor-weight "$AW" --device cuda > "$OUT/training.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0068_joint_feature_core/export_gamma.py \
 --dataset /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --encoder "$OUT/model/encoder.pt" \
 --stats /Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt \
 --start 1320 --count 320 --output "$OUT/gamma_validation.pt" --manifest "$OUT/gamma_manifest.json" --device cuda > "$OUT/export.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0040_peer_transfer/evaluate.py" \
 --checkpoint "$OUT/model/core.pt" --gamma-path "$OUT/gamma_validation.pt" --gamma-global-start 1320 --gamma-manifest "$OUT/gamma_manifest.json" \
 --dataset-path /Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5 --output-path "$OUT/eval_n32.json" \
 --start 1320 --count 32 --steps 256 --settle 64 --membrane-vth .06 --min-group-size 2 --background largest_component --thresholds .35 \
 --dendritic-projection shared --graph-spatial-decay .35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature .5 --geodesic-cap 16 --kuramoto-backend factorized --gate-mode raw --phase-endpoint --device cuda > "$OUT/eval_n32.log" 2>&1
printf 'completed\n' > "$OUT/COMPLETED"
