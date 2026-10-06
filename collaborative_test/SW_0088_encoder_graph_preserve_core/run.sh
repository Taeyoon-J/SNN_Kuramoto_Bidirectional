#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0-3 required}"; ARM="${2:?E/F/G required}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
case "$ARM" in E) RECON=0.0 ;; F) RECON=0.3 ;; G) RECON=1.0 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0088_encoder_graph_preserve_core"
OUT="$ROOT/trained_models/SW0088_${ARM}_seed1"; CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
test -s "$CORE" -a -s "$GAMMA" -a -s "$ENCODER" -a -s "$STATS" -a -s "$HDF5"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_encoder_graph.py" \
 --dataset "$HDF5" --anchor-gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS" \
 --output-dir "$OUT/model" --epoch-checkpoint-dir "$OUT/epochs" --arm "$ARM" \
 --seed 1 --graph-init-seed 0 --core-checkpoint "$CORE" --epochs 5 --warmup-epochs 0 \
 --batch-size 16 --core-lr 0.00003 --encoder-lr 0.00003 \
 --slot-reconstruction-weight "$RECON" --slot-num-slots 7 --slot-temperature 0.3 \
 --max-samples 2500 --device cuda > "$OUT/training.log" 2>&1
EP="$OUT/epochs/epoch_05"; test -s "$EP/core.pt" -a -s "$EP/encoder.pt" -a -s "$OUT/model/manifest.json"
/Data0/kevinswk/envs/snn/bin/python "$ROOT/collaborative_test/SW_0068_joint_feature_core/export_gamma.py" \
 --dataset "$HDF5" --encoder "$EP/encoder.pt" --stats "$STATS" --start 1320 --count 32 \
 --output "$EP/gamma_validation.pt" --manifest "$EP/gamma_manifest.json" --device cuda \
 > "$EP/export.log" 2>&1
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"
