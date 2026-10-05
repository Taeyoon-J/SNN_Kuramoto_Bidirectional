#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"; TAG="${2:?g1k or g10k required}"
case "$GPU" in 0|1) ;; *) echo "GPU must be 0 or 1" >&2; exit 2 ;; esac
case "$TAG" in g1k) WEIGHT=1000 ;; g10k) WEIGHT=10000 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW0076_s1_${TAG}"
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
DATASET=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
test -s "$CORE" -a -s "$GAMMA" -a -s "$ENCODER" -a -s "$STATS" -a -s "$DATASET"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
{
  printf 'experiment=SW0076_graph_output_anchor\nseed=1\ngpu=%s\nweight=%s\n' "$GPU" "$WEIGHT"
  printf 'core_sha256='; sha256sum "$CORE" | cut -d' ' -f1
  printf 'gamma_sha256='; sha256sum "$GAMMA" | cut -d' ' -f1
  printf 'script_sha256='; sha256sum "$ROOT/collaborative_test/SW_0076_graph_output_anchor/run.sh" | cut -d' ' -f1
} > "$OUT/launch_manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0068_joint_feature_core/train_joint.py \
 --dataset "$DATASET" --anchor-gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS" \
 --output-dir "$OUT/model" --seed 1 --core-checkpoint "$CORE" --freeze-core \
 --epochs 5 --warmup-epochs 0 --batch-size 16 --core-lr .0003 --encoder-lr .00003 \
 --activity-anchor-weight 0 --graph-output-anchor-weight "$WEIGHT" \
 --slot-reconstruction-weight 1 --slot-num-slots 7 --slot-temperature .3 --device cuda \
 > "$OUT/training.log" 2>&1
test -s "$OUT/model/core.pt" -a -s "$OUT/model/encoder.pt" -a -s "$OUT/model/manifest.json"
/Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0068_joint_feature_core/export_gamma.py \
 --dataset "$DATASET" --encoder "$OUT/model/encoder.pt" --stats "$STATS" \
 --start 1320 --count 320 --output "$OUT/gamma_validation.pt" \
 --manifest "$OUT/gamma_manifest.json" --device cuda > "$OUT/export.log" 2>&1
echo completed > "$OUT/TRAINING_COMPLETED"
