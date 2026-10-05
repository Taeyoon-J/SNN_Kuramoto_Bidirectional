#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 only}"; [[ "$GPU" == 0 ]] || { echo 'SW0082 is GPU0-only' >&2; exit 2; }
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0082_graph_flip_trajectory"
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
SW0081="$ROOT/collaborative_test/SW_0081_graph_flip_equivariance"
PREFLIGHT="$ROOT/trained_models/SW0081_preflight_seed1/PREFLIGHT_VALIDATED.json"
OUT="$ROOT/trained_models/SW0082_graphflip_trajectory_seed1"
test -s "$PREFLIGHT" -a -s "$CORE" -a -s "$GAMMA" -a -s "$HDF5" -a -s "$ENCODER" -a -s "$STATS"
[[ ! -e "$OUT" ]] || { echo "refusing existing trajectory output: $OUT" >&2; exit 3; }
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_preflight.py" \
 --marker "$PREFLIGHT" --repo-root "$ROOT" --core "$CORE" --gamma "$GAMMA" \
 --encoder "$ENCODER" --stats "$STATS" --sw0081-dir "$SW0081" > /dev/null
WEIGHT="$(/Data0/kevinswk/envs/snn/bin/python - "$PREFLIGHT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['weights']['weight_0p1x'])
PY
)"
PIDS="$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo 'GPU0 is busy' >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES=0 TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
printf 'experiment=SW0082_graph_flip_epoch_trajectory\nseed=1\ngpu=0\nequivariance_weight=%s\nequivariance_target_ratio=0.1x\n' "$WEIGHT" > "$OUT/launch_manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_graph.py" \
 --core-checkpoint "$CORE" --gamma-path "$GAMMA" --dataset-path "$HDF5" \
 --encoder "$ENCODER" --stats "$STATS" --output-dir "$OUT/model" \
 --epoch-checkpoint-dir "$OUT/epochs" --preflight-marker "$PREFLIGHT" \
 --seed 1 --start 0 --count 2500 \
 --epochs 5 --batch-size 16 --lr 0.00003 --equivariance-weight "$WEIGHT" --device cuda \
 > "$OUT/training.log" 2>&1
test -f "$OUT/model/TRAINING_COMPLETED" -a -s "$OUT/model/core.pt" -a -s "$OUT/model/manifest.json"
test -s "$OUT/epochs/epoch_01_core.pt" -a -s "$OUT/epochs/epoch_05_core.pt"
