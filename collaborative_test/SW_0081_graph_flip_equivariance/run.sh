#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"; ARM="${2:?w0p1 or w1x required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$ARM" in w0p1|w1x) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0081_graph_flip_equivariance"
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
PREFLIGHT="$ROOT/trained_models/SW0081_preflight_seed1/PREFLIGHT_VALIDATED.json"
OUT="$ROOT/trained_models/SW0081_graphflip_seed1_${ARM}"
test -s "$PREFLIGHT" -a -s "$CORE" -a -s "$GAMMA" -a -s "$HDF5" -a -s "$ENCODER" -a -s "$STATS"
[[ ! -e "$OUT" ]] || { echo "refusing existing arm output: $OUT" >&2; exit 3; }
/Data0/kevinswk/envs/snn/bin/python - "$PREFLIGHT" "$CORE" "$GAMMA" "$ENCODER" "$STATS" \
 "$DIR/train_graph.py" "$DIR/equivariance.py" "$ROOT" <<'PY'
import hashlib,json,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
m=json.load(open(sys.argv[1]))
for key,path in zip(('source_core_sha256','gamma_sha256','encoder_sha256','stats_sha256','trainer_sha256','equivariance_impl_sha256'),sys.argv[2:]):
 assert m[key]==sha(path), 'stale preflight for '+key
for name,digest in m['code_dependencies_sha256'].items():
 assert sha(sys.argv[8]+'/'+name)==digest, 'stale preflight dependency '+name
PY
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
if [[ "$ARM" == w0p1 ]]; then KEY=weight_0p1x; RATIO=0.1x; else KEY=weight_1x; RATIO=1x; fi
WEIGHT="$(/Data0/kevinswk/envs/snn/bin/python - "$PREFLIGHT" "$KEY" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['weights'][sys.argv[2]])
PY
)"
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
printf 'experiment=SW0081_graph_flip_equivariance\narm=%s\nseed=1\ngpu=%s\nweight=%s\ntarget_ratio=%s\n' "$ARM" "$GPU" "$WEIGHT" "$RATIO" > "$OUT/launch_manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_graph.py" \
 --core-checkpoint "$CORE" --gamma-path "$GAMMA" --dataset-path "$HDF5" \
 --encoder "$ENCODER" --stats "$STATS" --output-dir "$OUT/model" \
 --seed 1 --start 0 --count 2500 --epochs 5 --batch-size 16 --lr 0.00003 \
 --equivariance-weight "$WEIGHT" --device cuda > "$OUT/training.log" 2>&1
test -f "$OUT/model/TRAINING_COMPLETED" -a -s "$OUT/model/core.pt" -a -s "$OUT/model/manifest.json"
