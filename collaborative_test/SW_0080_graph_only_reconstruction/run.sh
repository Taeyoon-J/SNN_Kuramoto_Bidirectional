#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"; TAG="${2:?r0p3 or r1p0 required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
case "$TAG" in r0p3) WEIGHT=.3 ;; r1p0) WEIGHT=1.0 ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
PREFLIGHT="$ROOT/trained_models/SW0080_graphonly_preflight_s1/PREFLIGHT_VALIDATED.json"
OUT="$ROOT/trained_models/SW0080_graphonly_s1_${TAG}"
test -s "$CORE" -a -s "$GAMMA" -a -s "$HDF5" -a -s "$PREFLIGHT"
[[ ! -e "$OUT" ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
/Data0/kevinswk/envs/snn/bin/python - "$PREFLIGHT" "$CORE" "$GAMMA" \
 "$ROOT/collaborative_test/SW_0080_graph_only_reconstruction/train_graph.py" <<'PY'
import hashlib,json,pathlib,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m['source_core_sha256']==sha(sys.argv[2]) and m['gamma_sha256']==sha(sys.argv[3])
assert m['trainer_sha256']==sha(sys.argv[4]), 'preflight is stale for current trainer'
PY
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
{
 printf 'experiment=SW0080_graph_only_reconstruction\nseed=1\ngpu=%s\nreconstruction_weight=%s\n' "$GPU" "$WEIGHT"
 printf 'source_core_sha256='; sha256sum "$CORE" | cut -d' ' -f1
 printf 'gamma_sha256='; sha256sum "$GAMMA" | cut -d' ' -f1
 printf 'trainer_sha256='; sha256sum "$ROOT/collaborative_test/SW_0080_graph_only_reconstruction/train_graph.py" | cut -d' ' -f1
} > "$OUT/launch_manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0080_graph_only_reconstruction/train_graph.py \
 --core-checkpoint "$CORE" --gamma-path "$GAMMA" --dataset-path "$HDF5" \
 --output-dir "$OUT/model" --seed 1 --start 0 --count 2500 --epochs 5 \
 --batch-size 16 --lr 0.00003 --reconstruction-weight "$WEIGHT" --device cuda \
 > "$OUT/training.log" 2>&1
test -f "$OUT/model/TRAINING_COMPLETED" -a -s "$OUT/model/core.pt" -a -s "$OUT/model/manifest.json"
