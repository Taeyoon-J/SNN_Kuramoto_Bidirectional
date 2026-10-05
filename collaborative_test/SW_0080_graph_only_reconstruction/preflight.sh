#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
OUT="$ROOT/trained_models/SW0080_graphonly_preflight_s1"
test -s "$CORE" -a -s "$GAMMA" -a -s "$HDF5"
[[ ! -e "$OUT" ]] || { echo "refusing existing preflight output" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u collaborative_test/SW_0080_graph_only_reconstruction/train_graph.py \
 --core-checkpoint "$CORE" --gamma-path "$GAMMA" --dataset-path "$HDF5" \
 --output-dir "$OUT/model" --seed 1 --count 2500 --epochs 1 --batch-size 16 \
 --lr 0.00003 --reconstruction-weight 1.0 --max-steps 1 --device cuda > "$OUT/smoke.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python - "$OUT/model/manifest.json" "$CORE" "$GAMMA" \
 "$ROOT/collaborative_test/SW_0080_graph_only_reconstruction/train_graph.py" "$OUT/PREFLIGHT_VALIDATED.json" <<'PY'
import hashlib, json, pathlib, sys
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m['steps_completed']==1 and m['ground_truth_used_for_training'] is False
assert m['freeze_contract']['encoder_frozen'] and m['freeze_contract']['non_graph_core_frozen']
assert m['freeze_contract']['non_graph_max_abs_change']==0 and m['freeze_contract']['graph_max_abs_change']>0
record={'trainer_sha256':sha(sys.argv[4]),'source_core_sha256':sha(sys.argv[2]),'gamma_sha256':sha(sys.argv[3]),'preflight_manifest':sys.argv[1]}
pathlib.Path(sys.argv[5]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
