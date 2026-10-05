#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU 0 or 1 required}"
case "$GPU" in 0|1) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0081_graph_flip_equivariance"
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
OUT="$ROOT/trained_models/SW0081_preflight_seed1"
test -s "$CORE" -a -s "$GAMMA" -a -s "$HDF5" -a -s "$ENCODER" -a -s "$STATS"
[[ ! -e "$OUT" ]] || { echo "refusing existing preflight output: $OUT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU is busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_graph.py" \
 --core-checkpoint "$CORE" --gamma-path "$GAMMA" --dataset-path "$HDF5" \
 --encoder "$ENCODER" --stats "$STATS" --output-dir "$OUT/model" \
 --seed 1 --count 2500 --epochs 1 --batch-size 16 --lr 0.00003 \
 --propose-weight-0p1x --max-steps 1 --device cuda > "$OUT/preflight.log" 2>&1
/Data0/kevinswk/envs/snn/bin/python - "$OUT/model/manifest.json" "$CORE" "$GAMMA" "$ENCODER" "$STATS" \
 "$DIR/train_graph.py" "$DIR/equivariance.py" "$ROOT" "$OUT/PREFLIGHT_VALIDATED.json" <<'PY'
import hashlib,json,pathlib,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m['steps_completed']==1 and m['ground_truth_used'] is False
assert m['freeze_contract']['encoder_exact_unchanged']
assert m['freeze_contract']['non_graph_core_exact_unchanged']
assert m['freeze_contract']['graph_max_abs_change']>0
scale=m['preflight_scale_measurement']
assert scale['loss_finite'] and scale['initial_raw_equivariance_mse']>=0
assert scale['raw_equivariance_gradient_norm']>0
deps=('snn_kuramoto_bidirectional/s2net_cls.py','snn_kuramoto_bidirectional/graph_generator.py',
 'snn_kuramoto_bidirectional/input_layer_generator.py','snn_kuramoto_bidirectional/gamma_initializer.py',
 'snn_kuramoto_bidirectional/loss_function.py','snn_kuramoto_bidirectional/training/train_s2net_core.py')
dependency_hashes={name:sha(pathlib.Path(sys.argv[8])/name) for name in deps}
assert dependency_hashes==m['code_dependencies_sha256']
record={'source_core_sha256':sha(sys.argv[2]),'gamma_sha256':sha(sys.argv[3]),
 'encoder_sha256':sha(sys.argv[4]),'stats_sha256':sha(sys.argv[5]),
 'trainer_sha256':sha(sys.argv[6]),'equivariance_impl_sha256':sha(sys.argv[7]),
 'code_dependencies_sha256':dependency_hashes,
 'measured':scale,'weights':scale['proposed_weights_for_gradient_ratios'],
 'preflight_manifest':sys.argv[1]}
pathlib.Path(sys.argv[9]).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
