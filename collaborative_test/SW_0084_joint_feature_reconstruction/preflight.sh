#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0-3 required}"; ARM="${2:?arm A/B/C/D required}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
case "$ARM" in
 A) RECON=1.0; STARTER=fresh ;;
 B) RECON=0.0; STARTER=fresh ;;
 C) RECON=1.0; STARTER=sw0072 ;;
 D) RECON=0.3; STARTER=fresh ;;
 *) exit 2 ;;
esac
ROOT=/Data0/kevinswk/patch_v2_sw; DIR="$ROOT/collaborative_test/SW_0084_joint_feature_reconstruction"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
ENCODER=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/input_encoder/input_layer_encoder.pt
STATS=/Data0/kevinswk/patch_v2/trained_models/readme_patch_eval_20261002/feature_preprocessing.pt
HDF5=/Data0/kevinswk/datasets/object_centric_data/clevr_10-full.hdf5
CORE="$ROOT/trained_models/SW0072_frozen_seed0_graph_s1_e10/core.pt"
OUT="$ROOT/trained_models/SW0084_preflight_${ARM}"
test -s "$GAMMA" -a -s "$ENCODER" -a -s "$STATS" -a -s "$HDF5"
if [[ "$STARTER" == sw0072 ]]; then test -s "$CORE"; fi
[[ ! -e "$OUT" ]] || { echo "refusing existing preflight output: $OUT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
ARGS=(--dataset "$HDF5" --anchor-gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS"
 --output-dir "$OUT/model" --arm "$ARM" --seed 1 --graph-init-seed 0 --epochs 1 --warmup-epochs 0
 --batch-size 16 --core-lr 0.0003 --encoder-lr 0.00003 --slot-reconstruction-weight "$RECON"
 --slot-num-slots 7 --slot-temperature 0.3 --max-samples 2500 --max-steps 1 --device cuda)
if [[ "$STARTER" == sw0072 ]]; then ARGS+=(--core-checkpoint "$CORE"); fi
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_joint.py" "${ARGS[@]}" > "$OUT/preflight.log" 2>&1
VALIDATE=(--arm "$ARM" --manifest "$OUT/model/manifest.json" --model-dir "$OUT/model"
 --root "$ROOT" --gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS" --trainer "$DIR/train_joint.py")
if [[ "$STARTER" == sw0072 ]]; then VALIDATE+=(--initial-core "$CORE"); fi
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_preflight.py" "${VALIDATE[@]}" > "$OUT/preflight_validation.json"
/Data0/kevinswk/envs/snn/bin/python - "$OUT" "$OUT/model/manifest.json" "$DIR/train_joint.py" <<'PY'
import hashlib,json,pathlib,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
out=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2]); m=json.loads(manifest.read_text())
marker={'arm':m['experiment_arm'],'manifest_sha256':sha(manifest),'trainer_sha256':sha(sys.argv[3]),
 'gamma_sha256':m['anchor_gamma_sha256'],'encoder_sha256':m['source_encoder_sha256'],
 'stats_sha256':m['stats_sha256'],'source_core_sha256':m['source_core_sha256'],
 'code_dependencies_sha256':m['code_dependencies_sha256'],
 'reconstruction_weight':m['slot_reconstruction_weight'],
 'steps_completed':m['steps_completed'],'validation':'finite loss and joint updates verified'}
(out/'PREFLIGHT_VALIDATED.json').write_text(json.dumps(marker,indent=2)+'\n')
PY
