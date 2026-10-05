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
PREFLIGHT="$ROOT/trained_models/SW0084_preflight_${ARM}"
OUT="$ROOT/trained_models/SW0084_${ARM}_seed1"
test -s "$PREFLIGHT/PREFLIGHT_VALIDATED.json" -a -s "$PREFLIGHT/model/manifest.json"
test -s "$GAMMA" -a -s "$ENCODER" -a -s "$STATS" -a -s "$HDF5"
if [[ "$STARTER" == sw0072 ]]; then test -s "$CORE"; fi
[[ ! -e "$OUT" ]] || { echo "refusing existing arm output: $OUT" >&2; exit 3; }
VALIDATE=(--arm "$ARM" --manifest "$PREFLIGHT/model/manifest.json" --model-dir "$PREFLIGHT/model"
 --root "$ROOT" --gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS" --trainer "$DIR/train_joint.py")
if [[ "$STARTER" == sw0072 ]]; then VALIDATE+=(--initial-core "$CORE"); fi
/Data0/kevinswk/envs/snn/bin/python "$DIR/validate_preflight.py" "${VALIDATE[@]}" > /dev/null
/Data0/kevinswk/envs/snn/bin/python - "$PREFLIGHT/PREFLIGHT_VALIDATED.json" "$PREFLIGHT/model/manifest.json" "$ARM" <<'PY'
import hashlib,json,sys
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
marker=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
assert marker['manifest_sha256']==sha(sys.argv[2]) and marker['arm']==sys.argv[3]
assert marker['trainer_sha256']==m['trainer_sha256']
assert marker['gamma_sha256']==m['anchor_gamma_sha256']
assert marker['encoder_sha256']==m['source_encoder_sha256'] and marker['stats_sha256']==m['stats_sha256']
assert marker['source_core_sha256']==m['source_core_sha256']
PY
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU busy" >&2; exit 4; }
mkdir -p "$OUT/cache"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
ARGS=(--dataset "$HDF5" --anchor-gamma "$GAMMA" --encoder "$ENCODER" --stats "$STATS"
 --output-dir "$OUT/model" --epoch-checkpoint-dir "$OUT/epochs" --arm "$ARM"
 --seed 1 --graph-init-seed 0 --epochs 10 --warmup-epochs 0 --batch-size 16
 --core-lr 0.0003 --encoder-lr 0.00003 --slot-reconstruction-weight "$RECON"
 --slot-num-slots 7 --slot-temperature 0.3 --max-samples 2500 --device cuda)
if [[ "$STARTER" == sw0072 ]]; then ARGS+=(--core-checkpoint "$CORE"); fi
printf 'SW0084 arm=%s seed=1 GPU=%s reconstruction_weight=%s\n' "$ARM" "$GPU" "$RECON" > "$OUT/launch_manifest.txt"
/Data0/kevinswk/envs/snn/bin/python -u "$DIR/train_joint.py" "${ARGS[@]}" > "$OUT/training.log" 2>&1
test -s "$OUT/model/core.pt" -a -s "$OUT/model/encoder.pt" -a -s "$OUT/model/manifest.json"
for EPOCH in 05 10; do
  EP="$OUT/epochs/epoch_${EPOCH}"; test -s "$EP/core.pt" -a -s "$EP/encoder.pt"
  /Data0/kevinswk/envs/snn/bin/python collaborative_test/SW_0068_joint_feature_core/export_gamma.py \
    --dataset "$HDF5" --encoder "$EP/encoder.pt" --stats "$STATS" --start 1320 --count 32 \
    --output "$EP/gamma_validation.pt" --manifest "$EP/gamma_manifest.json" --device cuda \
    > "$EP/export.log" 2>&1
done
printf 'completed\n' > "$OUT/TRAINING_COMPLETED"
