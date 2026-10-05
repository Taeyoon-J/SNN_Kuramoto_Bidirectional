#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0055_unique_data_scale"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
MANIFEST="$ROOT/data/SW_0055_unique_data_scale/manifest.json"
PY=/Data0/kevinswk/envs/snn/bin/python
MARKER="$ROOT/data/SW_0055_unique_data_scale/TRAINING_PREFLIGHT_V2.txt"
test -s "$GAMMA" -a -s "$MANIFEST" -a -x "$PY"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_FILES=(
  "$ROOT/snn_kuramoto_bidirectional/s2net_cls.py"
  "$ROOT/snn_kuramoto_bidirectional/graph_generator.py"
  "$ROOT/snn_kuramoto_bidirectional/kuramoto_layer.py"
  "$ROOT/snn_kuramoto_bidirectional/dendric_layer.py"
  "$ROOT/snn_kuramoto_bidirectional/membrane_layer.py"
  "$ROOT/snn_kuramoto_bidirectional/sinusoidal_gating.py"
  "$ROOT/snn_kuramoto_bidirectional/hyperparameter.py"
  "$ROOT/snn_kuramoto_bidirectional/loss_function.py"
  "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py"
  "$DIR/run.sh"
  "$DIR/preflight_training.sh"
)
CODE_SUM="$(sha256sum "${CODE_FILES[@]}" | sha256sum | awk '{print $1}')"
EXPECTED="SW0055_TRAINING_PREFLIGHT_V2 $GAMMA_SUM $CODE_SUM"
if [[ -s "$MARKER" ]] && grep -Fxq "$EXPECTED" "$MARKER"; then echo "current SW0055 training preflight exists"; exit 0; fi
PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU_ID occupied" >&2; exit 2; }
TMP="$(mktemp -d "$ROOT/trained_models/.SW0055_preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
$PY - "$GAMMA" "$MANIFEST" "$GAMMA_SUM" "$TMP/gamma_one.pt" <<'PY'
import json,sys,torch
g=torch.load(sys.argv[1],map_location='cpu',weights_only=True); m=json.load(open(sys.argv[2]))
assert tuple(g.shape)==(2500,8,256) and torch.isfinite(g).all()
assert m['gamma_sha256']==sys.argv[3] and m['training_ids']['segments']==[[0,999],[1640,3139]]
assert m['held_out_ids_excluded']==[1000,1639] and m['first_1000_max_abs_diff']==0.0
torch.save(g[:1].clone(),sys.argv[4])
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$TMP/gamma_one.pt" --save-path "$TMP/core.pt" \
  --num-regions 256 --num-feature-maps 8 --device cuda --epochs 1 --batch-size 1 \
  --lr 0.0003 --seed 0 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 \
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
  --spike-plv-weight 5.0 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
  --checkpoint-dir "$TMP/checkpoints" --checkpoint-epochs 1 \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose > "$TMP/train.log" 2>&1
test -s "$TMP/core.pt" -a -s "$TMP/checkpoints/epoch_01.pt"
grep -q 'Epoch 0001/0001' "$TMP/train.log"
printf '%s\n' "$EXPECTED" > "$MARKER.tmp.$$"; mv "$MARKER.tmp.$$" "$MARKER"
echo "SW0055 real-gamma one-update preflight passed"
