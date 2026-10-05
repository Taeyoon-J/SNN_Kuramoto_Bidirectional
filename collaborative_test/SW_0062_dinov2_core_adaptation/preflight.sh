#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DATA="$ROOT/data/SW_0062_dinov2_core_adaptation"
GAMMA="$DATA/dino_gamma_train_0_999.pt"
MANIFEST="$DATA/manifest.json"
MARKER="$DATA/TRAINING_PREFLIGHT_V1.txt"
PY=/Data0/kevinswk/envs/snn/bin/python
test -s "$GAMMA" -a -s "$MANIFEST"
GAMMA_SHA="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_SHA="$(sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" \
  "$ROOT/snn_kuramoto_bidirectional/s2net_cls.py" "$ROOT/snn_kuramoto_bidirectional/loss_function.py" \
  "$ROOT/collaborative_test/SW_0062_dinov2_core_adaptation/run_pilot.sh" | sha256sum | awk '{print $1}')"
EXPECTED="SW0062_TRAINING_PREFLIGHT_V1 $GAMMA_SHA $CODE_SHA"
if [[ -s "$MARKER" ]] && grep -Fxq "$EXPECTED" "$MARKER"; then echo current; exit 0; fi
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied" >&2; exit 2; }
TMP="$(mktemp -d "$ROOT/trained_models/.SW0062_preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
"$PY" - "$GAMMA" "$MANIFEST" "$GAMMA_SHA" "$TMP/one.pt" <<'PY'
import json,sys,torch
g=torch.load(sys.argv[1],map_location='cpu',weights_only=True); m=json.load(open(sys.argv[2]))
assert tuple(g.shape)==(1000,8,256) and torch.isfinite(g).all()
assert m['pca_fit_ids']==[0,999] and m['validation_ids']==[1320,1639]
assert m['uses_masks_counts_or_labels'] is False and m['sha256']['train']==sys.argv[3]
torch.save(g[:1].clone(),sys.argv[4])
PY
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$TMP/one.pt" --save-path "$TMP/core.pt" --num-regions 256 \
  --num-feature-maps 8 --device cuda --epochs 1 --batch-size 1 --lr 0.0003 --seed 0 \
  --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2 --graph-mode learned \
  --graph-top-k 32 --graph-spatial-decay 0.35 --geodesic-steps 3 --geodesic-radius 1.5 \
  --geodesic-contrast 2 --geodesic-temperature 0.5 --geodesic-cap 16 \
  --kuramoto-backend factorized --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 --branch 4 \
  --gate-mode raw --plv-source phase --plv-combine mean --spike-plv-weight 5 \
  --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1 --plv-bimodality-weight 6 \
  --plv-balance-weight 10 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose > "$TMP/train.log" 2>&1
test -s "$TMP/core.pt"; grep -q 'Epoch 0001/0001' "$TMP/train.log"
printf '%s\n' "$EXPECTED" > "$MARKER.tmp.$$"; mv "$MARKER.tmp.$$" "$MARKER"
echo "SW0062 preflight passed"
