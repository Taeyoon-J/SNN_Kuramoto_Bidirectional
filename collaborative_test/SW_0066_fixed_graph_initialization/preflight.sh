#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"; case "$GPU" in 0|1) ;; *) exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw; PY=/Data0/kevinswk/envs/snn/bin/python
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
OUT="$ROOT/trained_models/SW0066_preflight"
[[ ! -e "$OUT" ]] || { echo "existing preflight" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"; [[ ! "$PIDS" =~ [0-9] ]]
mkdir -p "$OUT"; trap 'rm -rf "$OUT"' EXIT
"$PY" - "$GAMMA" "$OUT/one.pt" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu',weights_only=True); assert tuple(x.shape)==(2500,8,256) and torch.isfinite(x).all(); torch.save(x[:1],sys.argv[2])
PY
cd "$ROOT"; export CUDA_VISIBLE_DEVICES="$GPU"
"$PY" -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
 --gamma-seq-path "$OUT/one.pt" --save-path "$OUT/core.pt" --num-regions 256 --num-feature-maps 8 \
 --device cuda --epochs 1 --batch-size 1 --lr 0.0003 --seed 1 --graph-init-seed 0 \
 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 --theta-init gamma \
 --gamma-phase-mode standardize_tanh --freq-gain 2 --graph-mode learned --graph-top-k 32 \
 --graph-spatial-decay 0.35 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 \
 --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
 --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
 --low-n -4 --high-n 0 --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
 --spike-plv-weight 5 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
 --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 --structural-weight 0 \
 --plv-collapse-weight 1 --plv-bimodality-weight 6 --plv-balance-weight 10 \
 --plv-target-density 0.867 --plv-coherence-weight 0.5 --spike-per-component \
 --dendritic-projection shared --verbose > "$OUT/train.log" 2>&1
test -s "$OUT/core.pt"; grep -q 'Epoch 0001/0001' "$OUT/train.log"; echo passed
