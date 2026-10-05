#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU0/1 required}"
case "$GPU" in 0|1) ;; *) echo "GPU0/1 only" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
GAMMA="$ROOT/data/SW_0062_dinov2_core_adaptation/dino_gamma_train_0_999.pt"
OUT="$ROOT/trained_models/SW0062_dinov2_s0_e5_lr0p0003"
test -s "$GAMMA"
[[ ! -e "$OUT" ]] || { echo "refusing existing $OUT" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU occupied" >&2; exit 2; }
mkdir -p "$OUT/cache" "$OUT/checkpoints"
export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" --num-regions 256 \
  --num-feature-maps 8 --device cuda --epochs 5 --batch-size 16 --lr 0.0003 --seed 0 \
  --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 --branch 4 \
  --gate-mode raw --plv-source phase --plv-combine mean --spike-plv-weight 5 \
  --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
  --checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 5 \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1 --plv-bimodality-weight 6 \
  --plv-balance-weight 10 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose > "$OUT/training.log" 2>&1
test -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_05.pt"
echo completed > "$OUT/TRAINING_COMPLETED"
