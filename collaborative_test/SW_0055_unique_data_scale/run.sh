#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?GPU ID required}"; SEED="${2:?seed required}"
case "$SEED" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
DIR="$ROOT/collaborative_test/SW_0055_unique_data_scale"
GAMMA="$ROOT/data/SW_0055_unique_data_scale/gamma_train_2500.pt"
MARKER="$ROOT/data/SW_0055_unique_data_scale/TRAINING_PREFLIGHT_V1.txt"
GAMMA_SUM="$(sha256sum "$GAMMA" | awk '{print $1}')"
CODE_SUM="$(sha256sum "$ROOT/snn_kuramoto_bidirectional/training/train_s2net_core.py" "$ROOT/snn_kuramoto_bidirectional/loss_function.py" "$DIR/run.sh" "$DIR/preflight_training.sh" | sha256sum | awk '{print $1}')"
grep -Fxq "SW0055_TRAINING_PREFLIGHT_V1 $GAMMA_SUM $CODE_SUM" "$MARKER" || { echo "current SW0055 preflight required" >&2; exit 3; }
PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader)"
[[ ! "$PIDS" =~ [0-9] ]] || { echo "GPU $GPU_ID occupied" >&2; exit 2; }
OUT="$ROOT/trained_models/SW0055_unique2500_s${SEED}_e10_lr0p0003"
[[ ! -e "$OUT" ]] || { echo "refusing existing $OUT" >&2; exit 1; }
mkdir -p "$OUT/cache"
export TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" /Data0/kevinswk/envs/snn/bin/python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" \
  --num-regions 256 --num-feature-maps 8 --device cuda --epochs 10 --batch-size 16 \
  --lr 0.0003 --seed "$SEED" --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 \
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.35 \
  --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2.0 \
  --geodesic-temperature 0.5 --geodesic-cap 16 --kuramoto-backend factorized \
  --spike-spatial-grid-size 16 --k 256 --membrane-vth 0.06 \
  --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --branch 4 --gate-mode raw --plv-source phase --plv-combine mean \
  --spike-plv-weight 5.0 --loss-signal sigmoid_membrane --sample-activity-diversity-weight 0 \
  --checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 10 \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 --plv-coherence-weight 0.5 \
  --spike-per-component --dendritic-projection shared --verbose > "$OUT/training.log" 2>&1
test -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_10.pt"
grep -Fq "trained S2NetCore: $OUT/core.pt" "$OUT/training.log"
echo completed > "$OUT/TRAINING_COMPLETED"
