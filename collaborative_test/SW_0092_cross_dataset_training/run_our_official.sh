#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?gpu}"; SEED="${2:?seed}"
case "$GPU" in 0|1|3) ;; *) exit 2;; esac; case "$SEED" in 0|1|2) ;; *) exit 2;; esac
ROOT=/Data0/kevinswk/patch_v2_sw
GAMMA="$ROOT/data/SW_0092_cross_dataset/official_clevr6_gamma.pt"
SOURCE="$ROOT/trained_models/SW0055_unique2500_s0_e10_lr0p0003/core.pt"
OUT="$ROOT/trained_models/SW0092_our_on_official_s${SEED}"
test -s "$GAMMA" -a -s "$SOURCE"; test ! -e "$OUT"
while true; do
 p="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>&1)"
 [[ "$p" =~ [0-9] ]] || break
 sleep 30
done
mkdir -p "$OUT/cache" "$OUT/checkpoints"; export CUDA_VISIBLE_DEVICES="$GPU" TMPDIR="$OUT/cache" TRITON_CACHE_DIR="$OUT/cache"
cd "$ROOT"
/Data0/kevinswk/envs/snn/bin/python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
 --gamma-seq-path "$GAMMA" --save-path "$OUT/core.pt" --num-regions 256 --num-feature-maps 8 \
 --device cuda --epochs 10 --batch-size 16 --lr .0003 --seed "$SEED" --graph-checkpoint "$SOURCE" --freeze-graph \
 --osc-dim 4 --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 --theta-init gamma \
 --gamma-phase-mode standardize_tanh --freq-gain 2 --graph-mode learned --graph-top-k 32 --graph-spatial-decay .35 \
 --geodesic-steps 3 --geodesic-radius 1.5 --geodesic-contrast 2 --geodesic-temperature .5 --geodesic-cap 16 \
 --kuramoto-backend factorized --spike-spatial-grid-size 16 --k 256 --membrane-vth .06 \
 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 --branch 4 --gate-mode raw \
 --plv-source phase --plv-combine mean --spike-plv-weight 5 --loss-signal sigmoid_membrane \
 --sample-activity-diversity-weight 0 --checkpoint-dir "$OUT/checkpoints" --checkpoint-epochs 1 3 10 \
 --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 --structural-weight 0 \
 --plv-collapse-weight 1 --plv-bimodality-weight 6 --plv-balance-weight 10 --plv-target-density .867 \
 --plv-coherence-weight .5 --spike-per-component --dendritic-projection shared --verbose > "$OUT/training.log" 2>&1
test -s "$OUT/core.pt" -a -s "$OUT/checkpoints/epoch_01.pt" -a -s "$OUT/checkpoints/epoch_03.pt" -a -s "$OUT/checkpoints/epoch_10.pt"
echo complete > "$OUT/TRAINING_COMPLETED"

