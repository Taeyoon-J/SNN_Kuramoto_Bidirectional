#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
ROOT=/Data0/kevinswk/patch_v2_sw
OUT="$ROOT/trained_models/SW_0004_graph_teacher_pilot_seed0"
mkdir -p "$OUT"
source /Data0/kevinswk/miniforge3/etc/profile.d/conda.sh
conda activate /Data0/kevinswk/envs/snn
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup python -u -m snn_kuramoto_bidirectional.training.train_s2net_core \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt \
  --save-path "$OUT/core.pt" \
  --num-regions 256 --num-feature-maps 8 --device cuda \
  --epochs 10 --batch-size 16 --lr 1e-3 --seed 0 --osc-dim 4 \
  --gamma-drive-mode static --num-time-steps 64 --plv-settle 32 \
  --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --graph-spatial-decay 0.55 \
  --spike-spatial-grid-size 16 --k 256 \
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 \
  --low-n -4 --high-n 0 --gate-mode raw \
  --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
  --spike-rate-weight 0 --spike-smooth-weight 0 \
  --spike-diversity-weight 0 --structural-weight 0 \
  --plv-collapse-weight 1.0 --plv-bimodality-weight 1.0 \
  --plv-balance-weight 10.0 --plv-target-density 0.867 \
  --plv-coherence-weight 0.5 --spike-per-component \
  --graph-teacher-weight 0.1 --graph-teacher-temperature 0.1 --verbose \
  > "$OUT/training.log" 2>&1 < /dev/null &
echo "PID=$!"
echo "OUTPUT=$OUT"
