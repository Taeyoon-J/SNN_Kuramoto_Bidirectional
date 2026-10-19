#!/bin/bash
# Best configuration, 16x16 grid. FG-ARI 0.598 from the phase readout (3 seeds).
set -euo pipefail

PY=${PY:-/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python}
WORK=${WORK:-/work/USERS/tkim1}
GAMMA=${GAMMA:-$WORK/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt}
LABELS=${LABELS:-$WORK/clevr/with_masks/patch_labels.pt}
RUN=${RUN:-$WORK/runs/BEST}
GPU=${GPU:-0}
SEED=${SEED:-0}
# set SPIKE=1 to also get the spiking readout
SPIKE=${SPIKE:-0}

export PYTHONPATH=/export_home/tkim1:.
export TRITON_CACHE_DIR=/tmp/tkim1_triton_$GPU
export CUDA_VISIBLE_DEVICES=$GPU

TRAIN_CFG=(
  --gamma-seq-path "$GAMMA"
  --num-regions 256
  --num-feature-maps 8
  --device cuda
  --epochs 40
  --batch-size 16
  --lr 1e-3
  --seed "$SEED"
  --osc-dim 4

  --gamma-drive-mode static
  --num-time-steps 64
  --plv-settle 32
  --theta-init gamma
  --gamma-phase-mode standardize_tanh
  --freq-gain 2.0

  --graph-mode learned
  --graph-top-k 32
  --graph-spatial-decay 0.55
  --spike-spatial-grid-size 16
  --k 256

  --membrane-vth 0.06
  --membrane-low-m -4
  --membrane-high-m 0
  --low-n -4
  --high-n 0
  --gate-mode raw

  --plv-source phase
  --plv-combine mean
  --loss-signal sigmoid_membrane
  --plv-collapse-weight 1.0
  --plv-bimodality-weight 1.0
  --plv-balance-weight 10.0
  --plv-target-density 0.867
  --plv-coherence-weight 0.5

  --spike-rate-weight 0
  --spike-smooth-weight 0
  --spike-diversity-weight 0
  --structural-weight 0

  --verbose
)

EVAL_CFG=(
  --graph-top-k 32
  --gate-mode raw
  --osc-dim 4
  --gamma-seq-path "$GAMMA"
  --num-regions 256
  --grid 16
  --patch-labels "$LABELS"
  --num-images 150
  --skip 200
  --num-time-steps 256
  --settle 64
  --fixed-k 8
)

if [ "$SPIKE" = "1" ]; then
  TRAIN_CFG+=( --spike-per-component )
  EVAL_CFG+=( --spike-per-component )
fi

mkdir -p "$RUN"
$PY training/train_s2net_core.py "${TRAIN_CFG[@]}" --save-path "$RUN/core.pt"
$PY training/evaluate_binding.py --checkpoint "$RUN/core.pt" "${EVAL_CFG[@]}"
