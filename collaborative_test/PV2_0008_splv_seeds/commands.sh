#!/bin/bash
# PV2_0008. Three seeds, then one test read at a threshold fixed on validation.
# Env: /work/USERS/tkim1/envs/miniforge3/envs/snn (torch + scipy), one GPU each.
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
G=/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt
GG=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
T=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
M=/work/USERS/tkim1/clevr/with_masks/split_manifest.json
D35="--geodesic-steps 3 --geodesic-contrast 2.0 --graph-spatial-decay 0.35"

BASE="--gamma-seq-path $G --num-regions 256 --num-feature-maps 8 --device cuda
      --epochs 40 --batch-size 16 --lr 1e-3 --osc-dim 4
      --gamma-drive-mode static --num-time-steps 64 --plv-settle 32
      --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0
      --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 16 --k 256
      --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0
      --low-n -4 --high-n 0 --gate-mode raw --plv-source phase --plv-combine mean
      --loss-signal sigmoid_membrane
      --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0
      --structural-weight 0
      --plv-collapse-weight 1.0 --plv-bimodality-weight 1.0 --plv-balance-weight 10.0
      --plv-target-density 0.867 --plv-coherence-weight 0.5 --verbose
      --spike-per-component --spike-plv-weight 5.0 $D35"

# 1. train, one GPU per seed
cd /export_home/tkim1/snn_kuramoto_bidirectional
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$s TRITON_CACHE_DIR=/tmp/tkim1_triton_$s \
    $PY training/train_s2net_core.py $BASE --seed $s \
    --save-path $R/SPLV_w5_s$s/core.pt > $R/SPLV_w5_s$s.log 2>&1 &
done
wait

# 2. select the threshold on the FULL validation split, from the 3-seed mean
cd /export_home/tkim1/collaborative_test
for sy in 0.05 0.10 0.15 0.20 0.25 0.35 0.50; do
  for s in 0 1 2; do
    $PY evaluate_model.py --checkpoint $R/SPLV_w5_s$s/core.pt --gamma-seq-path $GG \
      --targets $T --manifest $M --split validation --spike-per-component $D35 \
      --synchrony-threshold $sy
  done
done
# 0.20 won: best 3-seed foreground IoU (0.6730), within 0.0009 of the best fg_ari.

# 3. read test ONCE at that threshold
for s in 0 1 2; do
  $PY evaluate_model.py --checkpoint $R/SPLV_w5_s$s/core.pt --gamma-seq-path $GG \
    --targets $T --manifest $M --split test --limit 300 --spike-per-component $D35 \
    --synchrony-threshold 0.20 --json-out $R/pv8_s${s}_300.json
done
