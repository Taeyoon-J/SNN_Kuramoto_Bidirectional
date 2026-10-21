#!/bin/bash
# Tier 1 of BENCHMARK_PLAN.md: the same architecture on Tetrominoes and Multi-dSprites.
#
# Grid 32, because the plan decided the paper table goes pixel-protocol at 32x32 or finer
# and these datasets are where that costs us almost nothing: a grid-32 cell is 1.09 px on
# Tetrominoes and 2.00 px on Multi-dSprites against 7.50 x 10.00 on CLEVR, so the
# quantisation handicap that caps CLEVR's pixel-protocol score at 0.8355 nearly vanishes.
#
# Everything these need was built and verified earlier: patch targets (purity 0.9941 and
# 0.9878), per-dataset input encoders trained by unsupervised reconstruction (224
# parameters each, TRAIN split only), and gamma sequences at unit variance.
#
# ONE SEED PER DATASET FIRST. These are new data for this architecture and the first thing
# to learn is whether the spike path survives at all -- PV2_0094 showed an absolute
# membrane threshold does not transfer across oscillator counts, and nothing says it
# transfers across datasets either. --membrane-threshold-mode population is carried for
# the same reason.
#
# No gate is registered here because there is no control to beat yet: this establishes
# that the model trains on these datasets and produces a number. The comparison to CAE,
# Rotating Features, SynCx and AKOrN comes after, under the pixel protocol.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
M=/work/USERS/tkim1/mod
OUT=$R/pv2_0121_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0121: Tier 1 -- Tetrominoes and Multi-dSprites, grid 32, seed 0 ==="
say "cell size at grid 32: Tetrominoes 1.09px, Multi-dSprites 2.00px, CLEVR 7.50x10.00"
say "patch purity: Tetrominoes .9941  Multi-dSprites .9878  CLEVR .9170"
say "published synchrony family on CLEVR (pixel ARI-FG): CAE .04 CAE++ .27 CtCAE .54 SynCx .59 RF .65"
say "no gate: this establishes the model trains on these datasets at all"

run () {  # dataset
  local d=$1
  local nm=T1_${d}_s0
  local G=$M/gamma_${d}_grid32.pt
  local T=$M/targets_${d}_grid32.pt
  local MAN=$M/split_manifest_${d}.json
  [ -s $R/$nm/core.pt ] && { say "$nm already trained, kept"; } || {
    local t0=$(date +%s); mkdir -p $R/$nm
    (cd /export_home/tkim1/snn_kuramoto_bidirectional && $PY training/train_s2net_core.py \
      --num-feature-maps 8 --num-regions 1024 --device cuda --batch-size 16 --lr 1e-3 \
      --osc-dim 4 --epochs 40 --seed 0 --gamma-drive-mode static --num-time-steps 64 \
      --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
      --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 32 --k 1024 \
      --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
      --membrane-threshold-mode population --membrane-threshold-k 0.5 \
      --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
      --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
      --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3 \
      --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0 \
      --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0 \
      --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 16 \
      --train-limit 1000 --gamma-seq-path $G \
      --save-path $R/$nm/core.pt) > $R/$nm.log 2>&1
    say "$nm $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/$nm.log | tail -1 | cut -d= -f2)"
    [ -s $R/$nm/core.pt ] || { say "$nm TRAINING FAILED"; tail -6 $R/$nm.log >> $OUT; return; }
  }
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/$nm/core.pt \
    --gamma-seq-path $G --targets $T --manifest $MAN \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$d EVALUATION FAILED"; echo "$out" | tail -6 >> $OUT
  else printf "%-20s %s\n" "$d" "$r" | tee -a $OUT; fi
}

say ""
say "dataset              fg_ari   fg_iou  obj_iou   groups  pred_fg"
run tetrominoes
run multi_dsprites
say "=== PV2_0121 DONE ==="
