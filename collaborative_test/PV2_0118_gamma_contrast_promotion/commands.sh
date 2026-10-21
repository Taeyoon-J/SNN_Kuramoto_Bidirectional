#!/bin/bash
# PV2_0118. Promotion test for the gamma-contrast loss: 40 epochs, three seeds.
#
# PV2_0115 screened it at 8 epochs, seed 0, and weight 5.0 passed the registered
# mechanism check -- activity-vs-foreground AUC 0.503 -> 0.585, with the sign of the
# separation flipped (in the control the foreground is LESS active than the background).
# It also moved all three metrics above control and took predicted foreground DOWN,
# 0.3699 -> 0.3122. That direction is what PV2_0114's arithmetic asked for: removing
# false foreground is worth +0.2006 of foreground IoU against +0.0428 for recovering
# every miss, and every other stage either branch built pushed foreground up.
#
# Weight 1.0 failed the mechanism check (AUC 0.485, below control) so it is dropped.
#
# Every readout-side route is now closed by measurement: adding unclaimed patches
# (PV2_0112, base rate 0.73%), widening groups (peer SW_0207), merging (PV2_0107b and
# SW_0234, which loses obj_iou even with a GT oracle picking pairs), deleting spurious
# groups (no signal, AUC 0.500), and --target-foreground (PV2_0117, shatters the
# partition). This loss is the only intervention left that moved the right way.
#
# GATE, registered before the run: all three patch metrics above the control at three
# seeds, matched_object_iou specifically up, predicted foreground at or below 0.20 --
# which at 40 epochs is a real constraint, unlike the 8-epoch screen where the control
# itself sat at 0.3699.
#
# FALSIFIER: if the three-seed means do not clear the control, the loss joins the closed
# list and the honest position is that 0.9 is out of reach on this contract without a
# change larger than a loss.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G16=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
T16=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
OUT=$R/pv2_0118_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0118: gamma-contrast at 40 epochs, three seeds ==="
say "published reference (contract v1, grid 16): .7239 .7251 .4951 at pred_fg .1290"
say "PV2_0115 screen, 8 epochs seed 0: ctrl .6945/.3652/.4666 pf .3699  w5 .7015/.4544/.4851 pf .3122"
say "PV2_0115 mechanism: AUC ctrl 0.503 -> w5 0.585 (w1 0.485, FAILED, dropped)"
say "GATE: all three above control at 3 seeds, obj_iou up, pred_fg <= 0.20"
say "FALSIFIER: if the means do not clear control, the loss joins the closed list"

E="0.7239 0.7251 0.4951"
got=$(cd $CT && $PY evaluate_model.py --checkpoint $R/BIM6_s1/core.pt \
  --gamma-seq-path $G16 --targets $T16 --manifest $MAN \
  --split validation --limit 300 --spike-per-component --geodesic-steps 3 \
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --num-time-steps 1024 \
  --settle 512 --spatial-sigma 1.0 --synchrony-threshold 0.05 --geodesic-chunk 64 2>&1 \
  | grep -E "^fg_ari|^foreground_iou|^matched_object_iou" | awk '{printf "%.4f ", $2}')
say "regression (expect $E): $got"
[ "$(echo $got)" = "$(echo $E)" ] || { say "REGRESSION FAILED -- stopping."; exit 1; }

BASE="--num-feature-maps 8 --num-regions 256 --device cuda --batch-size 16 --lr 1e-3 \
 --osc-dim 4 --epochs 40 --gamma-drive-mode static --num-time-steps 64 \
 --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
 --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 16 --k 256 \
 --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
 --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
 --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
 --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3 \
 --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0 \
 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0 \
 --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 64 \
 --train-limit 1000 --gamma-seq-path $G16"

train () {  # name seed extra...
  local nm=$1 sd=$2; shift 2
  [ -s $R/$nm/core.pt ] && { say "$nm already trained, kept"; return; }
  local t0=$(date +%s); mkdir -p $R/$nm
  (cd /export_home/tkim1/snn_kuramoto_bidirectional && \
   $PY training/train_s2net_core.py $BASE --seed $sd "$@" --save-path $R/$nm/core.pt) \
   > $R/$nm.log 2>&1
  say "$nm $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/$nm.log | tail -1 | cut -d= -f2)  gc $(grep -oE 'gamma_contrast=[-0-9.]+' $R/$nm.log | tail -1 | cut -d= -f2)"
  [ -s $R/$nm/core.pt ] || { say "$nm TRAINING FAILED"; tail -5 $R/$nm.log >> $OUT; }
}

ev () {  # label ckpt
  local lab=$1 ck=$2
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G16 --targets $T16 --manifest $MAN \
    --split validation --limit 300 --spike-per-component --geodesic-steps 3 \
    --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --num-time-steps 1024 \
    --settle 512 --spatial-sigma 1.0 --synchrony-threshold 0.05 --geodesic-chunk 64 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-30s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

for sd in 0 1 2; do
  train GCP_ctrl_s$sd $sd
  train GCP_w5_s$sd   $sd --gamma-contrast-weight 5.0
done

say ""
say "label                          fg_ari   fg_iou  obj_iou   groups  pred_fg"
for sd in 0 1 2; do
  ev "seed $sd, control"       $R/GCP_ctrl_s$sd/core.pt
  ev "seed $sd, gamma-contrast 5" $R/GCP_w5_s$sd/core.pt
done
say "=== PV2_0118 DONE ==="
