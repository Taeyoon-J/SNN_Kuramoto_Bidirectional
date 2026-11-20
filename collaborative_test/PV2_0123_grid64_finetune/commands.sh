#!/bin/bash
# PV2_0123. Fine-tune the replicated grid-64 model. The second half of route C.
#
# PV2_0122 showed the warm start carries: with zero gradient steps, grid 64 holds the
# parent's foreground IoU (0.7633 against 0.7629) and matched-object IoU (0.7188 against
# 0.7179). What it loses is FG-ARI, 0.7588 -> 0.7066, and it loses it together with the
# group count rising 15.27 -> 19.70. Each parent node's 2x2 block starts from identical
# parameters but receives different gamma, phases diverge inside the block, and one object
# splits across groups -- which FG-ARI penalises and neither IoU does.
#
# So the target of this fine-tune is the GROUP COUNT, not the loss. The peer needs only 16
# proposals (SW_0192); this uses 4 epochs over 1000 images at batch 4, which is 1000
# steps -- far more than they use and still cheap next to the 14 days from scratch.
#
# GATE, registered before the run: foreground IoU and matched-object IoU must stay at or
# above the replicated model's 0.7633 and 0.7188, AND FG-ARI must rise above 0.7066. The
# question the whole route turns on is whether it also clears the PARENT's 0.7588 -- if a
# finer grid cannot beat the grid it was copied from, route C has no argument left.
#
# FALSIFIER: if FG-ARI does not clear 0.7066, or either IoU falls, the fine-tune is not
# repairing the split and route C closes.
#
# SIZED FROM THE PEER, after a first attempt sized from habit. That one asked for 4 epochs
# over 1000 images at batch 4 -- 1000 steps -- and spent ten hours emitting
# "expandable_segments: memory mapping failed with OOM" without finishing an epoch: batch 4
# needs 17.52 GB and GPU 2 is shared with Tier 1. SW_0192 trains native64 on TRAIN64 with
# 16 proposals, so 64 images at batch 2 is 32 steps, the same order as theirs, at the 8.90 GB
# that was measured rather than assumed.
#
# --init-from already exists in the trainer and loads through load_s2net_core, which
# raises on any key the core does not expect; it was added for --freeze-phase-path. No
# new flag was needed and none was added.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G64=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid64.pt
T64=/work/USERS/tkim1/clevr/with_masks/targets_v3_grid64.pt
OUT=$R/pv2_0123_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0123: fine-tune the replicated grid-64 model ==="
say "replicated, no fine-tuning: .7066 / .7633 / .7188  groups 19.70  pred_fg .1257"
say "parent at grid 32:          .7588 / .7629 / .7179  groups 15.27  pred_fg .1281"
say "GATE: both IoUs hold at >= .7633 / .7188 AND fg_ari rises above .7066"
say "the route's real question: does it clear the parent's .7588?"
say "FALSIFIER: fg_ari flat or an IoU falls -> the fine-tune is not repairing the split"

t0=$(date +%s); mkdir -p $R/G64_ft_s0
cp $R/G64_warm_s0.pt $R/G64_ft_s0/init.pt
(cd /export_home/tkim1/snn_kuramoto_bidirectional && $PY training/train_s2net_core.py \
  --num-feature-maps 8 --num-regions 4096 --device cuda --batch-size 2 --lr 2e-4 \
  --osc-dim 4 --epochs 1 --seed 0 --gamma-drive-mode static --num-time-steps 64 \
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0 \
  --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 64 --k 4096 \
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0 \
  --membrane-threshold-mode population --membrane-threshold-k 0.5 \
  --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane \
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0 \
  --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3 \
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0 \
  --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0 \
  --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 8 \
  --train-limit 64 --gamma-seq-path $G64 \
  --init-from $R/G64_warm_s0.pt \
  --save-path $R/G64_ft_s0/core.pt) > $R/G64_ft_s0.log 2>&1
rc=$?
say "fine-tune $(( ($(date +%s)-t0)/60 )) min  rc=$rc  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/G64_ft_s0.log | tail -1 | cut -d= -f2)"
if [ ! -s $R/G64_ft_s0/core.pt ]; then
  say "TRAINING FAILED"; tail -8 $R/G64_ft_s0.log >> $OUT; exit 1
fi

say ""
say "label                        fg_ari   fg_iou  obj_iou   groups  pred_fg"
for pair in "replicated (no FT):$R/G64_warm_s0.pt" "fine-tuned:$R/G64_ft_s0/core.pt"; do
  lab=${pair%%:*}; ck=${pair#*:}
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G64 --targets $T64 --manifest $MAN --expect-contract 3 \
    --grid 64 --num-regions 4096 --split validation --limit 30 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 1 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 8.0 --min-group-size 4 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -6 >> $OUT
  else printf "%-28s %s\n" "$lab" "$r" | tee -a $OUT; fi
done
say "=== PV2_0123 DONE ==="
