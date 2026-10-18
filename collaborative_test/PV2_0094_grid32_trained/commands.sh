#!/bin/bash
# PV2_0094. The 32x32 model, trained with the fix PV2_0093 identified.
#
# PV2_0089 trained 40 epochs at 1024 nodes and the spike path died at epoch 17
# (spike_plv_collapse 9.210340 = log(10^4)). PV2_0093 separated the two candidates on
# 8-epoch arms, one variable each, and the answer is unambiguous:
#
#   control, as PV2_0089      4.26 -> 5.69   DYING
#   A: population threshold   4.72 -> 2.58   ALIVE, monotonically falling
#   B: spatial decay 0.175    4.32 -> 8.94   DEAD, worse than the control
#
# So the transduction, not the graph, is what did not transfer: --membrane-vth 0.06 is
# an ABSOLUTE threshold tuned for 256 oscillators, and at 1024 the membrane distribution
# sits below it. A population threshold places it relative to the batch's own statistics
# and the path stays alive. Hypothesis B is refuted -- the grid-unit spatial decay was
# not the problem and halving it makes things worse.
#
# This is the same 40-epoch budget as PV2_0089, one variable changed.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
FIG=/work/USERS/tkim1/figures
OUT=$R/pv2_0094_results.txt
mkdir -p $FIG; : > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0094: 32x32 with the population threshold, 40 epochs, seed 0 ==="

# regression row: contract v1 must be untouched
E="0.7239 0.7251 0.4951"
got=$(cd $CT && $PY evaluate_model.py --checkpoint $R/BIM6_s1/core.pt \
  --gamma-seq-path /work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt \
  --targets /work/USERS/tkim1/clevr/with_masks/targets_v1.pt --manifest $MAN \
  --split validation --limit 300 --spike-per-component --geodesic-steps 3 \
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --num-time-steps 1024 \
  --settle 512 --spatial-sigma 1.0 --synchrony-threshold 0.05 --geodesic-chunk 64 2>&1 \
  | grep -E "^fg_ari|^foreground_iou|^matched_object_iou" | awk '{printf "%.4f ", $2}')
say "regression (contract v1, expect $E): $got"
[ "$(echo $got)" = "$(echo $E)" ] || { say "REGRESSION FAILED -- stopping."; exit 1; }

cd /export_home/tkim1/snn_kuramoto_bidirectional
t0=$(date +%s)
$PY training/train_s2net_core.py \
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
  --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 32 \
  --train-limit 1000 --gamma-seq-path $G32 \
  --save-path $R/G32_POP40/core.pt > $R/G32_POP40.log 2>&1
say "--- trained in $(( ($(date +%s)-t0)/60 )) min ---"
say "    collapse: $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/G32_POP40.log | awk -F= '{printf "%.2f ", $2}' | awk '{print $1, "...", $(NF-2), $(NF-1), $NF}')"
[ -s $R/G32_POP40/core.pt ] || { say "TRAINING FAILED"; tail -5 $R/G32_POP40.log >> $OUT; exit 1; }

ev () {  # label extra...
  local lab=$1; shift
  local out r sp
  out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/G32_POP40/core.pt \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  sp=$(echo "$out" | grep -E "spike_rate|plv_mean" | awk '{printf "%s=%s ", $1, $2}')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-30s %s | %s\n" "$lab" "$r" "$sp" | tee -a $OUT; fi
}

say ""
say "label                            fg_ari   fg_iou  obj_iou  pred_fg | spikes"
ev "components" --spatial-sigma 1.0 --synchrony-threshold 0.05 --save-predictions $FIG/g32_cc.npz
for k in 10 14 20; do
  ev "spectral k=$k" --spectral-k $k --spatial-sigma 1.0 --save-predictions $FIG/g32_sp$k.npz
done

say ""
say "=== figures ==="
cd $CT
for tag in cc sp14; do
  [ -s $FIG/g32_$tag.npz ] || continue
  $PY visualize_patch_masks.py --predictions $FIG/g32_$tag.npz --targets $T32 \
    --image-dir /work/USERS/tkim1/clevr/with_masks/images10k --manifest $MAN \
    --split validation --rows 5 --pick spread \
    --title "ours, contract v2, 32x32, $tag" --out $FIG/g32_$tag.png 2>&1 | tail -2 | tee -a $OUT
done
say "=== PV2_0094 DONE ==="
