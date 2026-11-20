#!/bin/bash
# PV2_0117. Route A on the grid where the over-prediction actually is.
#
# PV2_0116 ran --target-foreground on BIM6_s1 at grid 16, whose predicted foreground is
# already 0.1290 against a true 0.1152 -- almost nothing to tighten. Every setting came
# out worse on all three metrics, which tests the dial against a checkpoint that does not
# have the fault the dial is for.
#
# PV2_0114 measured the over-prediction at GRID 32 on GC_w0: predicted foreground 0.1400
# against a true 0.1152, with 22.2% of predicted patches false and removing them worth
# +0.2006 of foreground IoU. This runs the dial there, on all three of PV2_0104's seeds.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0117_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0117: target-foreground at grid 32, where the over-prediction is ==="
say "control (PV2_0104, 3 seeds): .7110 / .7130 / .6588 at pred_fg .1369"
say "PV2_0114 at grid 32: 22.2% of predicted patches false; removing them worth +0.2006 fg-IoU"
say "GATE: all three above the no-dial control, obj_iou specifically up, pred_fg <= 0.20"

ev () {  # label ckpt extra...
  local lab=$1 ck=$2; shift 2
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-32s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say ""
say "label                             fg_ari   fg_iou  obj_iou   groups  pred_fg"
for pair in "0:$R/GC_w0/core.pt" "1:$R/RSC_freeze/core.pt" "2:$R/FRZ_s2/core.pt"; do
  sd=${pair%%:*}; ck=${pair#*:}
  ev "seed $sd, no dial" "$ck"
  for t in 0.13 0.1152 0.10; do
    ev "seed $sd, target-fg $t" "$ck" --target-foreground $t
  done
done
say "=== PV2_0117 DONE ==="
