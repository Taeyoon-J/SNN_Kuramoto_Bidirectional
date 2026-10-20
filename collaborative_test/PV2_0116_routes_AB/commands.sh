#!/bin/bash
# ROUTE_MAP_0.9_v3 routes A and B. Evaluation only, no training.
#
# A: --target-foreground has never appeared in a command in this repository. It
#    calibrates the synchrony threshold per image to a FIXED fraction, selected on
#    validation like any hyperparameter -- not an oracle. It is the only dial that acts
#    on the 0.2006 half (PV2_0114: removing false positives is worth +0.2006 of
#    foreground IoU, recovering every miss +0.0428). Its sign is unknown: unselective
#    tightening makes fg-IoU worse, 0.7450 -> 0.6593 at the measured confusion, while
#    perfect selectivity reaches 0.9456.
#
# B: PV2_0098's DS_6000 trained on six times the images -- loss 14.429044 against
#    16.673826, collapse 1.954825 against 2.154499 -- and has never been scored.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G16=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
T16=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
OUT=$R/pv2_0116_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0116: routes A and B, evaluation only ==="
say "published reference (contract v1, grid 16): .7239 .7251 .4951 at pred_fg .1290"
say "GATE: all three above the no-dial baseline at pred_fg <= 0.20, obj_iou up"

ev () {  # label ckpt extra...
  local lab=$1 ck=$2; shift 2
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G16 --targets $T16 --manifest $MAN \
    --split validation --limit 300 --spike-per-component --geodesic-steps 3 \
    --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --num-time-steps 1024 \
    --settle 512 --spatial-sigma 1.0 --synchrony-threshold 0.05 \
    --geodesic-chunk 64 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-34s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say ""
say "ROUTE A -- target-foreground on the published checkpoint BIM6_s1"
say "label                               fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "A: no dial (baseline)" $R/BIM6_s1/core.pt
for t in 0.16 0.14 0.121 0.11 0.10; do
  ev "A: target-foreground $t" $R/BIM6_s1/core.pt --target-foreground $t
done

say ""
say "ROUTE B -- the data-scale checkpoints PV2_0098 trained and never scored"
say "label                               fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "B: DS_1000 (1000 images)" $R/DS_1000/core.pt
ev "B: DS_6000 (6000 images)" $R/DS_6000/core.pt
say "=== PV2_0116 DONE ==="
