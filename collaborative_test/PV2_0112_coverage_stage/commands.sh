#!/bin/bash
# PV2_0112. A label-free coverage stage: unclaimed active regions become their OWN groups.
#
# Shape fixed by REVIEW_0007 (SW_0207 against SW_0234): never widen a group that already
# exists. Runs AFTER the merge stage so recovered groups cannot be absorbed by the merger.
# Pre-registered gate: all three metrics up, matched-object IoU specifically up, pred_fg <= 0.20.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0112_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0112: label-free coverage stage ==="
say "control (PV2_0104, 3 seeds): .7110 / .7130 / .6588  groups 16.50  pred_fg .1369"
say "gate: all three up, obj_iou specifically up, pred_fg <= 0.20"

# regression row first -- contract v1 must be untouched
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

ev () {  # label seed extra...
  local lab=$1 sd=$2; shift 2
  local r=$(cd $CT && $PY evaluate_model.py --checkpoint $R/FRZ_s$sd/core.pt \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 "$@" 2>&1 \
    | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|^groups|^predicted_foreground" \
    | awk '{printf "%.4f ", $2}')
  printf "%-28s %s\n" "$lab" "$r" | tee -a $OUT
}

say ""
say "label                          fg_ari   fg_iou  obj_iou   groups  pred_fg"
for sd in 0 1 2; do
  ev "seed $sd, coverage off" $sd
  for q in 0.95 0.90 0.85 0.80; do
    ev "seed $sd, q $q" $sd --coverage-quantile $q --coverage-min-size 2
  done
done
say "=== PV2_0112 DONE ==="
