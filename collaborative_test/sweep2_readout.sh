#!/bin/bash
# Multi-dSprites readout sweep, in the direction the peer's answer points.
#
# My first sweep went the WRONG WAY. I varied synchrony-threshold 0.05 -> 0.01 and held
# min-group-size at 2, reasoning that looser thresholds admit more edges and so merge the
# fragments. The peer's SW0250 readout, answered in REQUEST_0008, runs the opposite
# regime:
#
#            synchrony_threshold   min_group_size   graph top-k
#   ours                   0.05                 2            32
#   peer                   0.50                32           512
#
# Their pipeline is strict-threshold -> few large seeds -> discard the small ones ->
# agglomerate with a signed pair energy. Ours is loose-threshold -> many fragments ->
# keep everything -> no merge stage. We produce 65.37 groups per image at grid 64 for
# 2.37 true objects; they produce 14.44 (and their seed 2 produces 29.7, so they fragment
# too, just less).
#
# This sweeps the two settings we can change without porting code. It does NOT include
# their foreground_induced_merge stage, which is a real code port and is deferred --
# PV2_0107b lost matched-object IoU with a different merge rule, so it needs its own test.
#
# If our number moves here, much of the gap is readout configuration. If it does not, the
# gap is the 29,128-parameter mask-supervised encoder or the merge stage, and the sweep
# has cost one GPU afternoon to rule the cheap half out.
#
# Grid 64, 150 validation images, same checkpoint, no training.
set -u
cd /export_home/tkim1/collaborative_test
export PYTHONPATH=/export_home/tkim1:/export_home/tkim1/snn_kuramoto_bidirectional:.
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
M=/work/USERS/tkim1/mod
R=/work/USERS/tkim1/runs
d=multi_dsprites

echo "baseline for reference: thresh .05 minsz 2 -> fg_ari .2817  groups 65.37  (300 imgs, grid 64)"
echo "peer SW0250 zero-shot:  thresh .50 minsz 32            -> .6833 TEST1000, 14.44 groups"
echo "                        (their source checkpoint is CLEVR MASK-SUPERVISED, not unsupervised)"
echo
printf "%-8s %-8s %-9s %-9s %-9s %-9s %s\n" "thresh" "minsz" "fg_ari" "fg_iou" "obj_iou" "pred_fg" "groups"
for th in 0.05 0.10 0.30 0.50; do
for ms in 2 8 32; do
  out=$($PY evaluate_model.py --checkpoint $R/T1_${d}_g64/core.pt \
    --gamma-seq-path $M/gamma_${d}_grid64.pt --targets $M/targets_${d}_grid64.pt \
    --manifest $M/split_manifest_${d}.json --expect-contract v2-pixel \
    --grid 64 --num-regions 4096 --split validation --limit 150 \
    --eval-batch 4 --affinity-chunk 25 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 2 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 \
    --synchrony-threshold $th --spatial-sigma 4.0 --min-group-size $ms \
    --save-predictions $M/sw2_${d}_t${th}_m${ms}.npz 2>&1)
  v=$(echo "$out" | grep -E '^fg_ari|^foreground_iou|^matched_object_iou|predicted_foreground|groups_per_image_mean' | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$v" ]; then
    printf "%-8s %-8s FAILED\n" "$th" "$ms"; echo "$out" | grep -iE "error|Traceback" | head -1
  else
    printf "%-8s %-8s %s\n" "$th" "$ms" "$v"
  fi
done
done
echo "rc=done"
