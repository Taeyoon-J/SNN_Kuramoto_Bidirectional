#!/bin/bash
# PV2_0121 evaluation, rerun. The training finished; only the scoring failed, because the
# benchmark targets declare contract_version 'v2-pixel' and evaluate_model defaults to 1.
# Same omission as the grid-64 targets earlier in this session -- the guard caught both
# rather than letting a mislabelled number through.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
M=/work/USERS/tkim1/mod
OUT=$R/pv2_0121_eval.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0121 evaluation: Tier 1, grid 32, seed 0 ==="
say "published synchrony family on CLEVR, pixel ARI-FG: CAE .04 CAE++ .27 CtCAE .54 SynCx .59 RF .65"
say "this is the PATCH contract on each dataset's own targets, not yet the pixel protocol"
say ""
say "dataset              fg_ari   fg_iou  obj_iou   groups  pred_fg"
for d in tetrominoes multi_dsprites; do
  ck=$R/T1_${d}_s0/core.pt
  [ -s "$ck" ] || { say "$d: no checkpoint yet"; continue; }
  out=$($PY $CT/evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $M/gamma_${d}_grid32.pt --targets $M/targets_${d}_grid32.pt \
    --manifest $M/split_manifest_${d}.json --expect-contract v2-pixel \
    --grid 32 --num-regions 1024 --split validation --limit 100 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 4 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$d FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-20s %s\n" "$d" "$r" | tee -a $OUT; fi
done
say "=== PV2_0121 EVAL DONE ==="
