#!/bin/bash
# PV2_0122. Route C, retried with the peer's warm start instead of training from scratch.
#
# PV2_0120 trained grid 64 from scratch and did not finish one epoch in 8.5 hours, which
# projects to about 14 days for 40. SW_0192 reports native64 at roughly 2 hours a seed,
# and SW_0154 says why: "A row-major native16 node maps to its exact 4x4 block of native64
# children. Node-leading parameters repeat by that mapping; pairwise parameters use it on
# both axes." They never train the fine grid from scratch. REVIEW_0006 recorded this as a
# technique worth taking; route C was designed without it, which was my omission.
#
# Step 1, which this runs: evaluate the replicated parameters with NO fine-tuning at all.
# That is nearly free and it answers the question the whole route depends on -- does a
# grid-32 model, replicated onto grid 64, still segment? If it collapses, the warm start
# is not a warm start and fine-tuning from it would be fitting noise.
#
# grid_distance is deliberately not replicated: it is geometric and the model rebuilds it
# for its own grid.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G64=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid64.pt
T64=/work/USERS/tkim1/clevr/with_masks/targets_v3_grid64.pt
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0122_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0122: grid 64 by parameter replication, no fine-tuning yet ==="
say "parent GC_w0 at grid 32 (PV2_0104 seed 0): .7699 / .7531 / .7017 at pred_fg .1512"
say "technique ported from SW_0154; PV2_0120 from scratch was ~14 days and is abandoned"
say "QUESTION: does the replicated model still segment at all, before any fine-tuning?"
say "30 images, geodesic-chunk 1 at grid 64: evaluate_model has no batch flag, so --limit"
say "IS the batch, and _relax allocates batch x chunk x N x N -- 12.5 GiB at limit 100."
say "Both rows use the same 30 images so the comparison is like for like."

say ""
say "label                        fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev () {  # label ckpt grid regions gamma targets sigma minsize chunk contract
  local lab=$1 ck=$2 g=$3 nr=$4 gs=$5 tg=$6 sg=$7 ms=$8 ch=$9 ec=${10} lim=${11}
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $gs --targets $tg --manifest $MAN --expect-contract $ec \
    --grid $g --num-regions $nr --split validation --limit $lim \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk $ch \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma $sg --min-group-size $ms 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -6 >> $OUT
  else printf "%-28s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

ev "parent, grid 32"  $R/GC_w0/core.pt       32 1024 $G32 $T32 4.0 2 16 2 30
ev "replicated, grid 64"  $R/G64_warm_s0.pt  64 4096 $G64 $T64 8.0 4 1 3 30
say "=== PV2_0122 STEP 1 DONE ==="
