#!/bin/bash
# PV2_0119. The sweep PV2_0091 owes.
#
# PV2_0091 concluded "the gap is in what the spikes encode, not in how they are
# thresholded" and said so from values already recorded -- different runs and
# configurations, not one checkpoint swept in one session. Its own Limitations section
# says the clean sweep "is still owed". It was launched once, produced the components
# row (0.7239 / 0.7251 / 0.4951 at 0.1290, reproducing exactly), and was stopped at 25
# minutes per row under contention with the 32x32 training.
#
# This runs it as specified there: one checkpoint, spectral k from 6 to 32,
# --limit 150 --num-time-steps 512 so it is cheap.
#
# It settles an argued-but-unmeasured claim that ROUTE_MAP_0.9_v3 leans on. If the dial's
# own maximum on ONE checkpoint reaches past aligned Slot's 0.8102, the readout is not as
# closed as the collected values suggested; if it tops out near the recorded 0.7718 at
# k=20, PV2_0091's conclusion is measured rather than argued.
#
# The calibration gate matters here more than usual: PV2_0060 recorded k=20 at predicted
# foreground 0.808 against a true 0.115, so a high fg_ari at large k is bought by calling
# most of the image foreground -- which is exactly what PV2_0090 found Slot Attention
# doing. Every row carries pred_fg and the two IoUs beside it.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G16=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid16.pt
T16=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
OUT=$R/pv2_0119_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0119: the clean spectral-k sweep PV2_0091 owes ==="
say "one checkpoint BIM6_s1, limit 150, 512 steps, as PV2_0091's Limitations specified"
say "recorded values it should reproduce: components .7300, k=14 .7603, k=20 .7718"
say "aligned Slot Attention is .8102 at pred_fg .8531; a perfect segmenter .9756-1.0"
say "READ pred_fg ON EVERY ROW: fg_ari rises for free when foreground is over-predicted"

ev () {  # label extra...
  local lab=$1; shift
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/BIM6_s1/core.pt \
    --gamma-seq-path $G16 --targets $T16 --manifest $MAN \
    --split validation --limit 150 --spike-per-component --geodesic-steps 3 \
    --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --num-time-steps 512 \
    --settle 256 --spatial-sigma 1.0 --synchrony-threshold 0.05 \
    --geodesic-chunk 64 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -5 >> $OUT
  else printf "%-26s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say ""
say "label                       fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "components (no k)"
for k in 6 8 10 12 14 16 20 24 28 32; do
  ev "spectral k=$k" --spectral-k $k
done
say "=== PV2_0119 DONE ==="
