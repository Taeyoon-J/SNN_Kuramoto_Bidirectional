#!/bin/bash
# PV2_0112. A label-free coverage stage: unclaimed active regions become their OWN groups.
#
# Shape fixed by REVIEW_0007 (SW_0207 against SW_0234): never widen an existing group.
# Runs AFTER the merge stage. Gate: all three up, obj_iou specifically up, pred_fg <= 0.20.
#
# PV2_0104's three seeds are GC_w0, RSC_freeze and FRZ_s2 -- read out of its commands.sh.
# The first launch guessed FRZ_s0/FRZ_s1 and ten of fifteen evaluations died on
# FileNotFoundError; only seed 2 produced data. Names are read, not guessed.
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
say "GATE registered before the run: all three up, obj_iou specifically up, pred_fg <= 0.20"
say "FALSIFIER: fg_iou up while fg_ari or obj_iou falls = SW_0207 reproduced, stage closes"
say "regression row passed this session: 0.7239 0.7251 0.4951 (exact)"

ev () {  # label checkpoint extra...
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
  else printf "%-30s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say ""
say "label                            fg_ari   fg_iou  obj_iou   groups  pred_fg"
for pair in "0:$R/GC_w0/core.pt" "1:$R/RSC_freeze/core.pt" "2:$R/FRZ_s2/core.pt"; do
  sd=${pair%%:*}; ck=${pair#*:}
  ev "seed $sd, coverage off" "$ck"
  for q in 0.95 0.90 0.85 0.80; do
    ev "seed $sd, q $q" "$ck" --coverage-quantile $q --coverage-min-size 2
  done
done
say "=== PV2_0112 DONE ==="
