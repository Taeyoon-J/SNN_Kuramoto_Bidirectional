#!/bin/bash
# PV2_0110 recovery. BD_gfb trained 542 min and its scoring died on a strict
# load_state_dict: the checkpoint carries graph_generator.feedback_strength and the
# eval built the model without that flag. Passing the flag at eval time is the fix;
# no retraining is needed.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G16=/work/USERS/tkim1/gamma_sequences/wm_patch_gamma_seq_k8_grid16.pt
T16=/work/USERS/tkim1/clevr/with_masks/targets_v1.pt
OUT=$R/pv2_0110b_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/tmp/tkim1_triton_3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "=== PV2_0110b: re-score BD_gfb with the flag its checkpoint needs ==="
say "control (BD_ctrl, already scored): .5027 .5112 .3903  groups 5.42  plv_mean .7245"
say "NOTE: that control is PV2_0092's failed control to four decimals, so this base"
say "      does NOT reproduce the published .7239/.7251/.4951. Only arm-vs-control reads."
say "mechanism check: plv_mean must MOVE if feedback does what it is documented to do."

ev () {  # label ckpt extra...
  local lab=$1 ck=$2; shift 2
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G16 --targets $T16 --manifest $MAN \
    --grid 16 --num-regions 256 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --num-time-steps 1024 --settle 512 \
    --spatial-sigma 1.0 --synchrony-threshold 0.05 --geodesic-chunk 64 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|plv_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -6 >> $OUT
  else printf "%-30s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say ""
say "label                            fg_ari   fg_iou  obj_iou   groups  plv_mean pred_fg"
ev "control (both off)" $R/BD_ctrl/core.pt
ev "graph feedback 0.5" $R/BD_gfb/core.pt --graph-feedback-strength 0.5
say "=== PV2_0110b DONE ==="
