#!/bin/bash
# PV2_0089. The published configuration at a 32x32 grid (contract v2).
#
# FIRST ATTEMPT WITHDRAWN. train_s2net_core.py trains on the WHOLE tensor it is given
# unless --train-limit says otherwise, so pointing it at the 10,000-row grid-32 gamma
# trained on validation AND test. Contract v1's own training used
# wm_patch_gamma_seq_k8_grid16.pt, which is 1,000 rows -- verified here to be wm10k
# rows 0-999, per-image cosine 1.000000 -- all inside the train range [0, 6000).
# So the faithful port is --train-limit 1000, which is also 6x cheaper.
#
# ARM A is the direct port: only the grid changes. ARM B additionally halves the
# spatial decay and adds a relaxation round, because both are measured in GRID units
# and a grid-32 step covers half the image a grid-16 step does.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
MAN=$CT/data/split_manifest.json
FIG=/work/USERS/tkim1/figures
OUT=$R/pv2_0089_results.txt
mkdir -p $FIG
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }

GPU=2
export CUDA_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/tkim1_triton_$GPU
say "=== PV2_0089: the published configuration at 32x32, contract v2. GPU $GPU ==="

# -------------------------------------------------------------- regression row
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

COMMON="--num-feature-maps 8 --num-regions 1024 --device cuda --batch-size 16 --lr 1e-3
  --osc-dim 4 --epochs 40 --seed 0 --gamma-drive-mode static --num-time-steps 64
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0
  --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 32 --k 1024
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0
  --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0
  --structural-weight 0 --verbose --spike-per-component --geodesic-contrast 2.0
  --spike-plv-weight 5.0 --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0
  --plv-balance-weight 10.0 --plv-coherence-weight 0.5 --plv-target-density 0.867
  --geodesic-chunk 32 --train-limit 1000 --gamma-seq-path $G32"

ev () {  # label checkpoint steps decay [extra...]
  local lab=$1 ck=$2 st=$3 dc=$4; shift 4
  [ -s "$ck" ] || { say "$lab MISSING"; return; }
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps $st --geodesic-contrast 2.0 \
    --graph-spatial-decay $dc --geodesic-chunk 64 \
    --num-time-steps 1024 --settle 512 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|predicted_foreground" \
      | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -6 >> $OUT
  else printf "%-42s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

cd /export_home/tkim1/snn_kuramoto_bidirectional
arm () {  # name steps decay
  local name=$1 st=$2 dc=$3
  mkdir -p $R/$name
  local t0=$(date +%s)
  $PY training/train_s2net_core.py $COMMON --geodesic-steps $st \
    --graph-spatial-decay $dc --save-path $R/$name/core.pt > $R/$name.log 2>&1
  say "--- $name  $(( ($(date +%s)-t0)/60 )) min  $(grep -oE 'loss: [0-9.]+ -> [0-9.]+' $R/$name.log | tail -1) ---"
  grep -m1 "training on the first" $R/$name.log | sed 's/^/    /' | tee -a $OUT
  tail -1 $R/$name.log | cut -c1-140 | sed 's/^/    last  /' | tee -a $OUT
  [ -s $R/$name/core.pt ] || { say "    TRAINING FAILED"; tail -5 $R/$name.log >> $OUT; return; }

  say "label                                        fg_ari   fg_iou  obj_iou  pred_fg"
  ev "$name components" $R/$name/core.pt $st $dc --spatial-sigma 1.0 --synchrony-threshold 0.05 \
     --save-predictions $FIG/pred_${name}_cc.npz
  for k in 10 14 18; do
    ev "$name spectral k=$k" $R/$name/core.pt $st $dc --spectral-k $k --spatial-sigma 1.0 \
       --save-predictions $FIG/pred_${name}_sp$k.npz
  done
}

arm G32_A 3 0.35
arm G32_B 4 0.175
say "=== PV2_0089 DONE ==="
