#!/bin/bash
# PV2_0103. Why does seed 1 fail at 32x32, and can either known fix rescue it?
#
# PV2_0102: at 8 epochs the 32x32 model gives 0.7699 / 0.7068 / 0.2916 across seeds 0, 2,
# 1. Two of three land where we want; seed 1 collapses, and in the group-count arm it
# collapses differently (1.54 groups per image against 6.49 true objects). Three-seed mean
# 0.5894 +- 0.2599. spike_plv_collapse does not separate the failures.
#
# Two hypotheses, both already in this project's record, tested on the ONE seed that
# fails rather than on all three. That is six hours instead of eighteen, and if either
# rescues seed 1 the three-seed run follows.
#
#   A. VARIANCE. PV2_0012 recorded "half of a six-seed draw fails outright" at grid 16,
#      and PV2_0044's graph freeze was the answer -- which PV2_0081 characterised exactly:
#      it "reduced seed variance; it did not improve a seed". It has never been applied at
#      1024 nodes. Here seed 1 initialises from seed 0's working 8-epoch checkpoint and
#      trains with graph_generator frozen, so all seeds would share one graph, which is
#      what the published three seeds do (and is disclosed as a limitation in STATUS).
#
#   B. UNDERTRAINING. 8 epochs was inherited from PV2_0093's screening budget, never
#      selected. Seed 1's control ends at collapse 3.70 against seed 0's 2.58, so it may
#      simply not have converged. 16 epochs, no freeze, nothing else changed.
#
# If A works and B does not, the fragility is variance and the fix is the freeze. If B
# works and A does not, 8 epochs is the wrong budget and the peak must be found per seed.
# If both fail, the 8-epoch point is unusable and 32x32 offers the 40-epoch 0.6293.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0103_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="--num-feature-maps 8 --num-regions 1024 --device cuda --batch-size 16 --lr 1e-3
  --osc-dim 4 --seed 1 --gamma-drive-mode static --num-time-steps 64
  --plv-settle 32 --theta-init gamma --gamma-phase-mode standardize_tanh --freq-gain 2.0
  --graph-mode learned --graph-top-k 32 --spike-spatial-grid-size 32 --k 1024
  --membrane-vth 0.06 --membrane-low-m -4 --membrane-high-m 0 --low-n -4 --high-n 0
  --membrane-threshold-mode population --membrane-threshold-k 0.5
  --gate-mode raw --plv-source phase --plv-combine mean --loss-signal sigmoid_membrane
  --spike-rate-weight 0 --spike-smooth-weight 0 --spike-diversity-weight 0
  --structural-weight 0 --verbose --spike-per-component --geodesic-steps 3
  --geodesic-contrast 2.0 --graph-spatial-decay 0.35 --spike-plv-weight 5.0
  --plv-collapse-weight 1.0 --plv-bimodality-weight 6.0 --plv-balance-weight 10.0
  --plv-coherence-weight 0.5 --plv-target-density 0.867 --geodesic-chunk 32
  --train-limit 1000 --gamma-seq-path $G32"

ev () {  # label checkpoint
  local lab=$1 ck=$2
  [ -s "$ck" ] || { say "$lab MISSING"; return; }
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-30s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0103: rescuing seed 1 at 32x32 ==="
say "seed 1 as it stands: .2916 .0947 .2326 (groups 11.44, pred_fg .3101)"
say "seeds 0 and 2:       .7699 / .7068   |  16x16 published .7239 .7251 .4951"

cd /export_home/tkim1/snn_kuramoto_bidirectional

say ""
say "--- A: graph frozen from seed 0's working checkpoint, 8 epochs ---"
mkdir -p $R/RSC_freeze
t0=$(date +%s)
$PY training/train_s2net_core.py $BASE --epochs 8 \
   --init-from $R/GC_w0/core.pt --freeze-components graph_generator \
   --save-path $R/RSC_freeze/core.pt > $R/RSC_freeze.log 2>&1
say "    $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/RSC_freeze.log | tail -1 | cut -d= -f2)"
grep -m1 "initialised from" $R/RSC_freeze.log | sed 's/^/    /' | tee -a $OUT

say ""
say "--- B: 16 epochs, no freeze ---"
mkdir -p $R/RSC_e16
t0=$(date +%s)
$PY training/train_s2net_core.py $BASE --epochs 16 \
   --save-path $R/RSC_e16/core.pt > $R/RSC_e16.log 2>&1
say "    $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/RSC_e16.log | tail -1 | cut -d= -f2)"

say ""
say "label                          fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "seed 1 as it stands (8ep)" $R/S32_w0_s1/core.pt
ev "A: graph frozen, 8ep" $R/RSC_freeze/core.pt
ev "B: 16 epochs" $R/RSC_e16/core.pt
say "# for scale"
ev "seed 0 (works)" $R/GC_w0/core.pt
ev "seed 2 (works)" $R/S32_w0_s2/core.pt

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0103_results.txt"):
    m = re.match(r"(.+?)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)\s*$", line)
    if m and not m.group(1).startswith("["):
        rows[m.group(1).strip()] = tuple(float(m.group(i)) for i in (2,3,4,5,6))
base = rows.get("seed 1 as it stands (8ep)")
G16 = (0.7239, 0.7251, 0.4951)
if not base:
    print("seed 1 baseline missing; nothing to compare.")
else:
    print("seed 1 as it stands: %.4f / %.4f / %.4f  groups %.2f  pred_fg %.4f" % base)
    print("16x16 published:     %.4f / %.4f / %.4f" % G16)
    print()
    rescued = []
    for k in ("A: graph frozen, 8ep", "B: 16 epochs"):
        v = rows.get(k)
        if not v: continue
        print("  %-24s %.4f / %.4f / %.4f  groups %5.2f  pred_fg %.4f  (fg_ari %+.4f)"
              % (k, v[0], v[1], v[2], v[3], v[4], v[0]-base[0]))
        if v[0] > 0.60 and v[4] <= 0.20:
            rescued.append((k, v))
    print()
    if rescued:
        k, v = max(rescued, key=lambda r: r[1][0])
        print("SEED 1 IS RESCUED BY: %s" % k)
        print("  %.4f / %.4f / %.4f at predicted foreground %.4f, from %.4f."
              % (v[0], v[1], v[2], v[4], base[0]))
        print("Run all three seeds with this setting; if the spread closes, the three-seed")
        print("mean lands near where seeds 0 and 2 already are.")
    else:
        print("Neither fix rescues seed 1. The 8-epoch point is not usable at 32x32, and")
        print("what the finer grid actually offers is the 40-epoch model's 0.6293 -- below")
        print("the published 16x16. The overtraining finding stands; the operating point")
        print("does not.")
EOF
say "=== PV2_0103 DONE ==="
