#!/bin/bash
# PV2_0102. Does the 8-epoch 32x32 result hold at seeds 1 and 2?
#
# PV2_0101 confirmed on the same 300 images that the 8-epoch model beats the 40-epoch
# one by +0.1406 / +0.0440 / +0.1132, and beats the published 16x16 configuration on all
# three inside the calibrated regime:
#
#   32x32, 8 epochs   0.7699 / 0.7531 / 0.7017   pred_fg 0.1512
#   16x16, published  0.7239 / 0.7251 / 0.4951   pred_fg 0.1290
#
# Nothing in this project has done that. It is also seed 0 alone, and the 8-epoch point
# was inherited from PV2_0093's screening budget rather than selected, so it is not a
# tuned number -- which is good for honesty and bad for confidence that it is the peak.
#
# This runs seeds 1 and 2 of the control, and of the group-count w5 arm which was the
# better of the two at seed 0. PV2_0098 is re-queued behind it: that experiment answers
# an integrity question about PV2_0092's control and is not time-critical, while this
# result is the one everything else would be built on.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0102_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="--num-feature-maps 8 --num-regions 1024 --device cuda --batch-size 16 --lr 1e-3
  --osc-dim 4 --epochs 8 --gamma-drive-mode static --num-time-steps 64
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
  --train-limit 1000 --plv-target-groups 7.5 --gamma-seq-path $G32"

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
  else printf "%-22s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0102: seeds 1 and 2 of the 8-epoch 32x32 model ==="
say "seed 0: control .7699 .7531 .7017 (pred_fg .1512) | w5 .7803 .7807 .6914 (.1444)"
say "16x16 published: .7239 .7251 .4951 (pred_fg .1290)"

cd /export_home/tkim1/snn_kuramoto_bidirectional
for s in 1 2; do
  for w in 0 5; do
    name=S32_w${w}_s$s
    mkdir -p $R/$name
    t0=$(date +%s)
    $PY training/train_s2net_core.py $BASE --seed $s --plv-group-count-weight $w \
       --save-path $R/$name/core.pt > $R/$name.log 2>&1
    say "--- $name  $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/$name.log | tail -1 | cut -d= -f2) ---"
  done
done

say ""
say "label                   fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "w0 seed 0" $R/GC_w0/core.pt
ev "w0 seed 1" $R/S32_w0_s1/core.pt
ev "w0 seed 2" $R/S32_w0_s2/core.pt
ev "w5 seed 0" $R/GC_w5/core.pt
ev "w5 seed 1" $R/S32_w5_s1/core.pt
ev "w5 seed 2" $R/S32_w5_s2/core.pt

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
import numpy as np
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0102_results.txt"):
    m = re.match(r"w(\d+) seed (\d)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        rows.setdefault(int(m.group(1)), {})[int(m.group(2))] = \
            tuple(float(m.group(i)) for i in (3,4,5,6,7))
G16 = (0.7239, 0.7251, 0.4951, 0.1290)
print("16x16 published (3-seed, validation): %.4f / %.4f / %.4f at pred_fg %.4f" % G16)
print()
for w in sorted(rows):
    v = rows[w]
    if len(v) < 2:
        print("  w%-2d  only %d seed(s); not a mean." % (w, len(v))); continue
    a = np.array([v[s] for s in sorted(v)])
    m, sd = a.mean(0), a.std(0, ddof=1)
    print("  w%-2d  n=%d  fg_ari %.4f +- %.4f | fg_iou %.4f +- %.4f | obj_iou %.4f +- %.4f | pred_fg %.4f"
          % (w, len(v), m[0], sd[0], m[1], sd[1], m[2], sd[2], m[4]))
    beats = m[0] > G16[0] and m[1] > G16[1] and m[2] > G16[2]
    cal = m[4] <= 0.20
    print("       beats 16x16 on all three: %s | calibrated: %s" % (beats, cal))
    if beats and cal:
        print("       deltas %+.4f / %+.4f / %+.4f" % (m[0]-G16[0], m[1]-G16[1], m[2]-G16[2]))
print()
ok = [w for w, v in rows.items() if len(v) >= 3]
if ok:
    best = max(ok, key=lambda w: np.array([rows[w][s] for s in sorted(rows[w])]).mean(0)[0])
    a = np.array([rows[best][s] for s in sorted(rows[best])]).mean(0)
    if a[0] > G16[0] and a[1] > G16[1] and a[2] > G16[2] and a[4] <= 0.20:
        print("THREE SEEDS, ALL THREE METRICS, CALIBRATED. w%d is the candidate." % best)
        print("Next: a contract-v2 Slot baseline with --preprocess resize, an epoch sweep")
        print("to locate the peak, and only then one test read.")
    else:
        print("The seed-0 result does not survive three seeds on all three metrics.")
EOF
say "=== PV2_0102 DONE ==="

# PV2_0098 re-queued behind this
say ""
say "# re-queuing PV2_0098"
nohup bash /work/USERS/tkim1/datascale.sh > /work/USERS/tkim1/datascale.out 2>&1 &
say "# PV2_0098 relaunched as pid $!"
