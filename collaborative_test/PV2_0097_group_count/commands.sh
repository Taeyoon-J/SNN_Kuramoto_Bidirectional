#!/bin/bash
# PV2_0097. The objective has no term for how many groups form. Turn it on.
#
# PV2_0094/0095/0096 measured the 32x32 failure precisely: groups_per_image is 16 to 20
# against 6.49 true objects, and three readout knobs -- spatial kernel width, group size
# floor, graph density -- move it by almost nothing while fg_ari sits at 0.63. So the
# over-segmentation is in what the spikes encode.
#
# loss_function.py:598 already contains the term for this. plv_group_count_loss targets
# the effective group count through the participation ratio PR = N^2 / ||PLV||_F^2 --
# 1 when everything locks, N when nothing does, k for k equal blocks. It is exposed as
# --plv-group-count-weight and --plv-target-groups, and the published configuration
# leaves it at zero, controlling a scalar mean synchrony instead. Its own docstring says
# it was written to replace that, because targeting a mean is ill-posed: the ideal mean
# PLV is 0.867 if the background is one group and 0.003 if it is incoherent, so an
# intermediate target pulls away from both.
#
# Target 7.5 = 6.49 objects plus one background group, which is what the participation
# ratio counts.
#
# One variable: the weight. Everything else is PV2_0094's command, which trains healthy
# (collapse 1.86 against grid 16's 1.90). 8 epochs per arm, as PV2_0093 used, because
# the diagnostic here is groups_per_image and it is readable early.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0097_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="--num-feature-maps 8 --num-regions 1024 --device cuda --batch-size 16 --lr 1e-3
  --osc-dim 4 --epochs 8 --seed 0 --gamma-drive-mode static --num-time-steps 64
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
    --grid 32 --num-regions 1024 --split validation --limit 200 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-26s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0097: a term for the number of groups, at 32x32 ==="
say "true objects 6.49, participation-ratio target 7.5 (objects + background)"
say "PV2_0094 at 8 epochs is not measured; weight 0 here is the control"

cd /export_home/tkim1/snn_kuramoto_bidirectional
for w in 0 5 20; do
  name=GC_w$w
  mkdir -p $R/$name
  t0=$(date +%s)
  $PY training/train_s2net_core.py $BASE --plv-group-count-weight $w \
     --save-path $R/$name/core.pt > $R/$name.log 2>&1
  say "--- $name  $(( ($(date +%s)-t0)/60 )) min  $(grep -oE 'loss: [0-9.]+ -> [0-9.]+' $R/$name.log | tail -1) ---"
  say "    collapse: $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/$name.log | awk -F= '{printf "%.2f ", $2}')"
  grep -oE "plv_group_count=[0-9.e+-]+" $R/$name.log | tail -1 | sed 's/^/    final /' | tee -a $OUT
done

say ""
say "label                        fg_ari   fg_iou  obj_iou   groups  pred_fg"
for w in 0 5 20; do ev "GC_w$w" $R/GC_w$w/core.pt; done

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0097_results.txt"):
    m = re.match(r"GC_w(\d+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m: rows[int(m.group(1))] = tuple(float(m.group(i)) for i in (2,3,4,5,6))
if 0 not in rows:
    print("control missing; nothing can be judged.")
else:
    b = rows[0]
    print("control (weight 0): fg_ari %.4f  fg_iou %.4f  obj_iou %.4f  groups %.2f  pred_fg %.4f" % b)
    print("true objects 6.49")
    print()
    moved = []
    for w in sorted(rows):
        if w == 0: continue
        v = rows[w]
        print("  weight %-3d  fg_ari %+.4f  fg_iou %+.4f  obj_iou %+.4f  groups %.2f -> %.2f  pred_fg %.4f"
              % (w, v[0]-b[0], v[1]-b[1], v[2]-b[2], b[3], v[3], v[4]))
        if v[3] < b[3] - 2 and v[4] <= 0.20:
            moved.append((w, v))
    print()
    if moved:
        w, v = max(moved, key=lambda r: r[1][0])
        print("THE GROUP-COUNT TERM MOVES THE GROUP COUNT: %.2f -> %.2f at weight %d," % (b[3], v[3], w))
        print("fg_ari %+.4f, staying calibrated at %.4f." % (v[0]-b[0], v[4]))
        print("Run the full 40 epochs at this weight, then compare with grid 16 at")
        print("matched extent, then seeds 1 and 2.")
    else:
        print("The term does not reduce the group count inside the calibrated regime at")
        print("8 epochs. Either it needs the full budget, or the participation ratio of")
        print("the PLV matrix is not what the readout's connected components count.")
EOF
say "=== PV2_0097 DONE ==="
