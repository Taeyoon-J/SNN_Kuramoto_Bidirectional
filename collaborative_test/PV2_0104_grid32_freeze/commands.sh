#!/bin/bash
# PV2_0104. The graph freeze at 32x32, three seeds.
#
# PV2_0103 rescued seed 1 two ways. Its automated verdict picked the 16-epoch arm on
# fg_ari alone (0.6867 against 0.6777) -- the fifth time in this project that reading
# fg_ari alone would mislead. On all three the freeze is clearly better:
#
#   A freeze   0.6777 / 0.6895 / 0.6357   54 min
#   B 16 ep    0.6867 / 0.6008 / 0.4887  341 min
#
# +0.0887 foreground IoU and +0.1470 matched-object IoU for -0.0090 fg_ari, at a sixth
# of the cost. obj_iou is the metric over-prediction cannot buy.
#
# This completes the three-seed picture for the freeze. Seeds 1 and 2 initialise from
# seed 0's 8-epoch checkpoint with graph_generator frozen, so all three share one graph
# -- which is exactly what the published three seeds do at grid 16 (PV2_0044), and is
# disclosed as a limitation in STATUS.md. Seed 0 is that checkpoint, so it needs no run.
#
# Estimated beforehand, using seed 2's unfrozen score as a stand-in: fg_ari ~0.718,
# fg_iou ~0.686, obj_iou ~0.635 against the published 0.7239 / 0.7251 / 0.4951. That
# would be a tie on fg_ari, a loss on foreground IoU and a large win on per-object IoU --
# not a clean win. This run replaces the stand-in with a measurement.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0104_results.txt
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
  --train-limit 1000 --gamma-seq-path $G32
  --init-from $R/GC_w0/core.pt --freeze-components graph_generator"

ev () {
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
  else printf "%-18s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0104: graph freeze at 32x32, three seeds sharing seed 0's graph ==="
cd /export_home/tkim1/snn_kuramoto_bidirectional
mkdir -p $R/FRZ_s2
t0=$(date +%s)
$PY training/train_s2net_core.py $BASE --seed 2 --save-path $R/FRZ_s2/core.pt > $R/FRZ_s2.log 2>&1
say "FRZ_s2 $(( ($(date +%s)-t0)/60 )) min  collapse $(grep -oE 'spike_plv_collapse=[0-9.]+' $R/FRZ_s2.log | tail -1 | cut -d= -f2)"

say ""
say "label              fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "seed 0" $R/GC_w0/core.pt
ev "seed 1" $R/RSC_freeze/core.pt
ev "seed 2" $R/FRZ_s2/core.pt

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
import numpy as np
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0104_results.txt"):
    m = re.match(r"seed (\d)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m: rows[int(m.group(1))] = tuple(float(m.group(i)) for i in (2,3,4,5,6))
G16 = (0.7239, 0.7251, 0.4951, 0.1290)
UNFROZEN = (0.5894, 0.4876, 0.5007)   # PV2_0102, same budget, no freeze
print("16x16 published:        %.4f / %.4f / %.4f at pred_fg %.4f" % G16)
print("32x32 unfrozen (PV2_0102): %.4f / %.4f / %.4f  (std .2599 on fg_ari)" % UNFROZEN)
print()
if len(rows) < 3:
    print("only %d seeds; not a mean." % len(rows))
else:
    a = np.array([rows[s] for s in sorted(rows)])
    m, sd = a.mean(0), a.std(0, ddof=1)
    for s in sorted(rows):
        print("  seed %d  %.4f / %.4f / %.4f  groups %5.2f  pred_fg %.4f" % (s, *rows[s]))
    print()
    print("  MEAN    %.4f +- %.4f | %.4f +- %.4f | %.4f +- %.4f | pred_fg %.4f"
          % (m[0], sd[0], m[1], sd[1], m[2], sd[2], m[4]))
    print()
    print("  spread on fg_ari: %.4f frozen against 0.2599 unfrozen" % sd[0])
    wins = [m[i] > G16[i] for i in range(3)]
    names = ("fg_ari", "fg_iou", "obj_iou")
    print("  beats 16x16 on: %s" % (", ".join(n for n, w in zip(names, wins) if w) or "nothing"))
    print("  calibrated: %s (pred_fg %.4f)" % (m[4] <= 0.20, m[4]))
    print()
    if all(wins) and m[4] <= 0.20:
        print("THREE SEEDS, ALL THREE METRICS, CALIBRATED. Next: an epoch sweep to find")
        print("the peak, a contract-v2 Slot baseline with --preprocess resize, then one")
        print("test read.")
    elif sd[0] < 0.10:
        print("The freeze does what PV2_0044 did at grid 16: it closes the spread. The")
        print("mean does not clear 16x16 on all three, so 32x32's honest offer is a large")
        print("obj_iou gain against a foreground-IoU loss -- a different operating point,")
        print("not a strictly better model.")
    else:
        print("The freeze does not close the spread at 1024 nodes.")
EOF
say "=== PV2_0104 DONE ==="
