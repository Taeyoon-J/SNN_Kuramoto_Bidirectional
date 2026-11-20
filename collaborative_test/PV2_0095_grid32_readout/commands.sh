#!/bin/bash
# PV2_0095. The 32x32 readout fragments, and two of its knobs are in grid units.
#
# PV2_0094 trained a healthy 1024-node model and read it at matched extent against grid
# 16: obj_iou +0.0889, fg_ari -0.1195, fg_iou -0.0590. The figures show single true
# objects split across several predicted clusters, which is what fg_ari penalises and
# what Hungarian matching forgives.
#
# PV2_0093 found that an ABSOLUTE membrane threshold tuned at 256 oscillators does not
# transfer to 1024. Two readout knobs have the same shape of problem, and neither was
# touched when the grid changed:
#
#   --spatial-sigma 1.0   the Gaussian on the patch grid is in GRID units, so at grid 32
#                         it covers HALF the image distance it did at grid 16. Less
#                         smoothing, more fragments. Matched physical scale is 2.0.
#   --min-group-size 2    also in patches. An object covers ~4x more patches at grid 32,
#                         so 2 admits fragments that 8 would reject.
#
# Unlike spatial_rate, which PV2_0089 learned the hard way is an nn.Parameter restored by
# load_state_dict, both of these are pure evaluation-time arguments -- verified at
# evaluate_model.py:387-409 and :476 -- so this costs minutes, not a retrain.
#
# If fragmentation is the whole story, a matched sigma should recover fg_ari while
# keeping the +0.0889 obj_iou, which would make 32x32 strictly better than 16x16 on all
# three. If it does not, the fragmentation is in the spikes and not the smoothing.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
FIG=/work/USERS/tkim1/figures
OUT=$R/pv2_0095_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ev () {  # label extra...
  local lab=$1; shift
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/G32_POP40/core.pt \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|predicted_foreground|groups_per_image" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-34s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0095: the 32x32 readout knobs that are in grid units ==="
say "grid 16 reference at matched extent: fg_ari .7239  fg_iou .7251  obj_iou .4951  pred_fg .1290"
say "PV2_0094 (sigma 1.0, min 2):         fg_ari .6044  fg_iou .6661  obj_iou .5840  pred_fg .1268"
say ""
say "label                               fg_ari   fg_iou  obj_iou  pred_fg  groups"
for s in 1.0 1.5 2.0 3.0 4.0; do
  ev "sigma $s, min 2" --spatial-sigma $s --min-group-size 2
done
say ""
for m in 4 8 12; do
  ev "sigma 2.0, min $m" --spatial-sigma 2.0 --min-group-size $m
done
ev "sigma 3.0, min 8" --spatial-sigma 3.0 --min-group-size 8

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = []
for line in open("/work/USERS/tkim1/runs/pv2_0095_results.txt"):
    m = re.match(r"(sigma \S+, min \d+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        rows.append((m.group(1), *[float(m.group(i)) for i in (2,3,4,5)]))

G16 = (0.7239, 0.7251, 0.4951, 0.1290)   # grid 16, components, validation 300
print("against grid 16 (%.4f / %.4f / %.4f at pred_fg %.4f)" % G16)
print()
best_all = None
for name, a, f, o, p in rows:
    d = (a - G16[0], f - G16[1], o - G16[2])
    tag = ""
    if a > G16[0] and f > G16[1] and o > G16[2]:
        tag = "  <-- BEATS GRID 16 ON ALL THREE"
        if best_all is None or a > best_all[1]: best_all = (name, a, f, o, p)
    print("  %-22s %+.4f %+.4f %+.4f  pred_fg %.4f%s" % (name, *d, p, tag))
print()
if best_all:
    print("THE FRAGMENTATION WAS THE SMOOTHING. %s beats grid 16 on all three" % best_all[0])
    print("at %.4f / %.4f / %.4f, predicted foreground %.4f." % best_all[1:])
    print("Promote to seeds 1 and 2, then build the contract-v2 Slot baseline with")
    print("--preprocess resize before any comparison.")
else:
    cal = [r for r in rows if r[4] <= 0.20]
    if cal:
        b = max(cal, key=lambda r: r[1])
        print("No setting beats grid 16 on all three. Best calibrated: %s at" % b[0])
        print("  %.4f / %.4f / %.4f (pred_fg %.4f), fg_ari %+.4f vs grid 16."
              % (b[1], b[2], b[3], b[4], b[1] - G16[0]))
    print()
    print("So the fragmentation is not only the spatial kernel's width. It is in what")
    print("the spikes encode at this grid, and smoothing can trade it for extent but")
    print("not remove it.")
EOF
say "=== PV2_0095 DONE ==="
