#!/bin/bash
# PV2_0096. The third parameter in the wrong units: graph_top_k.
#
# PV2_0095 quantified the 32x32 failure: groups_per_image is 19.5 against 6.49 true
# objects. Three times too many. Widening the spatial kernel to sigma 4.0 recovers
# foreground IoU to 0.7091 against grid 16's 0.7251 and holds obj_iou at +0.0934, but
# fg_ari stays 0.0946 behind, and the over-segmentation is the whole of that gap.
# Raising --min-group-size made it worse, not better: the predicted groups are already
# too small, so a floor deletes objects rather than noise.
#
# --graph-top-k is 32 in both configurations. At 256 nodes that admits 12.5% of the
# graph; at 1024 nodes it admits 3.1%. The coupling graph is four times sparser relative
# to the node count, so groups cannot link up across an object. That is the same class
# of error PV2_0093 confirmed for the membrane threshold and PV2_0095 found in the
# spatial kernel: a constant that should have scaled with N.
#
# The graph is rebuilt from gamma at evaluation, so this is testable on the trained
# checkpoint in minutes. The checkpoint was TRAINED at top_k 32, so a larger k at
# evaluation is off-distribution -- which makes a gain here strong evidence and a loss
# here ambiguous, the same asymmetry PV2_0090's resize arm had.
#
# Matched fraction: 32/256 = 12.5% of 1024 is 128.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
OUT=$R/pv2_0096_results.txt
: > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ev () {  # label topk sigma
  local lab=$1 k=$2 s=$3
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint $R/G32_POP40/core.pt \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit 300 \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 --graph-top-k $k \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma $s --min-group-size 2 2>&1)
  # column order from evaluate_model.py: fg_ari, foreground_iou, matched_object_iou,
  # groups_per_image, predicted_foreground_fraction
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-26s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0096: graph_top_k is an absolute count at a 4x finer grid ==="
say "true objects per image on contract v2: 6.49"
say "grid 16 at matched extent:  fg_ari .7239  fg_iou .7251  obj_iou .4951  pred_fg .1290"
say ""
say "label                        fg_ari   fg_iou  obj_iou   groups  pred_fg"
for k in 32 64 128 192 256; do ev "top_k $k, sigma 4.0" $k 4.0; done
say ""
for k in 64 128; do ev "top_k $k, sigma 2.0" $k 2.0; done

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = []
for line in open("/work/USERS/tkim1/runs/pv2_0096_results.txt"):
    m = re.match(r"top_k (\d+), sigma ([\d.]+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2)),
                     *[float(m.group(i)) for i in (3,4,5,6,7)]))
G16 = (0.7239, 0.7251, 0.4951)
print("grid 16 at matched extent: %.4f / %.4f / %.4f   | true objects 6.49" % G16)
print()
print("%-22s %8s %8s %8s %8s %8s" % ("", "d fg_ari", "d fg_iou", "d obj_iou", "groups", "pred_fg"))
wins = []
for k, s, a, f, o, g, p in rows:
    tag = ""
    if p <= 0.20 and a > G16[0] and f > G16[1] and o > G16[2]:
        tag = "  <-- ALL THREE"; wins.append((k, s, a, f, o, p))
    print("  top_k %-4d sigma %-4.1f %+8.4f %+8.4f %+8.4f %8.2f %8.4f%s"
          % (k, s, a - G16[0], f - G16[1], o - G16[2], g, p, tag))
print()
if wins:
    b = max(wins, key=lambda r: r[2])
    print("32x32 BEATS GRID 16 ON ALL THREE at top_k %d, sigma %.1f:" % (b[0], b[1]))
    print("  %.4f / %.4f / %.4f at predicted foreground %.4f" % b[2:])
    print("Retrain at this top_k (it was trained at 32, so this is off-distribution),")
    print("then seeds 1 and 2, then the contract-v2 Slot baseline with --preprocess resize.")
else:
    cal = [r for r in rows if r[6] <= 0.20]
    if cal:
        b = max(cal, key=lambda r: r[2])
        g32 = next(r for r in rows if r[0] == 32 and r[1] == 4.0)
        print("Best calibrated: top_k %d sigma %.1f at %.4f / %.4f / %.4f, groups %.2f"
              % (b[0], b[1], b[2], b[3], b[4], b[5]))
        print("  against top_k 32 sigma 4.0: fg_ari %+.4f, groups %.2f -> %.2f"
              % (b[2] - g32[2], g32[5], b[5]))
        if b[5] < g32[5] - 2:
            print()
            print("The sparsity IS part of the over-segmentation -- groups fall toward the")
            print("true 6.49 -- but it does not close fg_ari on its own at this checkpoint,")
            print("which was trained at top_k 32. A retrain is what the gain would need.")
        else:
            print()
            print("top_k does not move the over-segmentation. The fragments are in what the")
            print("spikes encode, not in how densely the graph is wired.")
EOF
say "=== PV2_0096 DONE ==="
