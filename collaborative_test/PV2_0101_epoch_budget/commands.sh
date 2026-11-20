#!/bin/bash
# PV2_0101. Is the 32x32 model overtrained? Same images, same settings, two budgets.
#
# PV2_0097's control -- 8 epochs, otherwise PV2_0094's command -- scored 0.7734 / 0.7571
# / 0.7047 on validation 200, while PV2_0094's 40-epoch model scored 0.6293 / 0.7091 /
# 0.5885 on validation 300 at the same sigma 4.0. Better on all three with a fifth of the
# training.
#
# The validation sets differ, 200 against 300, so that comparison is not yet a result.
# This re-scores BOTH checkpoints on the SAME 300 images with the SAME arguments, which
# is the only way the difference means anything.
#
# If it holds, PV2_0094's 40-epoch budget was the error, the 32x32 model peaks early, and
# at 8 epochs it beats grid 16 on all three inside the calibrated regime -- which nothing
# in this project has done yet.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
R=/work/USERS/tkim1/runs
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
G32=/work/USERS/tkim1/gamma_sequences/wm10k_full_grid32.pt
T32=/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt
FIG=/work/USERS/tkim1/figures
OUT=$R/pv2_0101_results.txt
mkdir -p $FIG; : > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/tkim1_triton_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ev () {  # label checkpoint limit extra...
  local lab=$1 ck=$2 lim=$3; shift 3
  [ -s "$ck" ] || { say "$lab MISSING"; return; }
  local out r
  out=$(cd $CT && $PY evaluate_model.py --checkpoint "$ck" \
    --gamma-seq-path $G32 --targets $T32 --manifest $MAN --expect-contract 2 \
    --grid 32 --num-regions 1024 --split validation --limit $lim \
    --spike-per-component --geodesic-steps 3 --geodesic-contrast 2.0 \
    --graph-spatial-decay 0.35 --geodesic-chunk 16 \
    --membrane-threshold-mode population --membrane-threshold-k 0.5 \
    --num-time-steps 1024 --settle 512 --synchrony-threshold 0.05 \
    --spatial-sigma 4.0 --min-group-size 2 "$@" 2>&1)
  r=$(echo "$out" | grep -E "^fg_ari|^foreground_iou|^matched_object_iou|groups_per_image_mean|predicted_foreground" | awk '{print $2}' | tr '\n' ' ')
  if [ -z "$r" ]; then say "$lab FAILED"; echo "$out" | tail -4 >> $OUT
  else printf "%-32s %s\n" "$lab" "$r" | tee -a $OUT; fi
}

say "=== PV2_0101: 8 epochs against 40, same 300 images, same arguments ==="
say "grid 16 reference at matched extent: .7239 .7251 .4951  pred_fg .1290"
say ""
say "label                             fg_ari   fg_iou  obj_iou   groups  pred_fg"
ev "40 epochs (PV2_0094), val 300" $R/G32_POP40/core.pt 300 --save-predictions $FIG/g32_e40.npz
ev "8 epochs (PV2_0097 w0), val 300" $R/GC_w0/core.pt 300 --save-predictions $FIG/g32_e8.npz
ev "8 epochs + group-count w5, val 300" $R/GC_w5/core.pt 300 --save-predictions $FIG/g32_e8w5.npz
say ""
say "# the 200-image rows PV2_0097 reported, for provenance"
ev "40 epochs, val 200" $R/G32_POP40/core.pt 200
ev "8 epochs, val 200" $R/GC_w0/core.pt 200

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import re
rows = {}
for line in open("/work/USERS/tkim1/runs/pv2_0101_results.txt"):
    m = re.match(r"(.+?), val (\d+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        rows[(m.group(1).strip(), int(m.group(2)))] = tuple(float(m.group(i)) for i in (3,4,5,6,7))
G16 = (0.7239, 0.7251, 0.4951, 0.1290)
print("grid 16 at matched extent: %.4f / %.4f / %.4f at pred_fg %.4f" % G16)
print()
for (name, lim), v in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    print("  %-36s val %3d  %.4f / %.4f / %.4f  groups %5.2f  pred_fg %.4f"
          % (name, lim, v[0], v[1], v[2], v[3], v[4]))
print()
k40, k8 = ("40 epochs (PV2_0094)", 300), ("8 epochs (PV2_0097 w0)", 300)
if k40 in rows and k8 in rows:
    a, b = rows[k40], rows[k8]
    print("on the SAME 300 images, 8 epochs minus 40 epochs:")
    print("  fg_ari %+.4f   fg_iou %+.4f   obj_iou %+.4f   groups %.2f -> %.2f"
          % (b[0]-a[0], b[1]-a[1], b[2]-a[2], a[3], b[3]))
    print()
    if b[4] <= 0.20 and b[0] > G16[0] and b[1] > G16[1] and b[2] > G16[2]:
        print("32x32 AT 8 EPOCHS BEATS GRID 16 ON ALL THREE, INSIDE THE CALIBRATED REGIME:")
        print("  %.4f / %.4f / %.4f at predicted foreground %.4f against" % (b[0],b[1],b[2],b[4]))
        print("  %.4f / %.4f / %.4f at %.4f." % G16)
        print("  PV2_0094's 40-epoch budget was the error. Next: seeds 1 and 2, an epoch")
        print("  sweep to find the peak, and a contract-v2 Slot baseline with")
        print("  --preprocess resize before this is compared with anything external.")
    elif b[0] > a[0] + 0.05:
        print("The 32x32 model is overtrained at 40 epochs: %+.4f fg_ari from training" % (b[0]-a[0]))
        print("one fifth as long. It does not clear grid 16 on all three, but the budget")
        print("was wrong and every 32x32 number so far was read off an overtrained model.")
    else:
        print("The 8/40 difference does not survive the matched image set; PV2_0097's")
        print("control looked strong because it was scored on 200 images, not 300.")
EOF
say "=== PV2_0101 DONE ==="
