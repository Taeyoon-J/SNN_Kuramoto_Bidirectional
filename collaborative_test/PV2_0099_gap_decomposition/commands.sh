#!/bin/bash
# PV2_0099. Decomposing the gap to the peer's contract.
#
# PV2_0046 measured that one published checkpoint scores 0.6195 under our contract and
# 0.8946 under theirs -- 0.275 on the same model. ROUTE_MAP_0.9_v2.md treats the distance
# to 0.9 as a model gap; this measures how much of the peer comparison is measurement.
#
# The model is held fixed and only the contract moves: the SAME Slot Attention
# checkpoints, scored on a 2x2 of {grid 16, grid 32} x {whole 240x320 frame, centre
# 240x240 crop}. The crop is the one run_slot_attention.py applies, so cropped targets
# describe exactly the region the model sees -- PV2_0090's clean arm, extended to both
# grids.
#
# Peer's contract is 128x128 with 4x4 cells. We cannot reproduce it byte for byte: their
# records give the pool, the IDs and the cell rule but never the crop geometry. What this
# CAN separate is the two axes their contract differs on that we can control -- grid
# resolution and whether the frame is cropped -- on our own split, leaving only their
# label-rule and split differences unmeasured, and PV2_0046 already puts the label rule
# at 0.19.
set -u
export PYTHONPATH=/export_home/tkim1:.
PY=/work/USERS/tkim1/envs/miniforge3/envs/snn/bin/python
TF=/work/USERS/tkim1/envs/miniforge3/envs/tfgpu/bin/python
CT=/export_home/tkim1/collaborative_test
MAN=$CT/data/split_manifest.json
IMG=/work/USERS/tkim1/clevr/with_masks/images10k
D=/work/USERS/tkim1/decomp
OUT=/work/USERS/tkim1/runs/pv2_0099_results.txt
mkdir -p $D; : > $OUT
say () { echo "[$(date +%H:%M)] $*" | tee -a $OUT; }
export CUDA_VISIBLE_DEVICES=2 TF_CPP_MIN_LOG_LEVEL=3

say "=== PV2_0099: how much of the peer gap is the contract? ==="
say "same checkpoints throughout; only the measurement moves"

say ""
say "--- ground truth, four variants ---"
cd $CT
PYTHONPATH=/export_home/tkim1:/export_home/tkim1/snn_kuramoto_bidirectional:. $PY decomp.py \
  --tfrecord /work/USERS/tkim1/clevr/with_masks/clevr_with_masks_train.tfrecords \
  --manifest $MAN --split validation --limit 300 --out-dir $D 2>&1 | tail -6 | tee -a $OUT

say ""
say "--- Slot Attention predictions at grid 32 (grid 16 exists from PV2_0090) ---"
for seed in 0 1 2; do
  for pre in crop resize; do
    f=/work/USERS/tkim1/runs/sa_g32_s${seed}_${pre}.npz
    [ -s "$f" ] && continue
    $TF run_slot_attention.py --checkpoint-dir /work/USERS/tkim1/runs/SA_s$seed \
      --model-dir /work/USERS/tkim1/baselines/slot_attention --image-dir $IMG \
      --manifest $MAN --split validation --limit 300 --grid 32 --preprocess $pre \
      --seed 0 --out $f > /tmp/sag32_${seed}_${pre}.log 2>&1 \
      || { say "  seed $seed $pre FAILED"; tail -3 /tmp/sag32_${seed}_${pre}.log >> $OUT; }
  done
  say "  seed $seed done"
done

say ""
$PY - <<'EOF' 2>&1 | tee -a $OUT
import numpy as np, torch, sys
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
from evaluation import evaluate_patch_masks

D = "/work/USERS/tkim1/decomp"
gt = {(f, g): torch.load("%s/gt_%s_grid%d.pt" % (D, f, g), map_location="cpu")["patch_labels"]
      for f in ("whole", "crop") for g in (16, 32)}

def preds(seed, pre, g):
    p = ("/work/USERS/tkim1/runs/sa_align_s%d_%s.npz" % (seed, pre) if g == 16
         else "/work/USERS/tkim1/runs/sa_g32_s%d_%s.npz" % (seed, pre))
    return torch.as_tensor(np.load(p, allow_pickle=True)["patch_labels"]).long()

print("Slot Attention, validation 300, three seeds. Same checkpoints in every cell.")
print()
print("%-30s %8s %8s %8s" % ("contract", "fg_ari", "fg_iou", "obj_iou"))
cells = {}
for pre, frame in (("crop", "crop"), ("resize", "whole")):
    for g in (16, 32):
        vals = []
        for seed in (0, 1, 2):
            try:
                lab = preds(seed, pre, g)
            except Exception:
                continue
            t = gt[(frame, g)][: len(lab)]
            m = evaluate_patch_masks(lab[: len(t)], t)["mean"]
            vals.append([float(m["fg_ari"]), float(m["foreground_iou"]),
                         float(m["matched_object_iou"])])
        if vals:
            a = np.array(vals).mean(0)
            cells[(frame, g)] = a
            print("  %-28s %8.4f %8.4f %8.4f"
                  % ("%s frame, grid %d" % (frame, g), *a))

print()
print("decomposition on fg_ari, same model throughout:")
if ("whole", 16) in cells and ("crop", 16) in cells:
    print("  grid 16: whole %.4f -> crop %.4f   (crop is worth %+.4f)"
          % (cells[("whole",16)][0], cells[("crop",16)][0],
             cells[("crop",16)][0] - cells[("whole",16)][0]))
if ("whole", 32) in cells and ("crop", 32) in cells:
    print("  grid 32: whole %.4f -> crop %.4f   (crop is worth %+.4f)"
          % (cells[("whole",32)][0], cells[("crop",32)][0],
             cells[("crop",32)][0] - cells[("whole",32)][0]))
if ("whole", 16) in cells and ("whole", 32) in cells:
    print("  whole frame: grid 16 %.4f -> grid 32 %.4f  (finer grid is worth %+.4f)"
          % (cells[("whole",16)][0], cells[("whole",32)][0],
             cells[("whole",32)][0] - cells[("whole",16)][0]))
if ("crop", 16) in cells and ("crop", 32) in cells:
    print("  crop:        grid 16 %.4f -> grid 32 %.4f  (finer grid is worth %+.4f)"
          % (cells[("crop",16)][0], cells[("crop",32)][0],
             cells[("crop",32)][0] - cells[("crop",16)][0]))
if ("whole", 16) in cells and ("crop", 32) in cells:
    total = cells[("crop",32)][0] - cells[("whole",16)][0]
    print()
    print("  our contract's geometry -> peer-like geometry: %+.4f on fg_ari" % total)
    print("  PV2_0046 puts the LABEL RULE alone at 0.19, and the full observed")
    print("  cross-contract difference on one checkpoint was 0.6195 -> 0.8946 = 0.275.")
    print("  So geometry %+.4f + label rule ~0.19 accounts for %.4f of that 0.275."
          % (total, total + 0.19))
EOF
say "=== PV2_0099 DONE ==="
