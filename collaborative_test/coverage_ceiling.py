"""ORACLE. Completing the decomposition: what is the 12.7% coverage worth?

piece_quality measured that 12.7% of each true object is already called background before
grouping, and that only 3.1% of predicted groups straddle two objects. So merging is
largely achievable and coverage is the other limit. This prices it.

Four oracles, cumulative:
  as is          the model's own output
  + merge        groups on one object get one label (PV2_0105's merge-only)
  + coverage     unclaimed patches of a true object join that object's group
  + both         the ceiling of fixing grouping and figure/ground together

PV2_0082 puts a perfect segmenter at 0.9756 under this contract's own rule, so "+ both"
should approach it; the residual is where neither oracle reaches.

USES GROUND TRUTH. Diagnostic only, never a score.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
from evaluation import evaluate_patch_masks

MAN = "/export_home/tkim1/collaborative_test/data/split_manifest.json"
T32 = "/work/USERS/tkim1/clevr/with_masks/targets_v2_grid32.pt"
PRED = ["/work/USERS/tkim1/figures/frz32_s%d.npz" % s for s in (0, 1, 2)]

man = json.load(open(MAN)); s, e = man["ranges"]["validation"]
tgt_all = torch.load(T32, map_location="cpu")["patch_labels"][s:e]


def transform(pred, target, do_merge, do_cover):
    out = pred.clone()
    for i in range(pred.size(0)):
        p, t = pred[i], target[i]
        o = out[i]
        if do_merge:
            spare = int(max(int(t.max()), int(p.max()))) + 1
            for g in p.unique():
                if g == 0:
                    continue
                mask = p == g
                vals, counts = t[mask].unique(return_counts=True)
                fg = vals != 0
                if fg.any():
                    o[mask] = vals[fg][counts[fg].argmax()]
                else:
                    o[mask] = spare; spare += 1
        if do_cover:
            # a true object's unclaimed patches join whichever group already holds most
            # of that object; if none does, they form one new group
            spare = int(o.max()) + 1
            for obj in t.unique():
                if obj == 0:
                    continue
                here = t == obj
                claimed = o[here]
                ids = [int(v) for v in claimed.unique() if v != 0]
                if ids:
                    best = max(ids, key=lambda v: int((claimed == v).sum()))
                else:
                    best = spare; spare += 1
                o[here & (o == 0)] = best
    return out


print("ORACLE DIAGNOSTIC -- uses ground truth, never a score.")
print("32x32 frozen, three seeds, validation 300. PV2_0082: perfect segmenter = 0.9756")
print()
print("%-24s %8s %8s %8s %8s" % ("", "fg_ari", "fg_iou", "obj_iou", "groups"))
agg = {}
for name, dm, dc in (("as is", False, False), ("+ merge", True, False),
                     ("+ coverage", False, True), ("+ both", True, True)):
    vals = []
    for path in PRED:
        pred = torch.as_tensor(np.load(path, allow_pickle=True)["patch_labels"]).long()
        tgt = tgt_all[: pred.size(0)]; pred = pred[: tgt.size(0)]
        x = transform(pred, tgt, dm, dc)
        m = evaluate_patch_masks(x, tgt)["mean"]
        g = float(np.mean([len([v for v in x[i].unique() if v != 0]) for i in range(x.size(0))]))
        vals.append([float(m["fg_ari"]), float(m["foreground_iou"]),
                     float(m["matched_object_iou"]), g])
    a = np.array(vals).mean(0); agg[name] = a
    print("  %-22s %8.4f %8.4f %8.4f %8.2f" % (name, *a))

b = agg["as is"]
print()
print("worth on fg_ari:  merging %+.4f | coverage %+.4f | both %+.4f"
      % (agg["+ merge"][0]-b[0], agg["+ coverage"][0]-b[0], agg["+ both"][0]-b[0]))
print("残 to a perfect segmenter (0.9756): %.4f" % (0.9756 - agg["+ both"][0]))
print()
if agg["+ both"][0] >= 0.90:
    print("MERGING AND COVERAGE TOGETHER CLEAR 0.9. Both are readout-side problems with")
    print("ceilings above the goal, so a route exists that keeps the core untouched:")
    print("a hierarchical merge plus a foreground stage that stops losing 12.7% of each")
    print("object. Neither needs a new signal -- both act on spikes the model already has.")
else:
    print("Even both oracles together reach only %.4f. A route to 0.9 needs something"
          % agg["+ both"][0])
    print("the readout cannot supply, and the residual %.4f is in the spikes themselves."
          % (0.9756 - agg["+ both"][0]))
