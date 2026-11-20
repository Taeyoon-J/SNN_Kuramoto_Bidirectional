"""ORACLE DIAGNOSTIC. Two ceilings, separated.

A first version of this conflated two different oracle powers, and its own check caught
it: foreground IoU changed, which merging alone cannot do. The cause was that a predicted
group lying entirely on background was relabelled to 0 and therefore deleted, which is a
foreground correction, not a merge.

Separated here:

  MERGE ONLY     every predicted group keeps its foreground status; groups landing on the
                 same true object are given one label. A group lying entirely on
                 background keeps its own label and stays foreground. Foreground IoU is
                 then exactly unchanged by construction, which the output verifies.

  MERGE + DROP   the earlier behaviour. Groups whose majority overlap is background are
                 deleted. This is an oracle foreground correction on top of merging.

The gap between them is how much of the apparent ceiling was never about merging.

USES GROUND TRUTH. Diagnostic only, reported separately per the contract, never a score.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
from evaluation import evaluate_patch_masks


def merge(pred, target, drop_background_groups):
    out = torch.zeros_like(pred)
    for i in range(pred.size(0)):
        p, t = pred[i], target[i]
        # labels for groups that must stay foreground but match no object
        spare = int(max(int(t.max()), int(p.max()))) + 1
        for g in p.unique():
            if g == 0:
                continue
            mask = p == g
            vals, counts = t[mask].unique(return_counts=True)
            fg = vals != 0
            if fg.any():
                out[i][mask] = vals[fg][counts[fg].argmax()]
            elif drop_background_groups:
                out[i][mask] = 0
            else:
                out[i][mask] = spare     # stays foreground, its own group
                spare += 1
    return out


def groups(x):
    return float(np.mean([len([v for v in x[i].unique() if v != 0]) for i in range(x.size(0))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    s, e = man["ranges"][args.split]
    tgt_all = torch.load(args.targets, map_location="cpu")["patch_labels"][s:e]

    print("ORACLE DIAGNOSTIC -- uses ground truth, never a score.")
    print()
    print("%-30s %8s %8s %8s %8s" % ("", "fg_ari", "fg_iou", "obj_iou", "groups"))
    agg = {}
    for path in args.predictions:
        pred = torch.as_tensor(np.load(path, allow_pickle=True)["patch_labels"]).long()
        tgt = tgt_all[: pred.size(0)]
        pred = pred[: tgt.size(0)]
        name = Path(path).stem

        rows = [("as is", pred),
                ("merge only", merge(pred, tgt, False)),
                ("merge + oracle drop", merge(pred, tgt, True))]
        vals = {}
        for label, x in rows:
            m = evaluate_patch_masks(x, tgt)["mean"]
            v = (float(m["fg_ari"]), float(m["foreground_iou"]),
                 float(m["matched_object_iou"]), groups(x))
            vals[label] = v
            agg.setdefault(label, []).append(v)
            print("  %-28s %8.4f %8.4f %8.4f %8.2f" % (name + " " + label, *v))
        ok = abs(vals["merge only"][1] - vals["as is"][1]) < 1e-9
        print("  %-28s %s" % ("  foreground IoU preserved:",
                              "yes" if ok else "NO -- merge-only is still moving patches"))
        print("  %-28s merge %+.4f | drop adds %+.4f"
              % ("  fg_ari:", vals["merge only"][0] - vals["as is"][0],
                 vals["merge + oracle drop"][0] - vals["merge only"][0]))
        print()

    if len(agg.get("as is", [])) > 1:
        print("means over %d prediction sets:" % len(agg["as is"]))
        for label in ("as is", "merge only", "merge + oracle drop"):
            a = np.array(agg[label]).mean(0)
            print("  %-22s %8.4f %8.4f %8.4f %8.2f" % (label, *a))
        b = np.array(agg["as is"]).mean(0)
        m = np.array(agg["merge only"]).mean(0)
        d = np.array(agg["merge + oracle drop"]).mean(0)
        print()
        print("  merging alone is worth %+.4f fg_ari; the oracle foreground drop adds %+.4f."
              % (m[0] - b[0], d[0] - m[0]))


if __name__ == "__main__":
    main()
