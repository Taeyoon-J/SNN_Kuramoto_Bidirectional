"""ORACLE DIAGNOSTIC. Why are the pieces wrong, after merging is accounted for?

PV2_0105 split the remaining gap in two. From the frozen 32x32 model at 0.7109, perfect
merging buys +0.1133 to 0.8242, and PV2_0082's perfect segmenter reaches 0.9756. So
+0.151 is not about merging at all -- it is the pieces themselves.

This measures what is wrong with them, on the same predictions, with three quantities
that decompose it:

  PURITY        for each predicted group, the fraction of its patches that belong to its
                majority true object. 1.0 means the group sits inside one object and only
                merging is missing. Below 1.0 means groups STRADDLE objects, which no
                merging step can repair -- merging can only make a straddling group
                bigger.

  COVERAGE      for each true object, the fraction of its patches that any predicted
                group claims. Below 1.0 means parts of objects are called background and
                no readout can recover them.

  SPLIT COUNT   how many predicted groups a true object is divided into. This is the part
                merging repairs.

USES GROUND TRUTH. Diagnostic only, reported separately per the contract, never a score.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/export_home/tkim1/collaborative_test")


def stats(pred, target):
    purity, straddle, coverage, splits, lost = [], [], [], [], []
    for i in range(pred.size(0)):
        p, t = pred[i], target[i]
        for g in p.unique():
            if g == 0:
                continue
            vals, counts = t[p == g].unique(return_counts=True)
            counts = counts.float()
            purity.append(float(counts.max() / counts.sum()))
            fg = vals != 0
            # a group straddles if two or more TRUE OBJECTS each take >=20% of it
            if fg.sum() >= 2:
                share = counts[fg] / counts.sum()
                straddle.append(float((share >= 0.20).sum() >= 2))
            else:
                straddle.append(0.0)
        for o in t.unique():
            if o == 0:
                continue
            claimed = p[t == o]
            coverage.append(float((claimed != 0).float().mean()))
            ids = [v for v in claimed.unique() if v != 0]
            splits.append(len(ids) if ids else 0)
            lost.append(float(len(ids) == 0))
    return (np.mean(purity), np.mean(straddle), np.mean(coverage),
            np.mean(splits), np.mean(lost))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    s, e = man["ranges"][args.split]
    tgt_all = torch.load(args.targets, map_location="cpu")["patch_labels"][s:e]

    print("ORACLE DIAGNOSTIC -- uses ground truth, never a score.")
    if args.label:
        print(args.label)
    print()
    print("%-22s %8s %9s %9s %8s %8s"
          % ("", "purity", "straddle", "coverage", "splits", "lost"))
    rows = []
    for path in args.predictions:
        pred = torch.as_tensor(np.load(path, allow_pickle=True)["patch_labels"]).long()
        tgt = tgt_all[: pred.size(0)]
        v = stats(pred[: tgt.size(0)], tgt)
        rows.append(v)
        print("  %-20s %8.4f %9.4f %9.4f %8.2f %8.4f" % (Path(path).stem, *v))
    if len(rows) > 1:
        a = np.array(rows).mean(0)
        print("  %-20s %8.4f %9.4f %9.4f %8.2f %8.4f" % ("MEAN", *a))
    else:
        a = np.array(rows[0])
    print()
    print("purity   %.4f -- a predicted group is this much inside one true object" % a[0])
    print("straddle %.4f -- this fraction of groups span two objects at >=20%% each;" % a[1])
    print("            merging cannot repair these, it can only enlarge them")
    print("coverage %.4f -- a true object has this much of itself claimed as foreground;" % a[2])
    print("            the rest is called background and no readout recovers it")
    print("splits   %.2f  -- a true object is divided into this many predicted groups;" % a[3])
    print("            this is exactly what merging repairs")
    print("lost     %.4f -- this fraction of true objects is claimed by nothing" % a[4])
    print()
    if a[1] < 0.10 and a[2] > 0.90:
        print("The pieces are clean: they sit inside objects and cover them. What is")
        print("missing is only the merging, and PV2_0105's +0.1133 is collectable in")
        print("principle by a readout stage with no new signal.")
    elif a[1] >= 0.10:
        print("A meaningful fraction of groups STRADDLE objects. Merging cannot fix")
        print("those, so part of PV2_0105's ceiling is unreachable by merging alone and")
        print("the base grouping has to change.")
    if a[2] <= 0.90:
        print("Coverage is the other limit: %.1f%% of each object is already called" % (100*(1-a[2])))
        print("background before any grouping happens.")


if __name__ == "__main__":
    main()
