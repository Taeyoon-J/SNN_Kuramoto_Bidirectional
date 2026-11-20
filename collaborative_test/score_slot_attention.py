"""Score the Slot Attention reference through the contract's own functions.

Its slot ids are arbitrary, so one of them has to be called background, and the
rule has to be the same kind of rule our model gets: no ground truth. Two
non-oracle rules are reported, because which one is fair is not obvious and the
choice should be visible rather than buried:

- largest: the slot covering the most patches overall, the same rule our
  classifier uses when it calls its biggest synchrony component background.
- border: the slot covering the most patches on the image border, the usual
  heuristic when a background slot is split across several slots.

Both are reported. Neither looks at the targets.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_patch_masks


def remap(labels, rule, num_slots):
    grid = labels.shape[1]
    if rule == "largest":
        background = np.bincount(labels.reshape(-1), minlength=num_slots).argmax()
    else:
        border = np.zeros(num_slots, dtype=np.int64)
        for a in (labels[:, 0, :], labels[:, -1, :], labels[:, :, 0], labels[:, :, -1]):
            border += np.bincount(a.reshape(-1), minlength=num_slots)
        background = border.argmax()
    out = np.where(labels == background, 0, labels + 1)
    return out, int(background)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot-labels", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    blob = np.load(args.slot_labels, allow_pickle=True)
    # the npz already remapped once; recover the raw slot ids
    raw = blob["patch_labels"]
    raw = np.where(raw == 0, blob["background_slot"], raw - 1)
    names = [str(n) for n in blob["names"]]
    num_slots = int(blob["num_slots"])

    manifest = json.load(open(args.manifest))
    start, _ = manifest["ranges"][args.split]
    targets = torch.load(args.targets)
    if targets["names"][start:start + len(names)] != names:
        raise ValueError("Slot Attention image order does not match the manifest.")
    target = targets["patch_labels"][start:start + len(names)]

    report = {"reference": "Slot Attention, single published checkpoint (ckpt-500)",
              "images": len(names), "split": args.split, "rules": {}}
    print("%-10s %9s %9s %9s %9s" % ("rule", "fg_ari", "fg_iou", "obj_iou", "bg frac"))
    print("-" * 52)
    for rule in ("largest", "border"):
        remapped, background = remap(raw, rule, num_slots)
        prediction = torch.from_numpy(remapped).to(torch.int64)
        scored = evaluate_patch_masks(prediction, target)
        means = {k: float(v) for k, v in scored["mean"].items()}
        report["rules"][rule] = {
            "background_slot": background,
            "metrics": means,
            "valid_count": {k: int(v) for k, v in scored["valid_count"].items()},
            "predicted_background_fraction": float((remapped == 0).mean()),
        }
        print("%-10s %9.4f %9.4f %9.4f %9.4f"
              % (rule, means["fg_ari"], means["foreground_iou"],
                 means["matched_object_iou"], float((remapped == 0).mean())))
    print()
    print("target background fraction %.4f" % float((target == 0).float().mean()))
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=1)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
