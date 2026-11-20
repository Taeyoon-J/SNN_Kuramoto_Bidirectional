"""Render the masks that were actually scored.

Reads the npz `evaluate_model.py --save-predictions` wrote, so the figure shows the
same integer label map the three metrics were computed on -- nothing is re-derived
here, and nothing is expanded to pixels. Ground truth comes from the same targets
file the run was scored against.

Each row is one image: the photograph, the target label map, the prediction, and the
disagreement. Per-image fg_ari / fg_iou / obj_iou are printed on the row so a reader
can tell a good row from a bad one rather than being shown only good ones.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
from evaluation import evaluate_patch_masks


def label_colours(n=24, seed=0):
    rng = np.random.default_rng(seed)
    cols = rng.uniform(0.25, 0.95, size=(n, 3))
    cols[0] = (0.10, 0.10, 0.12)          # background is always dark
    return ListedColormap(cols)


def relabel(a):
    """Map arbitrary IDs to 0..k-1, keeping 0 as 0, so colours stay readable."""
    out = np.zeros_like(a)
    for new, old in enumerate([v for v in np.unique(a) if v != 0], start=1):
        out[a == old] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--pick", default="spread",
                    choices=["spread", "first", "best", "worst"],
                    help="'spread' takes the best, the median and the worst so the "
                         "figure is not a selection of successes")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = np.load(args.predictions, allow_pickle=True)
    pred = torch.as_tensor(blob["patch_labels"]).long()
    grid = int(blob["grid"])
    names = [str(n) for n in blob["names"]]

    tgt_blob = torch.load(args.targets, map_location="cpu")
    manifest = json.load(open(args.manifest))
    start, end = manifest["ranges"][args.split]
    target = tgt_blob["patch_labels"][start:end][: pred.size(0)]

    scored = evaluate_patch_masks(pred, target)
    per_image = {k: v.numpy() for k, v in scored["per_image"].items()}
    agg = {k: float(v) for k, v in scored["mean"].items()}
    ari = per_image["fg_ari"]
    order = np.argsort(np.nan_to_num(ari, nan=-1))
    if args.pick == "first":
        chosen = list(range(min(args.rows, pred.size(0))))
    elif args.pick == "best":
        chosen = list(order[::-1][: args.rows])
    elif args.pick == "worst":
        chosen = list(order[: args.rows])
    else:
        valid = [i for i in order if not np.isnan(ari[i])]
        picks = np.linspace(0, len(valid) - 1, args.rows).round().astype(int)
        chosen = [valid[p] for p in picks][::-1]

    cmap = label_colours()
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13.5, 3.2 * len(chosen)),
                             squeeze=False)
    for row, idx in enumerate(chosen):
        img = Image.open(Path(args.image_dir) / names[idx]).convert("RGB")
        t = relabel(target[idx].numpy())
        p = relabel(pred[idx].numpy())
        wrong = ((target[idx].numpy() != 0) | (pred[idx].numpy() != 0)).astype(float)
        agree = (pred[idx].numpy() != 0) == (target[idx].numpy() != 0)
        disagree = np.where(agree, np.nan, wrong)

        for col, (data, title, kw) in enumerate((
                (np.asarray(img), "image", {}),
                (t, "target  %d x %d" % (grid, grid), dict(cmap=cmap, vmin=0, vmax=23)),
                (p, "prediction (scored)", dict(cmap=cmap, vmin=0, vmax=23)),
                (disagree, "foreground disagreement",
                 dict(cmap="autumn", vmin=0, vmax=1)))):
            ax = axes[row][col]
            if col == 3:
                ax.imshow(t != 0, cmap="Greys", vmin=0, vmax=3)
            ax.imshow(data, interpolation="nearest", **kw)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=11)
        axes[row][0].set_ylabel(
            "%s\nfg_ari %.3f\nfg_iou %.3f\nobj_iou %.3f"
            % (names[idx].replace("CLEVR_masks_", "#").replace(".png", ""),
               per_image["fg_ari"][idx], per_image["foreground_iou"][idx],
               per_image["matched_object_iou"][idx]),
            fontsize=9, rotation=0, ha="right", va="center", labelpad=42)

    head = args.title or "patch masks, %dx%d grid" % (grid, grid)
    fig.suptitle("%s   |   %d images: fg_ari %.4f  fg_iou %.4f  obj_iou %.4f"
                 % (head, pred.size(0), agg["fg_ari"],
                    agg["foreground_iou"], agg["matched_object_iou"]),
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=(0.03, 0, 1, 0.985))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print("wrote %s  (%d rows, picked by '%s')" % (args.out, len(chosen), args.pick))
    print("aggregate: fg_ari %.4f  fg_iou %.4f  obj_iou %.4f"
          % (agg["fg_ari"], agg["foreground_iou"], agg["matched_object_iou"]))


if __name__ == "__main__":
    main()
