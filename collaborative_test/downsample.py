"""Does voting on 128x128 masks instead of native 240x320 masks explain the rest?

PV2_0099 closed only 0.107 of the 0.275 cross-contract difference. The axis it did not
test is the one the peer's contract states and ours does not: their ground truth comes
from a 128x128 HDF5, so the instance masks were downsampled before any cell voting
happened. Ours votes on the native 240x320 masks.

Downsampling an integer label map deletes small objects before they can be counted. That
is the same mechanism PV2_0099 found running the other way -- the crop made cells smaller
and SURFACED small objects, costing 0.087 -- so this should make the measurement easier
by a comparable amount or more.

The model never moves. The same Slot Attention predictions are scored against targets
that differ only in whether the masks were downsampled first.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from evaluation import evaluate_patch_masks


def cell_index(length, grid, device):
    edges = [round(i * length / grid) for i in range(grid + 1)]
    idx = torch.zeros(length, dtype=torch.long, device=device)
    for i in range(grid):
        idx[edges[i]:edges[i + 1]] = i
    return idx


def mask_cells(labels, grid, chunk=128):
    b, h, w = labels.shape
    dev = labels.device
    flat = (cell_index(h, grid, dev).view(-1, 1) * grid
            + cell_index(w, grid, dev).view(1, -1)).reshape(-1)
    ids = torch.unique(labels, sorted=True)
    out = torch.empty(b, grid * grid, dtype=torch.int64)
    for s in range(0, b, chunk):
        part = labels[s:s + chunk].reshape(-1, h * w)
        n = part.size(0)
        counts = torch.zeros(n, grid * grid, ids.numel(), device=dev)
        for j, v in enumerate(ids.tolist()):
            counts[:, :, j] = torch.zeros(n, grid * grid, device=dev).index_add_(
                1, flat, (part == v).float())
        out[s:s + n] = ids[counts.argmax(-1)].cpu()
    return out.view(b, grid, grid)


def describe(lab):
    fg = lab != 0
    objs = torch.tensor([lab[i][fg[i]].unique().numel() for i in range(lab.size(0))]).float()
    return float(fg.float().mean()), float(objs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from training.prepare_clevr_with_masks import iter_records, _parse_example, IMAGE_SHAPE, MAX_ENTITIES
    man = json.load(open(args.manifest))
    start, end = man["ranges"][args.split]
    end = min(end, start + args.limit)

    frames = []
    for i, rec in enumerate(iter_records(args.tfrecord, end)):
        if i < start:
            continue
        f = _parse_example(rec)
        m = f["mask"].reshape(MAX_ENTITIES, IMAGE_SHAPE[0], IMAGE_SHAPE[1]) > 127
        px = torch.zeros(IMAGE_SHAPE[0], IMAGE_SHAPE[1], dtype=torch.int64)
        for e in range(1, MAX_ENTITIES):
            px[torch.as_tensor(m[e])] = e
        frames.append(px)
    native = torch.stack(frames).to(args.device)

    # nearest-neighbour, which is the only correct resize for an integer label map
    small = F.interpolate(native.unsqueeze(1).float(), size=(128, 128),
                          mode="nearest").squeeze(1).long()
    print("native %s -> downsampled %s" % (tuple(native.shape[1:]), tuple(small.shape[1:])))
    print()
    print("%-34s %10s %12s" % ("ground truth", "foreground", "objects/img"))
    gts = {}
    for name, pix in (("native 240x320", native), ("downsampled 128x128", small)):
        for g in (16, 32):
            lab = mask_cells(pix, g)
            gts[(name, g)] = lab
            f, o = describe(lab)
            print("  %-32s %10.3f %12.2f" % ("%s, grid %d" % (name, g), f, o))

    print()
    print("the same Slot Attention predictions, scored against each")
    print("%-40s %8s %8s %8s" % ("", "fg_ari", "fg_iou", "obj_iou"))
    res = {}
    for g in (16, 32):
        for name in ("native 240x320", "downsampled 128x128"):
            vals = []
            for seed in (0, 1, 2):
                p = ("/work/USERS/tkim1/runs/sa_align_s%d_resize.npz" % seed if g == 16
                     else "/work/USERS/tkim1/runs/sa_g32_s%d_resize.npz" % seed)
                try:
                    lab = torch.as_tensor(np.load(p, allow_pickle=True)["patch_labels"]).long()
                except Exception:
                    continue
                t = gts[(name, g)][: len(lab)]
                m = evaluate_patch_masks(lab[: len(t)], t)["mean"]
                vals.append([float(m["fg_ari"]), float(m["foreground_iou"]),
                             float(m["matched_object_iou"])])
            if vals:
                a = np.array(vals).mean(0)
                res[(name, g)] = a
                print("  %-38s %8.4f %8.4f %8.4f" % ("grid %d, %s" % (g, name), *a))

    print()
    print("effect of voting on downsampled masks, fg_ari:")
    for g in (16, 32):
        k1, k2 = ("native 240x320", g), ("downsampled 128x128", g)
        if k1 in res and k2 in res:
            d = res[k2][0] - res[k1][0]
            print("  grid %2d: %.4f -> %.4f   (%+.4f)" % (g, res[k1][0], res[k2][0], d))
    print()
    k1, k2 = ("native 240x320", 16), ("downsampled 128x128", 32)
    if k1 in res and k2 in res:
        d = res[k2][0] - res[k1][0]
        print("our contract's GT -> peer-like GT (128 masks, grid 32): %+.4f" % d)
        print()
        print("PV2_0099 measured geometry at -0.083 and PV2_0046 the label rule at +0.19.")
        print("Observed cross-contract difference on one checkpoint: +0.275.")
        print("  geometry -0.083 + label rule +0.19 + downsampling %+.4f = %+.4f"
              % (d, -0.083 + 0.19 + d))
        if abs(-0.083 + 0.19 + d - 0.275) < 0.06:
            print("  The decomposition closes. The peer gap is measurement, not modelling.")
        else:
            print("  Still %.4f unaccounted for; the remaining axis is their split."
                  % (0.275 - (-0.083 + 0.19 + d)))


if __name__ == "__main__":
    main()
