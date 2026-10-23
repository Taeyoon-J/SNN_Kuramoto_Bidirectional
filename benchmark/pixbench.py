"""Score our Tier 1 model under the FIELD's protocol, so the number is comparable.

BENCHMARK_PLAN.md settled that the paper table goes pixel-protocol: every published
number in this field is per pixel, and a patch-protocol table can cite nothing. This
converts the patch prediction to pixels with the contract's own cell edges and scores it
against the PIXEL labels.

It also measures the ceiling the same way -- quantise the pixel truth to the patch grid
and expand it straight back -- so what the grid costs is separated from what the model
costs. On CLEVR that ceiling is 0.6973 at grid 16 and 0.8355 at grid 32; on these
datasets a grid-32 cell is 1.09 px and 2.00 px, so it should be near 1.
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from evaluation import patch_fg_ari, patch_foreground_iou, patch_matched_object_iou
from make_targets_grid import cell_index, mask_cells

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--predictions", required=True)
ap.add_argument("--grid", type=int, default=32)
ap.add_argument("--src", default="/work/USERS/tkim1/mod")
args = ap.parse_args()

man = json.load(open("%s/split_manifest_%s.json" % (args.src, args.dataset)))
lo, hi = man["ranges"]["validation"]
pix = torch.load("%s/pixel_labels_%s.pt" % (args.src, args.dataset), map_location="cpu")
pred = torch.as_tensor(np.load(args.predictions, allow_pickle=True)["patch_labels"]).long()
n = pred.shape[0]
pred = pred.reshape(n, args.grid, args.grid)
truth = pix[lo:lo + n]
H, W = truth.shape[1], truth.shape[2]
print("%s: %d images, source %dx%d, prediction grid %d" % (args.dataset, n, H, W, args.grid))

def expand(lab):
    row, _ = cell_index(H, args.grid, lab.device)
    col, _ = cell_index(W, args.grid, lab.device)
    return lab[:, row][:, :, col].contiguous()

def score(p, t, tag):
    a = patch_fg_ari(p, t); f = patch_foreground_iou(p, t); o = patch_matched_object_iou(p, t)
    print("  %-42s FG-ARI %.4f   fg-IoU %.4f   obj-IoU %.4f"
          % (tag, np.nanmean(a.numpy()), np.nanmean(f.numpy()), np.nanmean(o.numpy())))
    return np.nanmean(a.numpy())

print()
print("PIXEL protocol (the field's)")
ceil = score(expand(mask_cells(truth, args.grid)[0]), truth,
             "CEILING: perfect patch model, grid %d" % args.grid)
ours = score(expand(pred), truth, "OURS, grid %d" % args.grid)
print()
print("  gap to its own ceiling: %.4f   (CLEVR's grid-32 ceiling is 0.8355)" % (ceil - ours))
print()
print("  published, PIXEL ARI-FG, same dataset family:")
print("    CAE .00/.05/.04   CAE++ .78/.51/.27   CtCAE .84/.56/.54")
print("    Rotating Features .42/.84/.65   SynCx .89/.82/.59   (tetrominoes/dsprites/clevr)")
