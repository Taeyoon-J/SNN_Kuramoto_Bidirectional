"""What the patch grid costs, before any model is involved.

Multi-dSprites at grid 32 measured a ceiling of 0.5619 -- below CtCAE's published 0.56 and
far below SynCx's 0.82. If that holds, the protocol decides the comparison before the model
runs, and benchmark.md's "grid 32 or finer" is wrong for that dataset. This quantises the
pixel truth to each grid and expands it straight back: no model, just the grid's own cost.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from evaluation import patch_fg_ari, patch_foreground_iou, patch_matched_object_iou
from make_targets_grid import cell_index, mask_cells

print("%-16s %-6s %-9s %-8s %-8s %-8s" % ("dataset", "grid", "px/cell", "FG-ARI", "fg-IoU", "obj-IoU"))
for ds in ("tetrominoes", "multi_dsprites"):
    man = json.load(open("/work/USERS/tkim1/mod/split_manifest_%s.json" % ds))
    lo, _ = man["ranges"]["validation"]
    pix = torch.load("/work/USERS/tkim1/mod/pixel_labels_%s.pt" % ds, map_location="cpu")[lo:lo + 300]
    H, W = pix.shape[1], pix.shape[2]
    for g in (16, 32, 64, 128):
        if g > min(H, W):
            continue
        lab = mask_cells(pix, g)[0]
        row, _ = cell_index(H, g, lab.device); col, _ = cell_index(W, g, lab.device)
        up = lab[:, row][:, :, col].contiguous()
        print("%-16s %-6d %-9.2f %-8.4f %-8.4f %-8.4f" % (
            ds, g, H / g,
            np.nanmean(patch_fg_ari(up, pix).numpy()),
            np.nanmean(patch_foreground_iou(up, pix).numpy()),
            np.nanmean(patch_matched_object_iou(up, pix).numpy())))
    print()
