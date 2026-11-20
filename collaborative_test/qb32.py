"""Item 2: how close does the 32x32 model get to its own pixel-protocol ceiling?

The 16x16 ceiling is 0.6762 and our 16x16 model reaches 0.6125 -- a gap of 0.0637
with the rest taken by the grid. The 32x32 ceiling is 0.8340. This asks what the
32x32 model actually scores under the field's protocol. Saved predictions only,
no training. Diagnostic about protocols, not a contract score.
"""
import sys, json
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from evaluation import patch_fg_ari, patch_foreground_iou, patch_matched_object_iou
from make_targets_grid import cell_index, mask_cells

H, W = 240, 320
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]
pix = torch.load("/work/USERS/tkim1/figures/pixel_truth_val100.pt", map_location="cpu")

def expand(lab, grid):
    row, _ = cell_index(H, grid, lab.device); col, _ = cell_index(W, grid, lab.device)
    return lab[:, row][:, :, col].contiguous()

def score(p, t, tag):
    a = patch_fg_ari(p, t); f = patch_foreground_iou(p, t); o = patch_matched_object_iou(p, t)
    print("  %-38s FG-ARI %.4f   fg-IoU %.4f   obj-IoU %.4f"
          % (tag, np.nanmean(a.numpy()), np.nanmean(f.numpy()), np.nanmean(o.numpy())), flush=True)
    return np.nanmean(a.numpy())

p32 = torch.as_tensor(np.load("/work/USERS/tkim1/figures/cov_diag_pred.npz",
                              allow_pickle=True)["patch_labels"]).long()
p32 = p32.reshape(p32.shape[0], 32, 32)
n = p32.shape[0]
print("32x32 predictions: %d images (seed 0 / GC_w0)" % n)
print()
print("PIXEL protocol")
c32 = score(expand(mask_cells(pix[:n], 32)[0], 32), pix[:n], "CEILING perfect patch model 32x32")
o32 = score(expand(p32, 32), pix[:n], "OURS 32x32 components readout")
p16 = torch.as_tensor(np.load("/work/USERS/tkim1/figures/pred_grid16_cc.npz",
                              allow_pickle=True)["patch_labels"]).long()
p16 = p16.reshape(p16.shape[0], 16, 16)[:n]
c16 = score(expand(mask_cells(pix[:n], 16)[0], 16), pix[:n], "CEILING perfect patch model 16x16")
o16 = score(expand(p16, 16), pix[:n], "OURS 16x16 components readout")
print()
print("  gap to its own ceiling:  16x16 %.4f     32x32 %.4f" % (c16 - o16, c32 - o32))
print("  ceiling gained by the finer grid: %+.4f" % (c32 - c16))
print("  score  gained by the finer grid: %+.4f" % (o32 - o16))
