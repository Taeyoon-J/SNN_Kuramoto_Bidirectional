"""How much of a pixel-protocol score does our patch grid cost, before any model error?

The field scores object discovery per PIXEL; evaluation_contract.md scores per patch
and forbids expanding a mask to pixels. BENCHMARK_PLAN.md calls that the decision the
table rests on. This measures its size.

  CEILING  pixel ground truth -> quantised to the patch grid by the contract's own
           modal vote -> expanded straight back to pixels -> scored against the
           original. A PERFECT patch-level segmenter, zero model error.
  OURS     our saved patch predictions, expanded the same way.

Not a contract score. A diagnostic about protocols. The pixel truth is cached so
re-runs skip the tfrecord scan.
"""
import sys, json, os
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from evaluation import patch_fg_ari, patch_foreground_iou, patch_matched_object_iou
from make_targets_grid import cell_index, mask_cells

H, W, N = 240, 320, 100
CACHE = "/work/USERS/tkim1/figures/pixel_truth_val%d.pt" % N
man = json.load(open("/export_home/tkim1/collaborative_test/data/split_manifest.json"))
vs, _ = man["ranges"]["validation"]

if os.path.exists(CACHE):
    pix = torch.load(CACHE, map_location="cpu")
    print("pixel truth from cache", tuple(pix.shape), flush=True)
else:
    from training.prepare_clevr_with_masks import iter_records, _parse_example, MAX_ENTITIES
    print("scanning the tfrecord ...", flush=True)
    out = []
    for i, rec in enumerate(iter_records(
            "/work/USERS/tkim1/clevr/with_masks/clevr_with_masks_train.tfrecords",
            limit=vs + N)):
        if i < vs: continue
        m = _parse_example(rec)["mask"].reshape(MAX_ENTITIES, H, W) > 127
        f = torch.zeros(H, W, dtype=torch.int64)
        for e in range(MAX_ENTITIES):
            f[torch.as_tensor(m[e])] = e
        out.append(f)
    pix = torch.stack(out); torch.save(pix, CACHE)
    print("pixel truth", tuple(pix.shape), "cached", flush=True)

def expand(labels, grid):
    row, _ = cell_index(H, grid, labels.device)
    col, _ = cell_index(W, grid, labels.device)
    return labels[:, row][:, :, col].contiguous()

def score(pred, tgt, tag):
    assert pred.shape == tgt.shape, (tag, tuple(pred.shape), tuple(tgt.shape))
    a = patch_fg_ari(pred, tgt); f = patch_foreground_iou(pred, tgt)
    o = patch_matched_object_iou(pred, tgt)
    print("  %-36s FG-ARI %.4f   fg-IoU %.4f   obj-IoU %.4f"
          % (tag, np.nanmean(a.numpy()), np.nanmean(f.numpy()), np.nanmean(o.numpy())), flush=True)

print()
print("PIXEL protocol (the field's), %d validation images" % N)
for grid in (16, 32):
    q = mask_cells(pix, grid)[0]
    score(expand(q, grid), pix, "CEILING: perfect patch model %dx%d" % (grid, grid))

p = np.load("/work/USERS/tkim1/figures/pred_grid16_cc.npz", allow_pickle=True)["patch_labels"]
pred16 = torch.as_tensor(p).long()
pred16 = pred16.reshape(pred16.shape[0], 16, 16)[:N]
n = pred16.shape[0]
print("  (our saved predictions cover %d of the %d images)" % (n, N))
score(expand(pred16, 16), pix[:n], "OURS components readout 16x16")

print()
print("PATCH protocol (our contract), the same images")
t16 = torch.load("/work/USERS/tkim1/clevr/with_masks/targets_v1.pt",
                 map_location="cpu")["patch_labels"][vs:vs + n]
score(pred16, t16, "OURS components readout 16x16")
