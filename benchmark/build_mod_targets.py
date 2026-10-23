"""Patch targets for any multi_object_datasets suite, built with the contract's cell rule.

Tier 1 of BENCHMARK_PLAN.md. The same modal vote make_targets_grid.py applies to CLEVR
is applied here, so a number from one dataset can sit beside a number from another.

Two things are verified rather than assumed, both of which bit on the first attempt:

MASK AXIS ORDER differs by dataset. Measured on three records each, by how often
neighbouring pixels share a label (a real segmentation is well above 0.9):

    multi_dsprites   EHW 0.0349   HWE 0.9764   -> HWE
    tetrominoes      EHW 0.9384   HWE 0.8490   -> EHW
    CLEVR            EHW 0.9926   HWE 0.9796   -> EHW

Reading a channels-last mask as channels-first leaves a period-E stripe in the labels
and a purity near chance -- and raises no error, so the first Multi-dSprites build
produced targets with purity 0.2813 against 0.2 for a random 5-way assignment. The
coherence assert below is what stops that reaching a benchmark table.

CELL SIZE differs enormously. At grid 32 a cell is 1.09 px on Tetrominoes, 2.00 px on
Multi-dSprites and 7.50 x 10.00 px on CLEVR. The pixel-protocol ceiling we measured for
CLEVR -- 0.8355 at grid 32 -- is a consequence of its large images, not of the patch
architecture. On the small suites the grid is near pixel resolution and that handicap
nearly vanishes.
"""
import sys, json, argparse
import numpy as np, torch
sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from make_targets_grid import mask_cells
from training.prepare_clevr_with_masks import iter_records, _parse_example

KNOWN = {
    "tetrominoes":    dict(hw=35, entities=4, order="EHW",
                           records="/work/USERS/tkim1/mod/tetrominoes_train.tfrecords"),
    "multi_dsprites": dict(hw=64, entities=5, order="HWE",
                           records="/work/USERS/tkim1/mod/multi_dsprites_colored_on_colored.tfrecords"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(KNOWN))
ap.add_argument("--limit", type=int, default=8000)
ap.add_argument("--grids", type=int, nargs="+", default=[16, 32])
ap.add_argument("--out-dir", default="/work/USERS/tkim1/mod")
args = ap.parse_args()
cfg = KNOWN[args.dataset]
H = W = cfg["hw"]; E = cfg["entities"]; ORDER = cfg["order"]

print("%s: %dx%d, %d entities, mask order %s" % (args.dataset, H, W, E, ORDER), flush=True)
for g in args.grids:
    edges = [round(i * H / g) for i in range(g + 1)]
    sizes = sorted({edges[i + 1] - edges[i] for i in range(g)})
    print("  grid %-3d cell sizes %s%s" % (g, sizes, "  UNEVEN" if len(sizes) > 1 else ""))

frames, images = [], []
for i, rec in enumerate(iter_records(cfg["records"], limit=args.limit)):
    f = _parse_example(rec)
    raw = f["mask"]
    m = (raw.reshape(E, H, W) if ORDER == "EHW"
         else np.moveaxis(raw.reshape(H, W, E), -1, 0)) > 127
    lab = torch.zeros(H, W, dtype=torch.int64)
    for e in range(E):
        lab[torch.as_tensor(m[e])] = e
    frames.append(lab)
    images.append(torch.as_tensor(f["image"].reshape(H, W, 3).copy()))
    if (i + 1) % 2000 == 0:
        print("  read %d" % (i + 1), flush=True)
pix = torch.stack(frames); img = torch.stack(images)
print("pixel labels", tuple(pix.shape), " images", tuple(img.shape), flush=True)

chk = pix[:200]
coh = ((chk[:, :, 1:] == chk[:, :, :-1]).float().mean().item()
       + (chk[:, 1:, :] == chk[:, :-1, :]).float().mean().item()) / 2
print("neighbour-same %.4f (a real segmentation is well above 0.9)" % coh)
assert coh > 0.85, "label map is not spatially coherent -- wrong mask axis order?"

share = [(pix == e).float().mean().item() for e in range(E)]
print("pixel share per entity:", " ".join("%d:%.3f" % (e, s) for e, s in enumerate(share)))
assert int(np.argmax(share)) == 0, "entity 0 is not the largest; background convention differs"

stem = "%s/targets_%s" % (args.out_dir, args.dataset)
for grid in args.grids:
    labels, purity, ties, redges, cedges = mask_cells(pix, grid)
    print("grid %d: foreground %.4f, purity %.4f, ties %.4f"
          % (grid, (labels > 0).float().mean().item(), purity.mean().item(),
             ties.float().mean().item()), flush=True)
    torch.save({"patch_labels": labels, "patch_purity": purity, "patch_ties": ties,
                "grid": grid, "row_edges": redges, "col_edges": cedges,
                "dataset": args.dataset, "source": cfg["records"],
                "mask_order": ORDER, "count": int(pix.shape[0]),
                "contract_version": "v2-pixel", "manifest_version": 1},
               "%s_grid%d.pt" % (stem, grid))
    print("  saved %s_grid%d.pt" % (stem, grid), flush=True)

torch.save(pix, "%s/pixel_labels_%s.pt" % (args.out_dir, args.dataset))
torch.save(img, "%s/images_%s.pt" % (args.out_dir, args.dataset))
json.dump({"dataset": args.dataset, "version": 1, "mask_order": ORDER,
           "ranges": {"train": [0, 6000], "validation": [6000, 7000], "test": [7000, 8000]},
           "count": int(pix.shape[0]), "note": "same split shape as the CLEVR manifest"},
          open("%s/split_manifest_%s.json" % (args.out_dir, args.dataset), "w"), indent=2)
print("manifest written; done")
