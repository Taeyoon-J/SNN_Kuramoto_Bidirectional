"""Evaluation targets on a grid that need not divide the image.

Contract v1 fixes a 16x16 grid because 240/16 = 15 and 320/16 = 20 divide
exactly, and `clevr_mask_patch` requires that. A 32x32 grid on 240x320 does
not divide: 240/32 = 7.5.

The rule here is the same modal rule on cells whose boundaries are
round(i * H / g), so the cells partition the image exactly and alternate
between 7 and 8 rows. At g = 16 the boundaries are round(15 i) = 15 i, so this
reduces to contract v1's own cells and must reproduce `targets_v1.pt` bit for
bit -- which `--verify-against` checks before any other grid is trusted.

Ties go to the smallest instance ID, so background wins a tie. Background is 0.
Nothing is resized, cropped or expanded to pixels.
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")


def cell_index(length, grid, device):
    """Per-pixel cell index along one axis, from round(i * length / grid)."""
    edges = torch.tensor([round(i * length / grid) for i in range(grid + 1)], device=device)
    if edges[0] != 0 or edges[-1] != length:
        raise ValueError("cell edges must span the axis exactly.")
    if (edges[1:] <= edges[:-1]).any():
        raise ValueError("grid is finer than the axis; some cell would be empty.")
    index = torch.zeros(length, dtype=torch.long, device=device)
    for i in range(grid):
        index[edges[i]:edges[i + 1]] = i
    return index, edges


def mask_cells(labels, grid, chunk=256):
    """Modal instance ID per cell. labels [B, H, W] int64 -> [B, grid, grid]."""
    batch, height, width = labels.shape
    device = labels.device
    row, redges = cell_index(height, grid, device)
    col, cedges = cell_index(width, grid, device)
    flat_cell = (row.view(-1, 1) * grid + col.view(1, -1)).reshape(-1)
    ids = torch.unique(labels, sorted=True)

    out_labels = torch.empty(batch, grid * grid, dtype=torch.int64)
    out_purity = torch.empty(batch, grid * grid, dtype=torch.float32)
    out_ties = torch.empty(batch, grid * grid, dtype=torch.bool)
    size = torch.zeros(grid * grid, device=device).index_add_(
        0, flat_cell, torch.ones(flat_cell.numel(), device=device))

    for start in range(0, batch, chunk):
        part = labels[start:start + chunk].reshape(-1, height * width)
        n = part.size(0)
        counts = torch.zeros(n, grid * grid, ids.numel(), device=device)
        for j, value in enumerate(ids.tolist()):
            hit = (part == value).float()
            counts[:, :, j] = torch.zeros(n, grid * grid, device=device).index_add_(
                1, flat_cell, hit)
        best, selected = counts.max(dim=-1)
        out_labels[start:start + n] = ids[selected].cpu()
        out_purity[start:start + n] = (best / size).cpu()
        out_ties[start:start + n] = ((counts == best.unsqueeze(-1)).sum(-1) > 1).cpu()
        print("  %d / %d" % (min(start + chunk, batch), batch), flush=True)

    shape = (batch, grid, grid)
    return (out_labels.view(shape), out_purity.view(shape), out_ties.view(shape),
            redges.tolist(), cedges.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--contract", default="")
    ap.add_argument("--verify-against", default="",
                    help="targets_v1.pt; the grid-16 output must equal it exactly")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from training.prepare_clevr_with_masks import iter_records, _parse_example, IMAGE_SHAPE, MAX_ENTITIES

    manifest = json.load(open(args.manifest))
    limit = max(end for _, end in manifest["ranges"].values())
    height, width = IMAGE_SHAPE[0], IMAGE_SHAPE[1]

    pixels, names = [], []
    for index, record in enumerate(iter_records(args.tfrecord, limit)):
        feats = _parse_example(record)
        masks = feats["mask"].reshape(MAX_ENTITIES, height, width) > 127
        frame = torch.zeros(height, width, dtype=torch.int64)
        for entity in range(1, MAX_ENTITIES):
            frame[torch.as_tensor(masks[entity])] = entity
        pixels.append(frame)
        names.append("CLEVR_masks_%06d.png" % index)
        if (index + 1) % 2000 == 0:
            print("  read %d images" % (index + 1), flush=True)

    stack = torch.stack(pixels).to(args.device)
    labels, purity, ties, redges, cedges = mask_cells(stack, args.grid)

    if args.verify_against:
        ref = torch.load(args.verify_against, map_location="cpu")
        same = torch.equal(labels, ref["patch_labels"][: labels.size(0)])
        print("verify against %s: labels %s" % (args.verify_against,
                                                "IDENTICAL" if same else "DIFFER"))
        if not same:
            diff = (labels != ref["patch_labels"][: labels.size(0)]).float().mean()
            print("  %.4f%% of cells differ -- the grid rule does not reduce to contract v1"
                  % (100 * diff))
            sys.exit(1)

    fg = labels != 0
    per_image = torch.tensor([labels[i][fg[i]].unique().numel() for i in range(labels.size(0))])
    print("grid %d: foreground %.1f%% of cells | %.2f objects per image | ties %.2f%%"
          % (args.grid, 100 * fg.float().mean(), per_image.float().mean(),
             100 * ties.float().mean()))
    print("  row edges %s..." % redges[:6])
    print("  mean purity %.3f" % purity.mean())

    if args.out:
        torch.save({"names": names, "patch_labels": labels, "patch_purity": purity,
                    "patch_ties": ties, "patch_size": None, "grid": args.grid,
                    "row_edges": redges, "col_edges": cedges,
                    "contract_version": args.contract or ("%d" % args.grid),
                    "manifest_version": manifest["version"]}, args.out)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
