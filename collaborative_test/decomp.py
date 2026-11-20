"""Build the four ground-truth variants the gap decomposition needs.

{grid 16, grid 32} x {whole 240x320 frame, centre 240x240 crop}, all under the same
modal rule with ties to the smallest ID. The crop is the one run_slot_attention.py
applies, so the cropped targets describe exactly the region Slot Attention sees.

240/32 = 7.5 does not divide, so cells use round(i * length / grid) boundaries -- the
rule PV2_0088 verified identical to contract v1 at grid 16.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")


def cell_index(length, grid, device):
    edges = [round(i * length / grid) for i in range(grid + 1)]
    assert edges[0] == 0 and edges[-1] == length
    assert all(b > a for a, b in zip(edges, edges[1:])), "grid finer than the axis"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out-dir", required=True)
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
    whole = torch.stack(frames).to(args.device)
    h, w = whole.shape[1], whole.shape[2]
    side = min(h, w)
    crop = whole[:, (h - side) // 2:(h - side) // 2 + side,
                 (w - side) // 2:(w - side) // 2 + side]
    print("whole %s  crop %s at offset (%d,%d)"
          % (tuple(whole.shape[1:]), tuple(crop.shape[1:]), (h - side) // 2, (w - side) // 2))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, pix in (("whole", whole), ("crop", crop)):
        for g in (16, 32):
            lab = mask_cells(pix, g)
            fg = lab != 0
            objs = torch.tensor([lab[i][fg[i]].unique().numel() for i in range(lab.size(0))]).float()
            path = "%s/gt_%s_grid%d.pt" % (args.out_dir, name, g)
            torch.save({"patch_labels": lab, "contract_version": "decomp-%s-%d" % (name, g)}, path)
            print("  %-6s grid %2d  foreground %.3f  objects/image %.2f  -> %s"
                  % (name, g, fg.float().mean(), objs.mean(), Path(path).name))


if __name__ == "__main__":
    main()
