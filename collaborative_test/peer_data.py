"""Build our data under the peer's contract: their crop, their grid, their split IDs.

Confirmed by the user: the peer uses clevr_10-full.hdf5 from
google-deepmind/multi_object_datasets, which is the 240x320 originals centre-cropped to
192x192 and scaled to 128x128. PV2_0108 verified the reconstruction -- ground-truth
foreground under that geometry is 0.214, inside the 0.199-0.217 the peer reports.

Their splits: TRAIN pool 0-999 and 1640-70639, VALIDATION 1320-1639.

LEAKAGE GUARD. Their train pool is non-contiguous and straddles validation, while
--train-limit takes the FIRST N rows of whatever tensor it is given. One tensor over
0-9999 with --train-limit 4096 would therefore train on validation IDs 1320-1639. Two
separate tensors make that impossible rather than merely discouraged, and this script
asserts the two ID sets are disjoint before writing anything.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/export_home/tkim1/collaborative_test")
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")

CROP = 192
SIZE = 128
GRID = 32


def peer_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    left, top = (w - CROP) // 2, (h - CROP) // 2
    im = im.crop((left, top, left + CROP, top + CROP)).resize((SIZE, SIZE), Image.BILINEAR)
    return torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(SIZE, SIZE, 3).permute(2, 0, 1)


def build_gamma(ids, root, kernels, bias, device, batch=256):
    out = []
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        x = torch.stack([peer_image(Path(root) / ("CLEVR_masks_%06d.png" % i)) for i in chunk])
        x = x.to(device).float() / 255.0
        with torch.no_grad():
            f = F.conv2d(x, kernels, bias=bias)
            g = F.adaptive_avg_pool2d(f, (GRID, GRID)).flatten(2)
        out.append(g.cpu())
    g = torch.cat(out)
    return (g - g.mean()) / g.std()


def cell_index(length, grid, device):
    edges = [round(i * length / grid) for i in range(grid + 1)]
    idx = torch.zeros(length, dtype=torch.long, device=device)
    for i in range(grid):
        idx[edges[i]:edges[i + 1]] = i
    return idx


def mask_cells(labels, grid, chunk=64):
    b, h, w = labels.shape
    dev = labels.device
    flat = (cell_index(h, grid, dev).view(-1, 1) * grid + cell_index(w, grid, dev).view(1, -1)).reshape(-1)
    ids = torch.unique(labels, sorted=True)
    out = torch.empty(b, grid * grid, dtype=torch.int64)
    for s in range(0, b, chunk):
        part = labels[s:s + chunk].reshape(-1, h * w)
        n = part.size(0)
        counts = torch.zeros(n, grid * grid, ids.numel(), device=dev)
        for j, v in enumerate(ids.tolist()):
            counts[:, :, j] = torch.zeros(n, grid * grid, device=dev).index_add_(1, flat, (part == v).float())
        out[s:s + n] = ids[counts.argmax(-1)].cpu()
    return out.view(b, grid, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/work/USERS/tkim1/clevr/with_masks/images10k")
    ap.add_argument("--tfrecord", default="/work/USERS/tkim1/clevr/with_masks/clevr_with_masks_train.tfrecords")
    ap.add_argument("--encoder", default="/work/USERS/tkim1/checkpoints/input_encoder_k8_ks3_img128.pt")
    ap.add_argument("--out-dir", default="/work/USERS/tkim1/peer")
    ap.add_argument("--train-count", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    val_ids = list(range(1320, 1640))
    train_ids = list(range(0, 1000)) + list(range(1640, 1640 + args.train_count - 1000))
    assert len(train_ids) == args.train_count
    overlap = set(train_ids) & set(val_ids)
    assert not overlap, "LEAKAGE: %d ids in both" % len(overlap)
    assert max(train_ids) < 10000 and max(val_ids) < 10000, "id outside the extracted range"
    print("train %d ids: %d..%d then %d..%d" % (len(train_ids), train_ids[0], 999,
                                                train_ids[1000], train_ids[-1]))
    print("validation %d ids: %d..%d" % (len(val_ids), val_ids[0], val_ids[-1]))
    print("disjoint: yes")

    st = torch.load(args.encoder, map_location="cpu")
    st = st.get("state_dict", st)
    dev = torch.device(args.device)
    k, b = st["kernels"].to(dev), st["bias"].to(dev)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids)):
        g = build_gamma(ids, args.images, k, b, dev)
        torch.save(g, out / ("gamma_%s.pt" % name))
        print("  gamma_%s %s  mean %.6f std %.6f" % (name, tuple(g.shape), g.mean(), g.std()))

    # targets for validation only, under the same crop
    from training.prepare_clevr_with_masks import iter_records, _parse_example, IMAGE_SHAPE, MAX_ENTITIES
    want = set(val_ids)
    frames = {}
    for i, rec in enumerate(iter_records(args.tfrecord, max(val_ids) + 1)):
        if i not in want:
            continue
        f = _parse_example(rec)
        m = f["mask"].reshape(MAX_ENTITIES, IMAGE_SHAPE[0], IMAGE_SHAPE[1]) > 127
        px = torch.zeros(IMAGE_SHAPE[0], IMAGE_SHAPE[1], dtype=torch.int64)
        for e in range(1, MAX_ENTITIES):
            px[torch.as_tensor(m[e])] = e
        frames[i] = px
    pix = torch.stack([frames[i] for i in val_ids]).to(dev)
    h, w = pix.shape[1], pix.shape[2]
    pix = pix[:, (h - CROP) // 2:(h - CROP) // 2 + CROP, (w - CROP) // 2:(w - CROP) // 2 + CROP]
    pix = F.interpolate(pix.unsqueeze(1).float(), size=(SIZE, SIZE), mode="nearest").squeeze(1).long()
    lab = mask_cells(pix, GRID)
    fg = lab != 0
    objs = torch.tensor([lab[i][fg[i]].unique().numel() for i in range(lab.size(0))]).float()
    torch.save({"patch_labels": lab, "names": ["CLEVR_masks_%06d.png" % i for i in val_ids],
                "contract_version": "peer"}, out / "targets_val.pt")
    print("  targets_val %s  foreground %.3f  objects/image %.2f"
          % (tuple(lab.shape), fg.float().mean(), objs.mean()))
    print("  peer reports foreground prevalence 0.199-0.217 (PV2_0046)")

    json.dump({"version": "peer-1",
               "ranges": {"train": [0, args.train_count], "validation": [0, len(val_ids)],
                          "test": [0, 0]},
               "note": "two tensors; train and validation ids are disjoint by construction",
               "train_ids": [train_ids[0], 999, train_ids[1000], train_ids[-1]],
               "validation_ids": [val_ids[0], val_ids[-1]]},
              open(out / "manifest.json", "w"), indent=2)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
