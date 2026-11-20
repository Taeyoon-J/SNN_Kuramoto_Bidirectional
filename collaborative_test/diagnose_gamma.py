"""Which preprocessing produced wm10k_full_grid16.pt?

Global standardisation fixed the scale (std 1.000 against the reference 0.985)
but left a relative error of 0.568, so the features themselves differ. This
tries the plausible variants and reports the correlation of each against the
reference, allowing for a channel permutation.
"""
import argparse
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def load(names, root, size, crop, resample):
    out = []
    for n in names:
        im = Image.open(Path(root) / n).convert("RGB")
        if crop:
            w, h = im.size
            left, top = (w - crop) // 2, (h - crop) // 2
            im = im.crop((left, top, left + crop, top + crop))
        if size:
            im = im.resize((size, size), resample)
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
        out.append(t.view(im.size[1], im.size[0], 3).permute(2, 0, 1))
    return torch.stack(out).float()


def best_channel_match(a, b):
    """Correlation of a against b, maximised over a channel permutation."""
    C = a.size(1)
    af = (a - a.mean(dim=(0, 2), keepdim=True)).flatten(0, 0)
    bf = (b - b.mean(dim=(0, 2), keepdim=True))
    corr = torch.zeros(C, C)
    for i in range(C):
        x = a[:, i].flatten(); x = (x - x.mean()) / x.std().clamp_min(1e-9)
        for j in range(C):
            y = b[:, j].flatten(); y = (y - y.mean()) / y.std().clamp_min(1e-9)
            corr[i, j] = (x * y).mean()
    return corr.abs().max(dim=1).values.mean().item(), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--grid", type=int, default=16)
    args = ap.parse_args()

    state = torch.load(args.encoder, map_location="cpu")
    state = state.get("state_dict", state)
    kernels, bias = state["kernels"], state["bias"]

    ref = torch.load(args.reference, map_location="cpu").float()[: args.n]
    ref = ref.view(args.n, 8, args.grid, args.grid)

    names = sorted(p.name for p in Path(args.image_dir).iterdir()
                   if p.suffix.lower() == ".png")[: args.n]

    resamples = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
                 "nearest": Image.NEAREST, "lanczos": Image.LANCZOS}
    rows = []
    for size, crop, rname, pad, scale in product(
            [128, 0], [0, 192], ["bilinear", "bicubic", "nearest"], [0, 1], ["01", "pm1"]):
        if size == 0 and crop == 0 and rname != "bilinear":
            continue
        try:
            img = load(names, args.image_dir, size, crop, resamples[rname]) / 255.0
        except Exception as exc:
            continue
        if scale == "pm1":
            img = img * 2.0 - 1.0
        with torch.no_grad():
            feat = F.conv2d(img, kernels, bias=bias, padding=pad)
            pooled = F.adaptive_avg_pool2d(feat, (args.grid, args.grid))
        score, _ = best_channel_match(pooled, ref)
        rows.append((score, "size=%-4s crop=%-4s %-9s pad=%d scale=%-4s"
                     % (size or "native", crop or "-", rname, pad, scale)))

    rows.sort(reverse=True)
    print("%-8s %s" % ("corr", "variant"))
    for s, label in rows[:12]:
        print("%-8.4f %s" % (s, label))


if __name__ == "__main__":
    main()
