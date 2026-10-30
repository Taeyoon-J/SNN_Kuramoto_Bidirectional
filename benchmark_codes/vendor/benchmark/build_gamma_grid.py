"""Build the gamma sequence on an arbitrary patch grid.

The recipe is the one GammaInitializer uses: resize to the encoder's image
size, one valid 3x3 convolution with the trained 8 kernels, then
adaptive_avg_pool2d to the grid. Nothing here is new -- the point is to run it
at grid 32 as well as 16, and the grid-16 output is compared against the
existing wm10k_full_grid16.pt so the recipe is proven before grid 32 is used.
"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--grid", type=int, required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="", help="existing .pt the result must match")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", action="store_true",
                    help="apply gamma_ordering.order_gammas, which permutes the 8 "
                         "channels PER IMAGE to minimise the ordering loss. This is "
                         "the step that made the replayed grid-16 features correlate "
                         "at 0.9999 but mismatch element-wise.")
    ap.add_argument("--standardize", default="global",
                    choices=["none", "global", "per_channel", "per_image"],
                    help="the step that was missing when the grid-16 recipe was "
                         "first replayed: the saved sequences have unit variance "
                         "while raw pooled features have std 0.146")
    args = ap.parse_args()

    state = torch.load(args.encoder, map_location="cpu")
    state = state.get("state_dict", state)
    kernels, bias = state["kernels"], state["bias"]
    device = torch.device(args.device)
    kernels, bias = kernels.to(device), bias.to(device)

    names = sorted(p.name for p in Path(args.image_dir).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:args.limit]
    if len(names) < args.limit:
        print("WARNING: only %d images in %s" % (len(names), args.image_dir))

    out = []
    for start in range(0, len(names), args.batch_size):
        chunk = names[start:start + args.batch_size]
        batch = []
        for name in chunk:
            img = Image.open(Path(args.image_dir) / name).convert("RGB")
            img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
            batch.append(torch.frombuffer(img.tobytes(), dtype=torch.uint8)
                         .view(args.image_size, args.image_size, 3).permute(2, 0, 1))
        images = torch.stack(batch).to(device).float() / 255.0
        with torch.no_grad():
            feats = F.conv2d(images, kernels, bias=bias, stride=1, padding=0)
            pooled = F.adaptive_avg_pool2d(feats, (args.grid, args.grid))
        out.append(pooled.flatten(2).cpu())
        if (start // args.batch_size) % 10 == 0:
            print("  %d / %d" % (start + len(chunk), len(names)), flush=True)

    gamma = torch.cat(out)
    if args.standardize == "global":
        gamma = (gamma - gamma.mean()) / gamma.std()
    elif args.standardize == "per_channel":
        gamma = (gamma - gamma.mean(dim=(0, 2), keepdim=True)) / gamma.std(dim=(0, 2), keepdim=True)
    elif args.standardize == "per_image":
        dims = (1, 2)
        gamma = (gamma - gamma.mean(dim=dims, keepdim=True)) / gamma.std(dim=dims, keepdim=True)
    if args.order:
        sys.path.insert(0, "/export_home/tkim1")
        from snn_kuramoto_bidirectional.gamma_ordering import order_gammas
        gamma, _, _ = order_gammas(gamma)
    print("built %s  mean %.6f  std %.6f" % (tuple(gamma.shape), gamma.mean(), gamma.std()))

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu").float()
        n = min(ref.size(0), gamma.size(0))
        if ref.shape[1:] != gamma.shape[1:]:
            print("SHAPE MISMATCH ref %s vs built %s" % (tuple(ref.shape), tuple(gamma.shape)))
            sys.exit(1)
        delta = (ref[:n] - gamma[:n]).abs()
        rel = delta.max() / ref[:n].abs().max()
        print("compare against %s" % args.compare)
        print("  ref  mean %.6f std %.6f" % (ref[:n].mean(), ref[:n].std()))
        print("  max abs delta %.3e   relative %.3e" % (delta.max(), rel))
        print("  RECIPE REPRODUCES" if rel < 1e-3 else "  RECIPE DOES NOT REPRODUCE")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gamma, args.out)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
