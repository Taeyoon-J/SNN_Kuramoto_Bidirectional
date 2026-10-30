"""Build the uint8 [B,128,128,3] inputs SW0240's predict.py requires.

predict.py raises rather than resizing: "input must be nonempty uint8 RGB [B,128,128,3];
images are not resized automatically". So the upscale happens here, explicitly, and the
choice is recorded rather than buried.

NEAREST, not bilinear. The gamma encoder keys on local contrast -- PV2_0115 measured that
a patch's gamma deviation from its 5x5 neighbourhood separates true foreground at AUC
0.947 -- and bilinear invents intermediate colours exactly at object boundaries, which is
the signal that measurement is about. Nearest replicates pixels and invents nothing.

Row order is preserved, so row i is still "<dataset>_%06d.png" and still lines up with
split_manifest_<dataset>.json. That alignment is checked here, not assumed.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--src", default="/work/USERS/tkim1/mod")
ap.add_argument("--split", default="validation")
ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
ap.add_argument("--out", default="")
args = ap.parse_args()

man = json.load(open("%s/split_manifest_%s.json" % (args.src, args.dataset)))
lo, hi = man["ranges"][args.split]
if args.limit:
    hi = min(hi, lo + args.limit)
img = torch.load("%s/images_%s.pt" % (args.src, args.dataset), map_location="cpu")
sel = img[lo:hi]                                        # [N, H, W, 3] uint8
n, H, W, _ = sel.shape
print("%s %s[%d:%d]  source %dx%d  %d images" % (args.dataset, args.split, lo, hi, H, W, n))

x = sel.permute(0, 3, 1, 2).float()                     # [N, 3, H, W]
up = F.interpolate(x, size=(128, 128), mode="nearest").round().clamp(0, 255)
out = up.permute(0, 2, 3, 1).to(torch.uint8).numpy()    # [N, 128, 128, 3]
assert out.dtype == np.uint8 and out.shape[1:] == (128, 128, 3), out.shape

# nearest must not invent a colour: every output pixel is some input pixel
src_colours = {tuple(c) for c in sel.reshape(-1, 3).numpy()[:200000]}
new = [tuple(c) for c in out.reshape(-1, 3)[:200000] if tuple(c) not in src_colours]
print("  upscale %dx%d -> 128x128 nearest, factor %.3f" % (H, W, 128.0 / H))
print("  colours invented by the upscale: %d (nearest must be 0)" % len(new))
assert not new, "nearest invented colours -- wrong interpolation mode"

names = torch.load("%s/targets_%s_grid32.pt" % (args.src, args.dataset),
                   map_location="cpu")["names"][lo:hi]
assert names == man["splits"][args.split][: len(names)], "row order disagrees with the manifest"
print("  row order verified against the manifest: first %s, last %s" % (names[0], names[-1]))

path = args.out or "%s/images128_%s_%s.npy" % (args.src, args.dataset, args.split)
np.save(path, out)
print("  wrote %s  %s  %.1f MB" % (path, out.shape, out.nbytes / 2**20))
