"""Dump the benchmark images as PNGs so the gamma pipeline can read them.

The gamma recipe is two unsupervised stages: train_gamma_initializer fits 8 3x3
kernels as an autoencoder (224 parameters, MSE reconstruction, no labels), then
build_gamma_grid applies them and pools to the patch grid. Both take an image
DIRECTORY, and the benchmark images are currently .pt tensors.

Names follow the CLEVR convention so the manifests line up by index.
"""
import sys, argparse
from pathlib import Path
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--src", default="/work/USERS/tkim1/mod")
args = ap.parse_args()

img = torch.load("%s/images_%s.pt" % (args.src, args.dataset), map_location="cpu")
out = Path("%s/images_%s" % (args.src, args.dataset))
out.mkdir(parents=True, exist_ok=True)
print("%s: %s -> %s" % (args.dataset, tuple(img.shape), out), flush=True)
for i in range(img.shape[0]):
    Image.fromarray(img[i].numpy().astype("uint8")).save(out / ("%s_%06d.png" % (args.dataset, i)))
    if (i + 1) % 2000 == 0:
        print("  %d" % (i + 1), flush=True)
print("wrote %d PNGs" % img.shape[0])
