#!/usr/bin/env python3
"""Build label-free 8-channel 16x16 patch features from HDF5 RGB images."""
import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F


def build(images):
    x = torch.as_tensor(images).permute(0, 3, 1, 2).float().div_(255.0)
    rgb = F.adaptive_avg_pool2d(x, (16, 16))
    lum_full = 0.2126 * x[:, :1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
    luminance = F.adaptive_avg_pool2d(lum_full, (16, 16))
    chroma = F.adaptive_avg_pool2d(x.amax(dim=1, keepdim=True) - x.amin(dim=1, keepdim=True), (16, 16))
    dx = F.pad((lum_full[:, :, :, 1:] - lum_full[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((lum_full[:, :, 1:, :] - lum_full[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    gx = F.adaptive_avg_pool2d(dx, (16, 16))
    gy = F.adaptive_avg_pool2d(dy, (16, 16))
    magnitude = (gx.square() + gy.square()).sqrt()
    features = torch.cat((rgb, luminance, chroma, gx, gy, magnitude), dim=1)
    if tuple(features.shape[1:]) != (8, 16, 16):
        raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise RuntimeError("non-finite RGB patch features")
    return features.flatten(2).contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--start", type=int, default=1320)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite gamma or manifest")
    with h5py.File(args.hdf5, "r") as source:
        end = args.start + args.count
        if args.start < 0 or end > len(source["image"]):
            raise ValueError("requested image range is outside HDF5")
        gamma = build(source["image"][args.start:end])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gamma, args.output)
    manifest = {
        "experiment": "SW0061 label-free RGB patch features",
        "source_hdf5": str(args.hdf5),
        "source_key": "image",
        "ids": [args.start, args.start + args.count - 1],
        "count": args.count,
        "shape": list(gamma.shape),
        "channels": ["R", "G", "B", "luminance", "chroma", "abs_dx", "abs_dy", "gradient_magnitude"],
        "grid": [16, 16],
        "uses_masks_or_labels": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
