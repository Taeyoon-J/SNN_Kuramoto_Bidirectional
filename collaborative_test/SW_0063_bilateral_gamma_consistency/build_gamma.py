#!/usr/bin/env python3
"""Edge-aware smoothing of native gamma without labels or masks."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bilateral(native, rgb, alpha=0.5, color_sigma=0.10, spatial_sigma=1.0):
    b, c, n = native.shape
    if (c, n) != (8, 256) or tuple(rgb.shape[1:]) != (3, 16, 16):
        raise ValueError("expected native [B,8,256] and RGB [B,3,16,16]")
    gamma_grid = native.reshape(b, c, 16, 16)
    gamma_neighbors = F.unfold(gamma_grid, kernel_size=3, padding=1).reshape(b, c, 9, n)
    rgb_neighbors = F.unfold(rgb, kernel_size=3, padding=1).reshape(b, 3, 9, n)
    valid = F.unfold(torch.ones(b, 1, 16, 16), kernel_size=3, padding=1).reshape(b, 1, 9, n)
    center = rgb.flatten(2).unsqueeze(2)
    color_d2 = (rgb_neighbors - center).square().sum(dim=1, keepdim=True)
    spatial_d2 = torch.tensor((2, 1, 2, 1, 0, 1, 2, 1, 2), dtype=native.dtype).view(1, 1, 9, 1)
    weights = torch.exp(-color_d2 / (2 * color_sigma ** 2) - spatial_d2 / (2 * spatial_sigma ** 2)) * valid
    smooth = (gamma_neighbors * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-8)
    result = (1.0 - alpha) * native + alpha * smooth
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite bilateral gamma")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--start", type=int, default=1320)
    p.add_argument("--count", type=int, default=320)
    args = p.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing overwrite")
    native = torch.load(args.native, map_location="cpu", weights_only=True).float()
    if tuple(native.shape) != (args.count, 8, 256):
        raise ValueError(f"native gamma mismatch: {tuple(native.shape)}")
    with h5py.File(args.hdf5, "r") as dataset:
        images = torch.from_numpy(dataset["image"][args.start:args.start + args.count]).permute(0, 3, 1, 2).float().div_(255.0)
    rgb = F.adaptive_avg_pool2d(images, (16, 16))
    result = bilateral(native, rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    manifest = {
        "experiment": "SW0063 label-free bilateral native-gamma consistency",
        "ids": [args.start, args.start + args.count - 1],
        "shape": list(result.shape),
        "parameters": {"radius": 1, "alpha": 0.5, "color_sigma": 0.10, "spatial_sigma": 1.0},
        "uses_masks_counts_or_labels": False,
        "native_sha256": sha256(args.native),
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
