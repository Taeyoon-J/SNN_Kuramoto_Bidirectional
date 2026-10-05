#!/usr/bin/env python3
"""Ground-truth-after-feature diagnostic; never a model score or training input."""
import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def summarize(gamma, labels):
    features = F.normalize(gamma.transpose(1, 2).float(), dim=-1)
    within, between, same_neighbor, cross_neighbor = [], [], [], []
    for x, y in zip(features, labels.flatten(1)):
        centroids = []
        for value in torch.unique(y):
            rows = x[y == value]
            if len(rows) == 0:
                continue
            center = F.normalize(rows.mean(dim=0), dim=0)
            within.extend((rows - center).norm(dim=-1).tolist())
            centroids.append(center)
        if len(centroids) > 1:
            centers = torch.stack(centroids)
            distances = torch.pdist(centers)
            between.extend(distances.tolist())
        grid_x, grid_y = x.reshape(16, 16, -1), y.reshape(16, 16)
        for a, b, la, lb in (
            (grid_x[:, :-1], grid_x[:, 1:], grid_y[:, :-1], grid_y[:, 1:]),
            (grid_x[:-1, :], grid_x[1:, :], grid_y[:-1, :], grid_y[1:, :]),
        ):
            cosine = (a * b).sum(dim=-1)
            same_neighbor.extend(cosine[la == lb].tolist())
            cross_neighbor.extend(cosine[la != lb].tolist())
    values = {
        "within_centroid_distance": sum(within) / len(within),
        "between_centroid_distance": sum(between) / len(between),
        "within_between_ratio": (sum(within) / len(within)) / (sum(between) / len(between)),
        "same_segment_neighbor_cosine": sum(same_neighbor) / len(same_neighbor),
        "cross_segment_neighbor_cosine": sum(cross_neighbor) / len(cross_neighbor),
    }
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError("non-finite feature diagnostic")
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-gamma", type=Path, required=True)
    p.add_argument("--dino-gamma", type=Path, required=True)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    native = torch.load(args.native_gamma, map_location="cpu", weights_only=True).float()
    dino = torch.load(args.dino_gamma, map_location="cpu", weights_only=True).float()
    if tuple(native.shape) != (320, 8, 256) or tuple(dino.shape) != (320, 8, 256):
        raise ValueError("expected aligned [320,8,256] gamma tensors")
    with h5py.File(args.hdf5, "r") as dataset:
        labels = clevr_mask_patch(torch.from_numpy(dataset["mask"][1320:1640]), 8)["patch_labels"]
    result = {
        "experiment": "SW0062 post-hoc feature-quality diagnostic",
        "ids": [1320, 1639],
        "count": 320,
        "ground_truth_role": "diagnostic only after label-free feature extraction; never training or prediction input",
        "native_autoencoder_gamma": summarize(native, labels),
        "dino_pca8_gamma": summarize(dino, labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
