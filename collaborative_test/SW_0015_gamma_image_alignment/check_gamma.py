"""Directly test gamma-to-HDF5 image ordering via patch-boundary fingerprints."""
import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F


def boundary_fingerprint(patches):
    # [B,C,16,16] -> [B,480] local contrast strength, independent of channels.
    horizontal = torch.linalg.vector_norm(patches[:, :, :, 1:] - patches[:, :, :, :-1], dim=1)
    vertical = torch.linalg.vector_norm(patches[:, :, 1:, :] - patches[:, :, :-1, :], dim=1)
    vector = torch.cat((horizontal.flatten(1), vertical.flatten(1)), dim=1)
    vector = vector - vector.mean(dim=1, keepdim=True)
    return F.normalize(vector, dim=1)


def compare(gamma_path, images, indices):
    gamma = torch.load(gamma_path, map_location="cpu", weights_only=True)[indices].float()
    if gamma.shape[1:] != (8, 256):
        raise ValueError(f"Unexpected gamma shape {tuple(gamma.shape)}")
    gamma = gamma.reshape(-1, 8, 16, 16)
    source = torch.from_numpy(images[indices]).float().permute(0, 3, 1, 2)
    if source.max() > 1:
        source = source / 255.0
    rgb = F.avg_pool2d(source, 8)
    gamma_edges = boundary_fingerprint(gamma)
    rgb_edges = boundary_fingerprint(rgb)
    similarity = gamma_edges @ rgb_edges.T
    diagonal = similarity.diag()
    rank = (similarity > diagonal[:, None]).sum(dim=1) + 1
    return {
        "gamma_path": gamma_path, "count": len(indices),
        "same_index_top1": int((rank == 1).sum()),
        "same_index_top5": int((rank <= 5).sum()),
        "same_index_median_rank": float(rank.float().median()),
        "same_index_mean_similarity": float(diagonal.mean()),
        "all_pairs_mean_similarity": float(similarity.mean()),
        "random_expected_top1": 1.0,
        "random_expected_median_rank": (len(indices) + 1) / 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gamma", required=True)
    parser.add_argument("--eval-gamma", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    with h5py.File(args.dataset_path, "r") as dataset:
        images = dataset["image"][:1640]
    result = {
        "method": "nearest HDF5 image by 16x16 horizontal and vertical patch-boundary contrast",
        "train": compare(args.train_gamma, images, list(range(128))),
        "validation": compare(args.eval_gamma, images, list(range(1320, 1448))),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
