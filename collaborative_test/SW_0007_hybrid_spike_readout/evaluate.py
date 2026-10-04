"""Hybrid spike readout: peer foreground detection, then spike-pattern grouping.

The largest signed-correlation connected component is background. Only the
remaining patches are clustered by absolute spike-trace synchrony. Ground
truth is used exclusively by the final common patch evaluator.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.training.evaluate_binding import spectral_cluster


def signed_correlation(spikes):
    centered = spikes.float() - spikes.float().mean(dim=-1, keepdim=True)
    centered = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (centered @ centered.transpose(-1, -2)).clamp(-1.0, 1.0)


def foreground_from_components(correlation, threshold):
    adjacency = (correlation >= threshold).numpy().astype(np.uint8)
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    background = np.bincount(labels, minlength=count).argmax()
    return torch.from_numpy(np.flatnonzero(labels != background))


def grouped_foreground(correlation, foreground, k):
    output = torch.zeros(256, dtype=torch.int64)
    if foreground.numel() == 0:
        return output.reshape(16, 16)
    count = min(k, foreground.numel())
    if count == 1:
        output[foreground] = 1
    else:
        affinity = correlation.abs()[foreground][:, foreground].clamp(0.0, 1.0)
        output[foreground] = spectral_cluster(affinity, count).long() + 1
    return output.reshape(16, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("This validation sweep must use IDs within 1320–1639.")
    if not 0 <= args.settle < args.steps:
        raise ValueError("settle must be less than steps.")

    thresholds = (0.80, 0.95)
    counts = (3, 5, 7)
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, args.steps)
    predictions = {(threshold, k): [] for threshold in thresholds for k in counts}
    foreground_sizes = {threshold: [] for threshold in thresholds}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, spikes, _ = model(gamma[start:start + args.batch_size].to(args.device), return_core_out=True)
            for correlation in signed_correlation(spikes[:, :, args.settle:]).cpu():
                for threshold in thresholds:
                    foreground = foreground_from_components(correlation, threshold)
                    foreground_sizes[threshold].append(int(foreground.numel()))
                    for k in counts:
                        predictions[(threshold, k)].append(grouped_foreground(correlation, foreground, k))

    rows = []
    for (threshold, k), masks in predictions.items():
        labels = torch.stack(masks)
        scores = evaluate_patch_masks(labels, truth)
        rows.append({
            "threshold": threshold,
            "k": k,
            "fg_ari": float(scores["mean"]["fg_ari"]),
            "foreground_iou": float(scores["mean"]["foreground_iou"]),
            "matched_object_iou": float(scores["mean"]["matched_object_iou"]),
            "predicted_foreground_fraction": float((labels != 0).float().mean()),
            "predicted_groups_mean": float(torch.tensor([
                torch.unique(image[image != 0]).numel() for image in labels
            ], dtype=torch.float32).mean()),
        })
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "steps": args.steps,
        "settle": args.settle,
        "foreground_sizes_mean": {
            str(threshold): sum(values) / len(values)
            for threshold, values in foreground_sizes.items()
        },
        "rows": rows,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
