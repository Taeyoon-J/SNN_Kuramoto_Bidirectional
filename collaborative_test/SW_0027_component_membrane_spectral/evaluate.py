"""Validation-only membrane spectral affinities; no model changes."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.training.evaluate_binding import kmeans


def correlation(history):
    centered = history - history.mean(dim=-1, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normalized @ normalized.T).clamp(-1, 1)


def spectral_labels(affinity, count):
    nodes = affinity.shape[0]
    affinity = affinity + 1e-6 * torch.eye(nodes, dtype=affinity.dtype)
    inverse_degree = affinity.sum(dim=1).clamp_min(1e-8).rsqrt()
    operator = inverse_degree[:, None] * affinity * inverse_degree[None, :]
    try:
        _, eigenvectors = torch.linalg.eigh(operator)
    except RuntimeError:
        _, eigenvectors = torch.linalg.eigh(operator.double())
        eigenvectors = eigenvectors.float()
    raw = kmeans(eigenvectors[:, -count:], count)
    background = int(torch.bincount(raw, minlength=count).argmax())
    labels = torch.zeros_like(raw, dtype=torch.int64)
    next_id = 1
    for group in range(count):
        if group != background:
            labels[raw == group] = next_id
            next_id += 1
    return labels.reshape(16, 16)


def score(predicted, truth):
    metrics = evaluate_patch_masks(predicted, truth)["mean"]
    return {
        "fg_ari": float(metrics["fg_ari"]),
        "foreground_iou": float(metrics["foreground_iou"]),
        "matched_object_iou": float(metrics["matched_object_iou"]),
        "predicted_groups_mean": float(np.mean([
            torch.unique(image[image > 0]).numel() for image in predicted
        ])),
    }


def affinities(aggregate, components):
    signed = torch.stack([correlation(component) for component in components])
    nonnegative = signed.clamp_min(0)
    absolute = signed.abs()
    return {
        "aggregate_absolute": correlation(aggregate).abs(),
        "component_absolute_mean": absolute.mean(dim=0),
        "component_absolute_product": absolute.prod(dim=0),
        "component_positive_mean": nonnegative.mean(dim=0),
        "component_positive_product": nonnegative.prod(dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cluster-count", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=[
        "aggregate_absolute", "component_absolute_mean",
        "component_absolute_product", "component_positive_mean",
        "component_positive_product",
    ])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if args.cluster_count < 2 or args.cluster_count > 256:
        raise ValueError("cluster count must be in [2, 256]")
    valid_modes = set(affinities(torch.rand(256, 3), torch.rand(4, 256, 3)))
    if not args.modes or set(args.modes) - valid_modes:
        raise ValueError(f"modes must be a nonempty subset of {sorted(valid_modes)}")

    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    predictions = {mode: [] for mode in args.modes}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, aggregate = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            components = model.last_component_out
            for image_index in range(aggregate.shape[0]):
                affinity_by_mode = affinities(
                    aggregate[image_index, :, 64:].cpu(),
                    components[image_index, :, :, 64:].cpu(),
                )
                for mode in args.modes:
                    predictions[mode].append(
                        spectral_labels(affinity_by_mode[mode], args.cluster_count)
                    )
    rows = [{"mode": mode, **score(torch.stack(labels), truth)}
            for mode, labels in predictions.items()]
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "source": "actual component membrane histories after 64-step settle",
        "cluster_count": args.cluster_count,
        "rows": rows,
    }
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_ari": max(rows, key=lambda row: row["fg_ari"]),
                      "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
