"""Compare membrane spectral grouping to adaptive slots on fixed validation."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0013_dynamic_spike_slots"))
from evaluate_fixed_split import _core
from evaluate import dynamic_slots
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.training.evaluate_binding import kmeans


def correlation(history):
    centered = history - history.mean(dim=-1, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normalized @ normalized.T).clamp(-1, 1)


def spectral_labels(affinity, cluster_counts):
    nodes = affinity.shape[0]
    affinity = affinity + 1e-6 * torch.eye(nodes, dtype=affinity.dtype)
    inverse_degree = affinity.sum(dim=1).clamp_min(1e-8).rsqrt()
    operator = inverse_degree[:, None] * affinity * inverse_degree[None, :]
    try:
        _, eigenvectors = torch.linalg.eigh(operator)
    except RuntimeError:
        _, eigenvectors = torch.linalg.eigh(operator.double())
        eigenvectors = eigenvectors.float()
    output = {}
    for count in cluster_counts:
        raw = kmeans(eigenvectors[:, -count:], count)
        background = int(torch.bincount(raw, minlength=count).argmax())
        labels = torch.zeros_like(raw, dtype=torch.int64)
        next_id = 1
        for group in range(count):
            if group != background:
                labels[raw == group] = next_id
                next_id += 1
        output[count] = labels.reshape(16, 16)
    return output


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cluster-counts", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if not args.cluster_counts or min(args.cluster_counts) < 2 or max(args.cluster_counts) > 256:
        raise ValueError("Cluster counts must be integers from 2 through 256")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    histories = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(gamma[start:start + args.batch_size].to(args.device), return_core_out=True)
            histories.extend(membrane[:, :, 64:].cpu())
    predictions = {(mode, count): [] for mode in ("absolute", "positive")
                   for count in args.cluster_counts}
    for history in histories:
        signed = correlation(history)
        for mode, affinity in (("absolute", signed.abs()), ("positive", signed.clamp_min(0))):
            for count, labels in spectral_labels(affinity, args.cluster_counts).items():
                predictions[(mode, count)].append(labels)
    rows = []
    for (mode, count), labels in predictions.items():
        rows.append({"mode": mode, "clusters": count,
                     **score(torch.stack(labels), truth)})
    baseline = torch.from_numpy(np.stack([
        dynamic_slots(history.numpy(), 0.7, 6, image_id * 1009)
        for image_id, history in zip(ids, histories)
    ]))
    result = {
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "source": "actual membrane histories, centered temporal cosine",
        "baseline_adaptive_slots": score(baseline, truth),
        "rows": rows,
    }
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"baseline": result["baseline_adaptive_slots"],
                      "best_ari": max(rows, key=lambda row: row["fg_ari"])}, indent=2))


if __name__ == "__main__":
    main()
