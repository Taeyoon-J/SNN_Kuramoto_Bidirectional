"""Validation-only spike/membrane affinity mixture; no ground truth in masks."""
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


def correlation(traces):
    centered = traces.float() - traces.float().mean(dim=-1, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normalized @ normalized.transpose(-1, -2)).clamp(-1.0, 1.0)


def labels_from_affinity(affinity, threshold):
    adjacency = (affinity >= threshold).numpy().astype(np.uint8)
    count, groups = connected_components(csr_matrix(adjacency), directed=False)
    background = int(np.bincount(groups, minlength=count).argmax())
    labels = np.zeros(len(groups), dtype=np.int64)
    next_id = 1
    for group in range(count):
        if group != background:
            labels[groups == group] = next_id
            next_id += 1
    return torch.from_numpy(labels.reshape(16, 16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if args.settle >= args.steps:
        raise ValueError("settle must be smaller than steps")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, args.steps)
    affinities = []
    rates = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, spikes, _ = model(gamma[start:start + args.batch_size].to(args.device), return_core_out=True)
            rates.extend(spikes.mean(dim=(1, 2)).cpu().tolist())
            spike = correlation(model.last_component_spikes[:, :, :, args.settle:]).clamp_min(0).prod(dim=1)
            membrane = correlation(model.last_component_out[:, :, :, args.settle:]).clamp_min(0).prod(dim=1)
            affinities.extend(zip(spike.cpu(), membrane.cpu()))
    true_counts = torch.tensor([torch.unique(t[t != 0]).numel() for t in truth])
    rows = []
    for membrane_share in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixed = [(1.0 - membrane_share) * spike + membrane_share * membrane
                 for spike, membrane in affinities]
        for threshold in (0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95):
            predicted = torch.stack([labels_from_affinity(affinity, threshold) for affinity in mixed])
            scores = evaluate_patch_masks(predicted, truth)["mean"]
            counts = torch.tensor([torch.unique(p[p != 0]).numel() for p in predicted])
            rows.append({
                "membrane_share": membrane_share, "threshold": threshold,
                "fg_ari": float(scores["fg_ari"]),
                "foreground_iou": float(scores["foreground_iou"]),
                "matched_object_iou": float(scores["matched_object_iou"]),
                "predicted_groups_mean": float(counts.float().mean()),
                "exact_count_fraction": float((counts == true_counts).float().mean()),
            })
    result = {
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "source": "actual per-component spike and membrane histories after settle",
        "classifier": "linear mixture of nonnegative per-component centered-correlation products; connected components; largest group background",
        "mean_spike_rate": sum(rates) / len(rates),
        "true_groups_mean": float(true_counts.float().mean()),
        "rows": rows,
    }
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(path), "best_ari": max(rows, key=lambda row: row["fg_ari"])}, indent=2))


if __name__ == "__main__":
    main()
