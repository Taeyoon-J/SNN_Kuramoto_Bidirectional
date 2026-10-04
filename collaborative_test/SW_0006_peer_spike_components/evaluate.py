"""Validate the collaborator's connected-components spike readout on our split.

Adapted from patch_v2 commit 5a29422, spike_synchrony_components. The grouping
rule is the same: centered signed spike correlation, thresholded edges,
connected components, largest component as background, no oracle object count.
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


def _correlation(traces):
    traces = traces.float() - traces.float().mean(dim=-1, keepdim=True)
    traces = traces / traces.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (traces @ traces.transpose(-1, -2)).clamp(-1.0, 1.0)


def _component_labels(similarity, threshold):
    linked = (similarity >= threshold).numpy().astype(np.uint8)
    count, labels = connected_components(csr_matrix(linked), directed=False)
    background = int(np.bincount(labels, minlength=count).argmax())
    output = np.zeros_like(labels, dtype=np.int64)
    next_id = 1
    for group in range(count):
        if group != background:
            output[labels == group] = next_id
            next_id += 1
    return torch.from_numpy(output.reshape(16, 16))


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

    thresholds = (0.50, 0.70, 0.80, 0.90, 0.95)
    modes = ("aggregate", "component_product")
    ids = list(range(args.start, args.start + args.count))
    if args.start < 1000 or args.start + args.count > 10000:
        raise ValueError("Use held-out IDs from the same 10k gamma set.")
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, args.steps)
    predictions = {(mode, threshold): [] for mode in modes for threshold in thresholds}
    spike_rates = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            _, spikes, _ = model(batch, return_core_out=True)
            spike_rates.extend(spikes.mean(dim=(1, 2)).cpu().tolist())
            aggregate = _correlation(spikes[:, :, args.settle:])
            component_spikes = model.last_component_spikes[:, :, :, args.settle:]
            per_component = _correlation(component_spikes)
            product = per_component.clamp_min(0.0).prod(dim=1)
            for mode, affinities in (("aggregate", aggregate), ("component_product", product)):
                for affinity in affinities.cpu():
                    for threshold in thresholds:
                        predictions[(mode, threshold)].append(_component_labels(affinity, threshold))

    truth_counts = torch.tensor([
        torch.unique(image[image != 0]).numel() for image in truth
    ], dtype=torch.int64)
    rows = []
    for (mode, threshold), masks in predictions.items():
        labels = torch.stack(masks)
        scores = evaluate_patch_masks(labels, truth)
        predicted_counts = torch.tensor([
            torch.unique(image[image != 0]).numel() for image in labels
        ], dtype=torch.int64)
        rows.append({
            "mode": mode,
            "threshold": threshold,
            "fg_ari": float(scores["mean"]["fg_ari"]),
            "foreground_iou": float(scores["mean"]["foreground_iou"]),
            "matched_object_iou": float(scores["mean"]["matched_object_iou"]),
            "predicted_foreground_fraction": float((labels != 0).float().mean()),
            "predicted_groups_mean": float(predicted_counts.float().mean()),
            "exact_count_fraction": float((predicted_counts == truth_counts).float().mean()),
            "within_one_count_fraction": float(((predicted_counts - truth_counts).abs() <= 1).float().mean()),
        })
    result = {
        "source_commit": "patch_v2:5a29422",
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "spike_rate_mean": sum(spike_rates) / len(spike_rates),
        "true_groups_mean": float(truth_counts.float().mean()),
        "rows": rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
        print(metric, max(rows, key=lambda item: item[metric]))


if __name__ == "__main__":
    main()
