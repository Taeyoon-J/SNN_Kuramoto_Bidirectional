"""Validation classifier baseline: infer spectral cluster count without GT."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0013_dynamic_spike_slots"))
from evaluate_fixed_split import _core
from evaluate import dynamic_slots
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.training.evaluate_binding import kmeans


def correlation(history):
    centered = history - history.mean(dim=-1, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (normalized @ normalized.T).clamp(-1, 1)


def spatial_kernel(sigma):
    coords = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16),
                                        indexing="ij"), -1).reshape(256, 2).float()
    squared_distance = torch.cdist(coords, coords).square()
    return torch.exp(-squared_distance / (2 * sigma * sigma))


def fixed_spectral_labels(affinity, count):
    _, vectors = decomposition(affinity)
    return labels_from_vectors(vectors, count)


def decomposition(affinity):
    n = affinity.shape[0]
    affinity = affinity + 1e-6 * torch.eye(n, dtype=affinity.dtype)
    inv = affinity.sum(1).clamp_min(1e-8).rsqrt()
    operator = inv[:, None] * affinity * inv[None, :]
    try:
        values, vectors = torch.linalg.eigh(operator)
    except RuntimeError:
        values, vectors = torch.linalg.eigh(operator.double())
        values, vectors = values.float(), vectors.float()
    return values.flip(0), vectors.flip(1)


def choose_eigengap(values, minimum, maximum):
    maximum = min(maximum, len(values) - 1)
    candidates = torch.arange(minimum, maximum + 1)
    gaps = values[candidates - 1] - values[candidates]
    return int(candidates[gaps.argmax()])


def choose_threshold(values, threshold, minimum, maximum):
    return max(minimum, min(maximum, int((values >= threshold).sum())))


def labels_from_vectors(vectors, count):
    raw = kmeans(vectors[:, :count].contiguous(), count)
    background = int(torch.bincount(raw, minlength=count).argmax())
    labels = torch.zeros_like(raw, dtype=torch.int64)
    next_id = 1
    for group in range(count):
        if group != background:
            labels[raw == group] = next_id
            next_id += 1
    return labels.reshape(16, 16)


def summarize(name, predictions, truth, chosen_total):
    prediction = torch.stack(predictions)
    metrics = evaluate_patch_masks(prediction, truth)["mean"]
    pred_count = torch.tensor([torch.unique(x[x != 0]).numel() for x in prediction])
    true_count = torch.tensor([torch.unique(x[x != 0]).numel() for x in truth])
    errors = pred_count - true_count
    return {
        "name": name,
        "fg_ari": float(metrics["fg_ari"]),
        "foreground_iou": float(metrics["foreground_iou"]),
        "matched_object_iou": float(metrics["matched_object_iou"]),
        "predicted_groups_mean": float(pred_count.float().mean()),
        "true_groups_mean": float(true_count.float().mean()),
        "count_mae": float(errors.abs().float().mean()),
        "count_bias": float(errors.float().mean()),
        "exact_count_fraction": float((errors == 0).float().mean()),
        "within_one_count_fraction": float((errors.abs() <= 1).float().mean()),
        "predicted_foreground_fraction": float((prediction != 0).float().mean()),
        "selected_total_cluster_histogram": dict(sorted(Counter(chosen_total).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "dataset-path", "output-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], default="shared")
    parser.add_argument("--membrane-vth", type=float, default=.06)
    parser.add_argument("--min-clusters", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--max-clusters", type=int, nargs="+", default=[10, 12, 14])
    parser.add_argument("--eigen-thresholds", type=float, nargs="+", default=[.80, .85, .90, .95])
    parser.add_argument("--selected-only", action="store_true",
                        help="Run only pilot-selected adaptive configurations plus controls")
    args = parser.parse_args()
    if not 1 <= args.count <= 320:
        raise ValueError("count must be in [1, 320]")
    ids = list(range(1320, 1320 + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256, args.dendritic_projection)
    model.membrane_layer.vth = args.membrane_vth
    kernel = spatial_kernel(1.5)
    histories = {"membrane": [], "gated_spike": []}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, spikes, membrane = model(gamma[start:start + args.batch_size].to(args.device),
                                         return_core_out=True)
            histories["membrane"].extend(membrane[:, :, 64:].cpu())
            histories["gated_spike"].extend(spikes[:, :, 64:].cpu())
    affinity_sets = {"membrane": [], "membrane_spatial": [], "gated_spike_spatial": []}
    for membrane, spike in zip(histories["membrane"], histories["gated_spike"]):
        mem_affinity = correlation(membrane).abs()
        affinity_sets["membrane"].append(mem_affinity)
        affinity_sets["membrane_spatial"].append(mem_affinity * kernel)
        affinity_sets["gated_spike_spatial"].append(correlation(spike).abs() * kernel)
    decomposed = {source: [decomposition(a) for a in affinities]
                  for source, affinities in affinity_sets.items()}
    count_low = 3 if args.selected_only else min(args.min_clusters)
    count_high = 10 if args.selected_only else max(args.max_clusters)
    label_cache = {
        source: [{count: labels_from_vectors(vectors, count)
                  for count in range(count_low, count_high + 1)}
                 for _, vectors in items]
        for source, items in decomposed.items()
    }
    rows = []
    configurations = (
        [("gated_spike_spatial", "eigengap", None, 5, 10),
         ("membrane_spatial", "threshold", .8, 3, 10)]
        if args.selected_only else
        [(source, policy, threshold, low, high)
         for source in decomposed
         for low in args.min_clusters for high in args.max_clusters if low < high
         for policy, threshold in [("eigengap", None)] +
         [("threshold", value) for value in args.eigen_thresholds]]
    )
    for source, policy, threshold, low, high in configurations:
        items = decomposed[source]
        if policy == "eigengap":
            predictions, counts = [], []
            for image_index, (values, _) in enumerate(items):
                count = choose_eigengap(values, low, high)
                counts.append(count)
                predictions.append(label_cache[source][image_index][count])
            rows.append(summarize(f"{source}:eigengap:{low}-{high}", predictions, truth, counts))
        else:
            predictions, counts = [], []
            for image_index, (values, _) in enumerate(items):
                count = choose_threshold(values, threshold, low, high)
                counts.append(count)
                predictions.append(label_cache[source][image_index][count])
            rows.append(summarize(
                f"{source}:threshold:{threshold:g}:{low}-{high}", predictions, truth, counts))
    # Existing GT-free adaptive-slot count baseline on the same histories.
    slots = [torch.from_numpy(dynamic_slots(x.numpy(), .7, 6, image_id * 1009))
             for image_id, x in zip(ids, histories["membrane"])]
    rows.append(summarize("membrane:adaptive_slots:.7:6", slots, truth,
                          [int(torch.unique(x[x != 0]).numel()) + 1 for x in slots]))
    # Fixed-k controls show grouping quality at the established count, not count inference.
    for source in affinity_sets:
        predictions = [labels[10] for labels in label_cache[source]]
        rows.append(summarize(f"{source}:fixed_k10_control", predictions, truth,
                              [10] * len(predictions)))
    result = {
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "membrane_vth": args.membrane_vth,
        "prediction_contract": "All per-image counts inferred from signal eigenspectrum or adaptive slots; GT used only for validation scoring. Fixed-k rows are controls.",
        "background_rule": "largest spectral cluster; adaptive slots use largest slot and zero traces",
        "spatial_kernel": "sigma 1.5 when source name ends in _spatial",
        "rows": rows,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "best_adaptive_ari": max((r for r in rows if "fixed" not in r["name"]),
                                               key=lambda r: r["fg_ari"]),
                      "best_adaptive_count": max((r for r in rows if "fixed" not in r["name"]),
                                                 key=lambda r: r["exact_count_fraction"])}, indent=2))


if __name__ == "__main__":
    main()
