"""Read-only diagnostic of the peer geodesic graph on our trained graph weights.

This reproduces patch_v2:0dd2115's distance rule without changing core code or
training. Ground-truth labels are used only for graph-edge diagnostics.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def peer_geodesic_graph(module, gamma, steps=3, radius=1.5, contrast=2.0,
                        temperature=0.5, cap=16.0):
    """Reproduce the optional graph prior from peer commit 0dd2115."""
    z = F.normalize(module.projection(gamma.transpose(1, 2)), dim=-1)
    similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
    euclidean = module.grid_distance.unsqueeze(0)
    step = euclidean * (1.0 + contrast * (1.0 - similarity))
    step = torch.where(euclidean <= radius, step, torch.full_like(step, cap))
    distance = step
    for _ in range(steps):
        through = distance.unsqueeze(2) + distance.unsqueeze(1)
        relaxed = -temperature * torch.logsumexp(-through / temperature, dim=-1)
        distance = torch.minimum(distance, relaxed)
    distance = distance.clamp(max=cap)
    logits = similarity / module.log_temperature.exp().clamp_min(1e-3)
    logits = logits - F.softplus(module.spatial_rate) * distance
    _, indices = logits.topk(min(module.top_k, logits.size(-1)), dim=-1)
    values = logits.gather(-1, indices)
    weights = values.softmax(dim=-1) * module.log_coupling_gain.exp()
    adjacency = torch.zeros_like(logits).scatter_(-1, indices, weights)
    return (0.5 * (adjacency + adjacency.transpose(1, 2)))[0], distance[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use only held-out validation IDs 1320–1639.")

    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 64)
    graph_module = model.graph_generator
    categories = ("same_object", "different_objects", "foreground_background")
    totals = {
        name: {category: {"weight_sum": 0.0, "count": 0, "nonzero": 0}
               for category in categories}
        for name in ("euclidean", "peer_geodesic")
    }
    distances = {"min": math.inf, "max": -math.inf, "negative_count": 0, "count": 0}
    eye = torch.eye(256, dtype=torch.bool, device=args.device)
    with torch.no_grad():
        for index, image in enumerate(gamma):
            input_gamma = image.unsqueeze(0).to(args.device)
            graphs = {"euclidean": graph_module(input_gamma)[0]}
            graphs["peer_geodesic"], distance = peer_geodesic_graph(graph_module, input_gamma)
            distances["min"] = min(distances["min"], float(distance.min()))
            distances["max"] = max(distances["max"], float(distance.max()))
            distances["negative_count"] += int((distance < 0).sum())
            distances["count"] += distance.numel()

            labels = truth[index].flatten().to(args.device)
            foreground = labels != 0
            same = labels[:, None] == labels[None, :]
            masks = {
                "same_object": foreground[:, None] & foreground[None, :] & same & ~eye,
                "different_objects": foreground[:, None] & foreground[None, :] & ~same,
                "foreground_background": foreground[:, None] ^ foreground[None, :],
            }
            for name, graph in graphs.items():
                for category, mask in masks.items():
                    values = graph[mask]
                    entry = totals[name][category]
                    entry["weight_sum"] += float(values.sum())
                    entry["count"] += values.numel()
                    entry["nonzero"] += int((values > 0).sum())

    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "peer_source_commit": "patch_v2:0dd2115",
        "peer_geodesic_settings": {
            "steps": 3, "radius": 1.5, "contrast": 2.0,
            "temperature": 0.5, "cap": 16.0,
        },
        "graph_categories": {
            name: {
                category: {
                    "mean_weight": entry["weight_sum"] / entry["count"],
                    "nonzero_fraction": entry["nonzero"] / entry["count"],
                    "pair_count": entry["count"],
                }
                for category, entry in per_graph.items()
            }
            for name, per_graph in totals.items()
        },
        "geodesic_distance": {
            "min": distances["min"], "max": distances["max"],
            "negative_fraction": distances["negative_count"] / distances["count"],
        },
        "warning": "Ground truth labels used only for edge diagnostics, not training or prediction.",
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
