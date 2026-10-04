"""Validation-only spatial-scale and cluster-count sweep for membrane readout."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0028_spatial_membrane_spectral"))
from evaluate_fixed_split import _core
from evaluate import correlation, score
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.training.evaluate_binding import kmeans


def spectral_labels_for_counts(affinity, counts):
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
    for count in counts:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.75, 1, 1.25, 1.5, 2])
    parser.add_argument("--cluster-counts", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if not args.sigmas or any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("all sigmas must be positive")
    if not args.cluster_counts or min(args.cluster_counts) < 2 or max(args.cluster_counts) > 256:
        raise ValueError("cluster counts must be in [2, 256]")

    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    kernels = {sigma: spatial_kernel(sigma) for sigma in args.sigmas}
    predictions = {(sigma, count): [] for sigma in args.sigmas for count in args.cluster_counts}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            for history in membrane.cpu():
                base_affinity = correlation(history[:, 64:]).abs()
                for sigma, kernel in kernels.items():
                    labels_by_count = spectral_labels_for_counts(
                        base_affinity * kernel, args.cluster_counts
                    )
                    for count, labels in labels_by_count.items():
                        predictions[(sigma, count)].append(labels)
    rows = [{"sigma": sigma, "clusters": count,
             **score(torch.stack(predictions[(sigma, count)]), truth)}
            for sigma in args.sigmas for count in args.cluster_counts]
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual membrane histories after 64-step settle",
              "spatial_kernel": "exp(-squared 2D grid distance / (2 sigma^2))",
              "rows": rows}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_ari": max(rows, key=lambda row: row["fg_ari"]),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
