"""Validation-only spatial prior on membrane spectral readout."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral"))
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def spatial_kernel(sigma):
    coords = torch.stack(torch.meshgrid(
        torch.arange(16), torch.arange(16), indexing="ij"
    ), dim=-1).reshape(256, 2).float()
    squared_distance = torch.cdist(coords, coords).square()
    return torch.exp(-squared_distance / (2 * sigma * sigma))


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
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1.5, 3, 6, 12, 24])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if args.cluster_count < 2 or args.cluster_count > 256:
        raise ValueError("cluster count must be in [2, 256]")
    if not args.sigmas or any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("all spatial sigmas must be positive")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    kernels = {sigma: spatial_kernel(sigma) for sigma in args.sigmas}
    predictions = {"control": []}
    predictions.update({f"sigma_{sigma:g}": [] for sigma in args.sigmas})
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            for history in membrane.cpu():
                base_affinity = correlation(history[:, 64:]).abs()
                predictions["control"].append(spectral_labels(base_affinity, args.cluster_count))
                for sigma, kernel in kernels.items():
                    predictions[f"sigma_{sigma:g}"].append(
                        spectral_labels(base_affinity * kernel, args.cluster_count)
                    )
    rows = [{"mode": mode, **score(torch.stack(labels), truth)}
            for mode, labels in predictions.items()]
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual membrane histories after 64-step settle",
              "cluster_count": args.cluster_count,
              "spatial_kernel": "exp(-squared 2D grid distance / (2 sigma^2))",
              "rows": rows}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_ari": max(rows, key=lambda row: row["fg_ari"]),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
