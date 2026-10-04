"""Diagnostic-only: separate membrane information from spatial prior."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0028_spatial_membrane_spectral"))
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def rank_auc(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not positive.size or not negative.size:
        return None
    ranks = rankdata(np.concatenate([positive, negative]))
    return float((ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    kernel = spatial_kernel(1.5)
    grid = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16),
                                      indexing="ij"), dim=-1).reshape(256, 2).float()
    distance = torch.cdist(grid, grid)
    upper = torch.triu(torch.ones(256, 256, dtype=torch.bool), diagonal=1)
    nearby = upper & (distance <= 3)
    predictions = {key: [] for key in ("spatial_only", "membrane_spatial", "shuffled_membrane_spatial")}
    pairs = {key: {"same": [], "different": []}
             for key in ("membrane", "spatial_only", "membrane_spatial", "shuffled_membrane_spatial")}
    generator = torch.Generator().manual_seed(0)
    spatial_only = spectral_labels(kernel, 10)
    with torch.no_grad():
        image_index = 0
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            for history in membrane.cpu():
                affinity = correlation(history[:, 64:]).abs()
                permutation = torch.randperm(256, generator=generator)
                shuffled = affinity[permutation][:, permutation]
                values = {
                    "membrane": affinity,
                    "spatial_only": kernel,
                    "membrane_spatial": affinity * kernel,
                    "shuffled_membrane_spatial": shuffled * kernel,
                }
                predictions["spatial_only"].append(spatial_only)
                predictions["membrane_spatial"].append(spectral_labels(values["membrane_spatial"], 10))
                predictions["shuffled_membrane_spatial"].append(
                    spectral_labels(values["shuffled_membrane_spatial"], 10)
                )
                labels = truth[image_index].reshape(-1)
                foreground = (labels[:, None] != 0) & (labels[None, :] != 0)
                same = nearby & foreground & (labels[:, None] == labels[None, :])
                different = nearby & foreground & (labels[:, None] != labels[None, :])
                for key, matrix in values.items():
                    pairs[key]["same"].extend(matrix[same].tolist())
                    pairs[key]["different"].extend(matrix[different].tolist())
                image_index += 1
    rows = [{"mode": mode, **score(torch.stack(labels), truth)}
            for mode, labels in predictions.items()]
    diagnostics = {}
    for key, counts in pairs.items():
        same = counts["same"]
        different = counts["different"]
        diagnostics[key] = {
            "same_pairs": len(same), "different_pairs": len(different),
            "same_mean": float(np.mean(same)) if same else None,
            "different_mean": float(np.mean(different)) if different else None,
            "same_vs_different_auc": rank_auc(same, different),
        }
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual membrane histories after 64-step settle",
              "ablation": "spatial kernel only versus actual membrane affinity versus random position-permutation of membrane affinity",
              "diagnostic_only_ground_truth_use": "same/different foreground object pairs within 3 patches; never used for prediction",
              "rows": rows, "pair_diagnostics": diagnostics}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
