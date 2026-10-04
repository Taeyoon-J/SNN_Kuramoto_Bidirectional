"""Validation-only border-aware background selection after membrane grouping."""
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
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def border_background(labels, threshold):
    result = labels.clone()
    border = torch.zeros_like(labels, dtype=torch.bool)
    border[0] = True
    border[-1] = True
    border[:, 0] = True
    border[:, -1] = True
    for group in torch.unique(labels).tolist():
        if group == 0:
            continue
        members = labels == group
        fraction = (members & border).sum().float() / members.sum().clamp_min(1)
        if fraction > 0 and fraction >= threshold:
            result[members] = 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0, 0.05, 0.1, 0.2, 0.4, 0.6])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if not args.thresholds or any(t < 0 or t > 1 for t in args.thresholds):
        raise ValueError("thresholds must be in [0, 1]")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    kernel = spatial_kernel(1.5)
    predictions = {"control": []}
    predictions.update({f"border_{t:g}": [] for t in args.thresholds})
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            for history in membrane.cpu():
                base = spectral_labels(correlation(history[:, 64:]).abs() * kernel, 10)
                predictions["control"].append(base)
                for threshold in args.thresholds:
                    predictions[f"border_{threshold:g}"].append(
                        border_background(base, threshold)
                    )
    rows = []
    for mode, labels in predictions.items():
        predicted = torch.stack(labels)
        rows.append({"mode": mode, **score(predicted, truth),
                     "predicted_foreground_fraction": float((predicted != 0).float().mean())})
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual membrane histories after 64-step settle",
              "readout": "spatial Gaussian sigma1.5, spectral k10; largest background",
              "rule": "add a cluster to background if border-patch fraction >= threshold and > 0",
              "true_foreground_fraction": float((truth != 0).float().mean()),
              "rows": rows}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"best_ari": max(rows, key=lambda row: row["fg_ari"]),
                      "best_foreground_iou": max(rows, key=lambda row: row["foreground_iou"]),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
