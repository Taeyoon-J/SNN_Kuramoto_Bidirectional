"""Validation-only border-limited veto of spatial membrane groups."""
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
from evaluate import affinities, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def border_distance():
    coords = torch.arange(16)
    distance = torch.minimum(coords, 15 - coords)
    return torch.minimum(distance[:, None], distance[None, :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--border-widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if not args.border_widths or any(width < 1 or width > 8 for width in args.border_widths):
        raise ValueError("border widths must be 1 through 8")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    kernel = spatial_kernel(1.5)
    distance = border_distance()
    predictions = {"control": []}
    predictions.update({f"width_{width}": [] for width in args.border_widths})
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(
                gamma[start:start + args.batch_size].to(args.device), return_core_out=True
            )
            components = model.last_component_out
            for index in range(membrane.shape[0]):
                history = membrane[index, :, 64:].cpu()
                component_history = components[index, :, :, 64:].cpu()
                affinity = affinities(history, component_history)
                spatial = spectral_labels(affinity["aggregate_absolute"] * kernel, 10)
                nonspatial = spectral_labels(affinity["component_positive_mean"], 10)
                predictions["control"].append(spatial)
                for width in args.border_widths:
                    keep = (distance >= width) | (nonspatial != 0)
                    predictions[f"width_{width}"].append(spatial * keep)
    rows = []
    for mode, labels in predictions.items():
        predicted = torch.stack(labels)
        rows.append({"mode": mode, **score(predicted, truth),
                     "predicted_foreground_fraction": float((predicted != 0).float().mean())})
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual membrane histories after 64-step settle",
              "rule": "veto spatial foreground only within width patches of border where nonspatial component-positive classifier says background",
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
