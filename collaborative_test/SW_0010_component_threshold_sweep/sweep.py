"""Validation-only sweep of the peer component-product spike edge threshold."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0006_peer_spike_components"))
from evaluate_fixed_split import _core
from evaluate import _component_labels, _correlation
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("This threshold sweep must stay within validation IDs 1320–1639.")

    thresholds = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    predictions = {threshold: [] for threshold in thresholds}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            model(batch, return_core_out=True)
            traces = model.last_component_spikes[:, :, :, 64:]
            affinity = _correlation(traces).clamp_min(0.0).prod(dim=1)
            for matrix in affinity.cpu():
                for threshold in thresholds:
                    predictions[threshold].append(_component_labels(matrix, threshold))

    rows = []
    for threshold, masks in predictions.items():
        labels = torch.stack(masks)
        scores = evaluate_patch_masks(labels, truth)
        rows.append({
            "threshold": threshold,
            "fg_ari": float(scores["mean"]["fg_ari"]),
            "foreground_iou": float(scores["mean"]["foreground_iou"]),
            "matched_object_iou": float(scores["mean"]["matched_object_iou"]),
            "predicted_foreground_fraction": float((labels != 0).float().mean()),
            "predicted_groups_mean": float(torch.tensor([
                torch.unique(image[image != 0]).numel() for image in labels
            ], dtype=torch.float32).mean()),
        })
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "steps": 256,
        "settle": 64,
        "readout": "peer component_product, largest component background",
        "rows": rows,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
