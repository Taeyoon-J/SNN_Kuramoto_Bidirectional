"""Compare predicted versus true foreground instance counts in saved patch masks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def foreground_counts(labels):
    return torch.tensor([
        torch.unique(image[image != 0]).numel() for image in labels
    ], dtype=torch.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-masks", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    data = torch.load(args.patch_masks, map_location="cpu", weights_only=True)
    truth = foreground_counts(data["truth"])
    prediction = foreground_counts(data["predictions"])
    if truth.shape != prediction.shape:
        raise ValueError("Truth and prediction batch sizes differ.")
    report = {
        "images": len(truth),
        "true_mean": float(truth.float().mean()),
        "predicted_mean": float(prediction.float().mean()),
        "exact_count_fraction": float((truth == prediction).float().mean()),
        "within_one_fraction": float(((truth - prediction).abs() <= 1).float().mean()),
        "true_range": [int(truth.min()), int(truth.max())],
        "predicted_range": [int(prediction.min()), int(prediction.max())],
        "true_histogram": dict(sorted(Counter(truth.tolist()).items())),
        "predicted_histogram": dict(sorted(Counter(prediction.tolist()).items())),
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
