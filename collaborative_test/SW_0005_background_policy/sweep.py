"""Screen a prediction-only background policy on the fixed validation split.

The saved spike-synchrony partition is unchanged. Additional clusters with a
large perimeter fraction can be designated as background. Truth is only used
for post-hoc validation scores, never to choose a cluster per image.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from snn_kuramoto_bidirectional.evaluation import evaluate_patch_masks


def relabel_with_border_rule(labels, threshold, min_border_patches):
    border = torch.zeros((16, 16), dtype=torch.bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    output = labels.clone()
    for image in output:
        for group in image.unique().tolist():
            if group == 0:
                continue
            region = image == group
            border_count = int((region & border).sum())
            if border_count >= min_border_patches and border_count / int(region.sum()) >= threshold:
                image[region] = 0
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    blob = torch.load(args.predictions, map_location="cpu", weights_only=True)
    truth = blob["truth"]
    original = blob["predictions"]["spike_synchrony"]
    rows = []
    for minimum in (1, 2, 3, 4):
        for threshold in (0.05, 0.10, 0.20, 0.30, 0.40):
            prediction = relabel_with_border_rule(original, threshold, minimum)
            scores = evaluate_patch_masks(prediction, truth)
            rows.append({
                "min_border_patches": minimum,
                "threshold": threshold,
                "fg_ari": float(scores["mean"]["fg_ari"]),
                "foreground_iou": float(scores["mean"]["foreground_iou"]),
                "matched_object_iou": float(scores["mean"]["matched_object_iou"]),
                "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            })
    baseline = evaluate_patch_masks(original, truth)
    result = {
        "baseline": {key: float(value) for key, value in baseline["mean"].items()},
        "rows": rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    for metric in ("fg_ari", "foreground_iou", "matched_object_iou"):
        print(metric, max(rows, key=lambda item: item[metric]))


if __name__ == "__main__":
    main()
