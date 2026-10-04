"""Score Slot Attention pixel predictions with the shared patch evaluation API."""
import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from protocol import validate_slice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as source:
        image_dataset = source["image"]
        start, end = validate_slice(args.start, args.count, image_dataset.shape[0])
        if start != args.start or end - start != args.count:
            raise ValueError("Invalid requested validation slice.")
        target_pixel_masks = torch.from_numpy(source["mask"][start:end])
    saved = np.load(args.predictions)
    expected_ids = np.arange(start, end, dtype=np.int64)
    if not np.array_equal(saved["image_ids"], expected_ids):
        raise ValueError("Prediction image_ids do not exactly match requested HDF5 rows.")
    pixel_predictions = torch.from_numpy(saved["labels"].astype(np.int64))
    if tuple(pixel_predictions.shape) != (args.count, 128, 128):
        raise ValueError("Expected saved labels with shape [count, 128, 128].")
    target = clevr_mask_patch(target_pixel_masks, 8)
    prediction = clevr_mask_patch(pixel_predictions, 8)["patch_labels"]
    scores = evaluate_patch_masks(prediction, target["patch_labels"])
    summary = {
        "protocol": json.loads(Path(args.protocol).read_text(encoding="utf-8")),
        "evaluation": "validation only; single official transfer checkpoint, not a three-seed mean",
        "patch_size": 8,
        "patch_grid": [16, 16],
        "ground_truth_used_for_prediction": False,
        "scores": {
            group: {key: (None if torch.isnan(value) else float(value.item()))
                    for key, value in values.items()}
            for group, values in scores.items() if group != "per_image"
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    metric_names = list(scores["per_image"])
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", *metric_names, "pred_objects", "target_patch_objects"])
        for index, image_id in enumerate(expected_ids):
            pred_count = int(torch.unique(prediction[index][prediction[index] != 0]).numel())
            target_count = int(torch.unique(target["patch_labels"][index][
                target["patch_labels"][index] != 0]).numel())
            writer.writerow([
                int(image_id), *[float(scores["per_image"][name][index]) for name in metric_names],
                pred_count, target_count,
            ])
    torch.save({
        "prediction": prediction,
        "target": target["patch_labels"],
        "image_ids": expected_ids.tolist(),
        "patch_size": 8,
    }, output_dir / "patch_masks.pt")
    (output_dir / "SCORING_COMPLETED").write_text("scored\n", encoding="utf-8")
    print(json.dumps(summary["scores"], indent=2), flush=True)


if __name__ == "__main__":
    main()
