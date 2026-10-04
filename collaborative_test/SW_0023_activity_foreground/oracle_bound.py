"""Diagnostic upper bound: score frozen membrane slots with oracle FG only."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0013_dynamic_spike_slots"))
from evaluate_fixed_split import _core
from evaluate import dynamic_slots
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks


def summary(predicted, truth):
    mean = evaluate_patch_masks(predicted, truth)["mean"]
    return {"fg_ari": float(mean["fg_ari"]),
            "foreground_iou": float(mean["foreground_iou"]),
            "matched_object_iou": float(mean["matched_object_iou"]),
            "predicted_foreground_fraction": float((predicted != 0).float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    ids = list(range(1320, 1640))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    histories = []
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            _, _, membrane = model(gamma[start:start + 8].to(args.device), return_core_out=True)
            histories.extend(membrane[:, :, 64:].cpu().numpy())
    no_background = torch.from_numpy(np.stack([
        dynamic_slots(trace, 0.7, 6, image_id * 1009, assign_background=False)
        for image_id, trace in zip(ids, histories)
    ]))
    largest_background = torch.from_numpy(np.stack([
        dynamic_slots(trace, 0.7, 6, image_id * 1009, assign_background=True)
        for image_id, trace in zip(ids, histories)
    ]))
    oracle_foreground = truth != 0
    result = {
        "checkpoint": args.checkpoint,
        "ids": [1320, 1639],
        "warning": "ORACLE DIAGNOSTIC ONLY: GT foreground is applied after prediction; these scores are NOT deployable model performance.",
        "largest_slot_background": summary(largest_background, truth),
        "largest_slot_plus_oracle_foreground": summary(largest_background * oracle_foreground, truth),
        "no_slot_background_plus_oracle_foreground": summary(no_background * oracle_foreground, truth),
    }
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
