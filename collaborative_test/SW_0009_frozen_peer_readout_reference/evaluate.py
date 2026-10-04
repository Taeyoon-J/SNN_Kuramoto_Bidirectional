"""Evaluate the validation-selected SW_0008 readout on reference test IDs.

The classifier is frozen: per-component signed spike correlations, product of
positive similarities, connected components at threshold 0.50, and largest
component as background. No ground truth enters prediction.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks


def correlation(traces):
    traces = traces.float() - traces.float().mean(dim=-1, keepdim=True)
    traces = traces / traces.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (traces @ traces.transpose(-1, -2)).clamp(-1.0, 1.0)


def classify(component_spikes):
    per_component = correlation(component_spikes)
    similarity = per_component.clamp_min(0.0).prod(dim=1)
    labels = []
    for affinity in similarity.cpu():
        adjacency = (affinity >= 0.50).numpy().astype(np.uint8)
        count, groups = connected_components(csr_matrix(adjacency), directed=False)
        background = np.bincount(groups, minlength=count).argmax()
        predicted = np.zeros_like(groups, dtype=np.int64)
        foreground_groups = [group for group in range(count) if group != background]
        for new_id, group in enumerate(foreground_groups, start=1):
            predicted[groups == group] = new_id
        labels.append(torch.from_numpy(predicted.reshape(16, 16)))
    return torch.stack(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=1000)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1000 or args.start + args.count > 1320:
        raise ValueError("Only reference test IDs 1000–1319 are permitted here.")

    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    predictions, spike_rates = [], []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            _, spikes, _ = model(batch, return_core_out=True)
            spike_rates.extend(spikes.mean(dim=(1, 2)).cpu().tolist())
            predictions.append(classify(model.last_component_spikes[:, :, :, 64:]))
    labels = torch.cat(predictions)
    scores = evaluate_patch_masks(labels, truth)
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "selection": "SW_0008 validation: component_product, threshold 0.50; frozen before reference test",
        "steps": 256,
        "settle": 64,
        "spike_rate_mean": sum(spike_rates) / len(spike_rates),
        "predicted_foreground_fraction": float((labels != 0).float().mean()),
        "predicted_groups_mean": float(torch.tensor([
            torch.unique(image[image != 0]).numel() for image in labels
        ], dtype=torch.float32).mean()),
        "mean": {key: float(value) for key, value in scores["mean"].items()},
        "valid_count": {key: int(value) for key, value in scores["valid_count"].items()},
        "per_image": {key: value.tolist() for key, value in scores["per_image"].items()},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    torch.save({"ids": ids, "truth": truth, "predictions": labels}, output / "patch_masks.pt")
    print(json.dumps({"mean": result["mean"], "spike_rate_mean": result["spike_rate_mean"]}, indent=2))


if __name__ == "__main__":
    main()
