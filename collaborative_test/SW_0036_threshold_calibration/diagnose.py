"""Checkpoint validation diagnostic for membrane threshold dynamics."""
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
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0027_component_membrane_spectral"))
sys.path.insert(0, str(ROOT / "collaborative_test/SW_0028_spatial_membrane_spectral"))
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def auc(positive, negative):
    if not positive or not negative:
        return None
    values = np.asarray(positive + negative, dtype=np.float64)
    ranks = rankdata(values)
    n, m = len(positive), len(negative)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def macro_distance_auc(matrices, truth, squared_distance, strata):
    pairs = {distance: [[], []] for distance in strata}
    upper = torch.triu(torch.ones(256, 256, dtype=torch.bool), diagonal=1)
    for matrix, image_truth in zip(matrices, truth):
        labels = image_truth.flatten()
        foreground = (labels[:, None] != 0) & (labels[None, :] != 0)
        same = labels[:, None] == labels[None, :]
        for distance in strata:
            mask = upper & foreground & (squared_distance == distance)
            pairs[distance][0].extend(matrix[mask & same].tolist())
            pairs[distance][1].extend(matrix[mask & ~same].tolist())
    values = [auc(*pair) for pair in pairs.values()]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "gamma-path", "dataset-path", "output-path"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[.06, .25, .5, 1., 2., 3., 4.])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], default="shared")
    parser.add_argument(
        "--intervention",
        default="inference-only membrane vth sweep on frozen weights")
    parser.add_argument(
        "--warning",
        default=("Diagnostic only: checkpoint was trained at vth .06. GT is "
                 "used only for metrics and pair AUC, never prediction."))
    args = parser.parse_args()
    if not 1 <= args.count <= 320 or any(value <= 0 for value in args.thresholds):
        raise ValueError("Require 1-320 images and positive thresholds")
    ids = list(range(1320, 1320 + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256, args.dendritic_projection)
    kernel = spatial_kernel(1.5)
    grid = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16),
                                      indexing="ij"), -1).reshape(256, 2)
    squared_distance = ((grid[:, None] - grid[None, :]) ** 2).sum(-1)
    upper = torch.triu(torch.ones(256, 256, dtype=torch.bool), diagonal=1)
    strata = sorted(int(x) for x in torch.unique(squared_distance[upper & (squared_distance <= 9)]))
    rows = []
    for threshold in args.thresholds:
        model.membrane_layer.vth = float(threshold)
        captured = []

        def hook(module, inputs, output):
            captured.append((output[0] > module.v_th).detach().cpu())

        handle = model.membrane_layer.register_forward_hook(hook)
        membrane_histories, spike_histories, binary_histories = [], [], []
        try:
            with torch.no_grad():
                for start in range(0, args.count, args.batch_size):
                    captured.clear()
                    _, spikes, membrane = model(
                        gamma[start:start + args.batch_size].to(args.device),
                        return_core_out=True)
                    batch = membrane.shape[0]
                    folded = torch.stack(captured, -1).reshape(
                        batch, model.osc_dim, 256, 256)
                    membrane_histories.extend(membrane[:, :, 64:].cpu())
                    spike_histories.extend(spikes[:, :, 64:].cpu())
                    binary_histories.extend(folded[:, :, :, 64:])
        finally:
            handle.remove()
        membrane_affinity = [correlation(x).abs() for x in membrane_histories]
        spike_affinity = [correlation(x).abs() for x in spike_histories]
        binary_aggregate = [x.float().mean(0) for x in binary_histories]
        binary_affinity = [correlation(x).abs() for x in binary_aggregate]
        membrane_prediction = torch.stack([
            spectral_labels(affinity * kernel, 10) for affinity in membrane_affinity])
        spike_prediction = torch.stack([
            spectral_labels(affinity * kernel, 10) for affinity in spike_affinity])
        binary = torch.stack(binary_histories).float()
        binary_std = binary.std(-1, unbiased=False)
        row = {
            "threshold": float(threshold),
            "binary_event_rate": float(binary.mean()),
            "binary_constant_fraction": float((binary_std <= 1e-8).float().mean()),
            "binary_temporal_std_mean": float(binary_std.mean()),
            "membrane_distance_macro_auc": macro_distance_auc(
                membrane_affinity, truth, squared_distance, strata),
            "gated_spike_distance_macro_auc": macro_distance_auc(
                spike_affinity, truth, squared_distance, strata),
            "binary_distance_macro_auc": macro_distance_auc(
                binary_affinity, truth, squared_distance, strata),
            "membrane_spatial_readout": score(membrane_prediction, truth),
            "gated_spike_spatial_readout": score(spike_prediction, truth),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "intervention": args.intervention,
        "settle": 64, "steps": 256, "spatial_sigma": 1.5, "spectral_k": 10,
        "warning": args.warning,
        "rows": rows,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
