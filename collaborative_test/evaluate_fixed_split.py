"""Evaluate one checkpoint with the shared patch-level CLEVR contract.

The spike-synchrony readout was selected on validation only. Its k and
background rule are fixed here; ground truth is used only after prediction.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))

from snn_kuramoto_bidirectional.evaluation import (
    clevr_mask_patch,
    evaluate_patch_masks,
    spatial_components_to_patch_labels,
)
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.loss_function import signal_synchrony
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.evaluate_binding import spectral_cluster


def _core(device, checkpoint, steps, dendritic_projection="shared",
          geodesic_steps=0, geodesic_radius=1.5, geodesic_contrast=2.0,
          geodesic_temperature=0.5, geodesic_cap=16.0,
          graph_spatial_decay=0.55, kuramoto_backend="pairwise"):
    hp = S2NetHyperparameters(
        num_feature_maps=8,
        num_regions=256,
        sc=None,
        gamma_drive_mode="static",
        num_time_steps=steps,
        theta_init="gamma",
        gamma_phase_mode="standardize_tanh",
        osc_dim=4,
        freq_gain=2.0,
        graph_mode="learned",
        graph_top_k=32,
        graph_spatial_decay=graph_spatial_decay,
        kuramoto_backend=kuramoto_backend,
        geodesic_steps=geodesic_steps,
        geodesic_radius=geodesic_radius,
        geodesic_contrast=geodesic_contrast,
        geodesic_temperature=geodesic_temperature,
        geodesic_cap=geodesic_cap,
        k=256.0,
        low_n=-4.0,
        high_n=0.0,
        membrane_vth=0.06,
        membrane_low_m=-4.0,
        membrane_high_m=0.0,
        gate_mode="raw",
        spike_classify_method="spatial_components",
        spike_spatial_grid_size=(16, 16),
        spike_per_component=True,
        dendritic_projection=dendritic_projection,
    ).validate()
    model = S2NetCore(hp, device=device).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def _cluster_labels(affinity, k):
    raw = spectral_cluster(affinity.cpu(), k)
    background = int(torch.bincount(raw, minlength=k).argmax())
    pred = torch.zeros_like(raw, dtype=torch.int64)
    next_id = 1
    for group in range(k):
        if group != background:
            pred[raw == group] = next_id
            next_id += 1
    return pred.reshape(16, 16)


def main():
    import h5py

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--dendritic-projection", choices=["shared", "per_region"], default="shared")
    parser.add_argument("--geodesic-steps", type=int, default=0)
    parser.add_argument("--geodesic-radius", type=float, default=1.5)
    parser.add_argument("--geodesic-contrast", type=float, default=2.0)
    parser.add_argument("--geodesic-temperature", type=float, default=0.5)
    parser.add_argument("--geodesic-cap", type=float, default=16.0)
    parser.add_argument("--graph-spatial-decay", type=float, default=0.55)
    parser.add_argument("--kuramoto-backend", choices=["pairwise", "factorized"], default="pairwise")
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.start < 1000 and args.start + args.count > 0:
        raise ValueError("Evaluation must not include training IDs 0–999.")
    if args.settle >= args.steps:
        raise ValueError("settle must be less than steps.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    if max(indices) >= len(gamma):
        raise ValueError("Gamma blob does not contain requested image IDs.")
    gamma = gamma[indices].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][indices]), 8)["patch_labels"]

    model = _core(
        args.device, args.checkpoint, args.steps, args.dendritic_projection,
        args.geodesic_steps, args.geodesic_radius, args.geodesic_contrast,
        args.geodesic_temperature, args.geodesic_cap, args.graph_spatial_decay,
        args.kuramoto_backend,
    )
    baseline_masks, synchrony_masks = [], []
    spike_rates = []
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            groups, spikes, _ = model(batch, return_core_out=True)
            baseline_masks.append(spatial_components_to_patch_labels(groups, 16))
            affinity = signal_synchrony(spikes, settle=args.settle).cpu()
            synchrony_masks.extend(_cluster_labels(row, args.k) for row in affinity)
            spike_rates.extend(spikes.mean(dim=(1, 2)).cpu().tolist())
    predictions = {
        "spatial_components": torch.cat(baseline_masks),
        "spike_synchrony": torch.stack(synchrony_masks),
    }
    summary = {
        "checkpoint": args.checkpoint,
        "ids": [args.start, args.start + args.count - 1],
        "steps": args.steps,
        "settle": args.settle,
        "spike_synchrony_k": args.k,
        "spike_rate_mean": sum(spike_rates) / len(spike_rates),
        "metrics": {},
    }
    for name, prediction in predictions.items():
        scores = evaluate_patch_masks(prediction, truth)
        summary["metrics"][name] = {
            "mean": {key: value.item() for key, value in scores["mean"].items()},
            "valid_count": {key: int(value) for key, value in scores["valid_count"].items()},
            "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            "predicted_groups_mean": float(torch.tensor([
                torch.unique(image[image != 0]).numel() for image in prediction
            ], dtype=torch.float32).mean()),
            "per_image": {key: value.tolist() for key, value in scores["per_image"].items()},
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save({"ids": indices, "truth": truth, "predictions": predictions}, output / "patch_masks.pt")
    print(json.dumps({"metrics": {k: v["mean"] for k, v in summary["metrics"].items()},
                      "spike_rate_mean": summary["spike_rate_mean"]}, indent=2))


if __name__ == "__main__":
    main()
