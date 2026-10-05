#!/usr/bin/env python3
"""Evaluate one frozen SW0072 core on the peer's fixed validation300 contract."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import (
    evaluate_patch_masks,
    spatial_components_to_patch_labels,
)
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

START = 6000
COUNT = 300
STEPS = 1024
SETTLE = 512
THRESHOLD = .35


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--gamma", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["ranges"]["validation"] != [6000, 7000]:
        raise ValueError("unexpected peer validation range")
    gamma_blob = torch.load(args.gamma, map_location="cpu", weights_only=True)
    target_blob = torch.load(args.targets, map_location="cpu", weights_only=True)
    if tuple(gamma_blob.shape) != (10000, 8, 256):
        raise ValueError(f"unexpected peer gamma shape {tuple(gamma_blob.shape)}")
    if target_blob.get("contract_version") != 1:
        raise ValueError("peer targets are not contract v1")
    labels = target_blob["patch_labels"][START:START + COUNT].long()
    names = target_blob["names"][START:START + COUNT]
    if names != manifest["splits"]["validation"][:COUNT]:
        raise ValueError("peer target names do not match the split manifest")
    if tuple(labels.shape) != (COUNT, 16, 16):
        raise ValueError(f"unexpected target shape {tuple(labels.shape)}")
    gamma = gamma_blob[START:START + COUNT].float()
    if not torch.isfinite(gamma).all():
        raise ValueError("nonfinite peer gamma")

    model = _core(args.device, args.checkpoint, STEPS, "shared", 3, 1.5, 2.0,
                  .5, 16.0, .35, "factorized", "raw")
    model.membrane_layer.vth = .06
    spike_rows, component_rows = [], []
    with torch.no_grad():
        for start in range(0, COUNT, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            _, spikes, _, _ = model(batch, return_core_out=True, return_theta=True)
            if model.last_component_spikes is None:
                raise RuntimeError("SW0072 readout requires component spikes")
            spike_rows.append(spikes.float().cpu())
            component_rows.append(model.last_component_spikes.float().cpu())
    spikes = torch.cat(spike_rows)
    components = torch.cat(component_rows)
    groups = spike_synchrony_components(
        spikes, synchrony_threshold=THRESHOLD, min_group_size=2, settle=SETTLE,
        components=components, background="largest_component",
        foreground_threshold=.15, synchrony_quantile=.35,
        spatial_sigma=None, spatial_grid_size=16, affinity_mode="spike",
    )
    prediction = spatial_components_to_patch_labels(groups, 16)
    scored = evaluate_patch_masks(prediction, labels)
    metrics = {key: float(value) for key, value in scored["mean"].items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("nonfinite score")
    pred_counts = torch.tensor([len(row) for row in groups], dtype=torch.float32)
    true_counts = torch.tensor([
        int(torch.unique(row[row != 0]).numel()) for row in labels
    ], dtype=torch.float32)
    report = {
        "experiment": "SW0086 SW0072 cross-contract transfer",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset": "peer clevr_with_masks",
        "split": "validation first 300",
        "peer_rows": [START, START + COUNT - 1],
        "images": COUNT,
        "gamma": {"path": str(args.gamma), "shape": list(gamma_blob.shape),
                  "selected_mean": float(gamma.mean()), "selected_std": float(gamma.std())},
        "targets": {"path": str(args.targets), "contract_version": 1,
                    "foreground_fraction": float((labels != 0).float().mean())},
        "inference": {"steps": STEPS, "settle": SETTLE, "membrane_vth": .06,
                      "synchrony_threshold": THRESHOLD, "affinity_mode": "spike",
                      "background": "largest_component", "min_group_size": 2,
                      "graph_spatial_decay": .35, "geodesic_steps": 3,
                      "geodesic_radius": 1.5, "geodesic_contrast": 2.0,
                      "geodesic_temperature": .5, "geodesic_cap": 16.0,
                      "kuramoto_backend": "factorized", "gate_mode": "raw"},
        "metrics": metrics,
        "valid_count": {key: int(value) for key, value in scored["valid_count"].items()},
        "diagnostics": {
            "predicted_foreground_fraction": float((prediction != 0).float().mean()),
            "predicted_object_count_mean": float(pred_counts.mean()),
            "target_object_count_mean": float(true_counts.mean()),
            "object_count_mae": float((pred_counts - true_counts).abs().mean()),
            "empty_prediction_images": int((pred_counts == 0).sum()),
        },
        "ground_truth_used_for_prediction": False,
        "interpretation": "cross-contract transfer diagnostic; no retraining or peer-contract tuning",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "metrics": metrics,
                      "diagnostics": report["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()

