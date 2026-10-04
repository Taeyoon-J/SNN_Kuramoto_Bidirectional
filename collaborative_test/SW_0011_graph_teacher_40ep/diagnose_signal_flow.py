"""Diagnostic: does Kuramoto phase grouping survive in membrane and spikes?"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.loss_function import phase_locking_value, signal_synchrony


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use validation IDs 1320–1639 only.")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    totals = {
        name: {category: {"sum": 0.0, "count": 0}
               for category in ("same_object", "different_objects", "all_offdiagonal")}
        for name in ("phase_plv", "membrane_synchrony", "spike_synchrony")
    }
    spike_sum = 0.0
    spike_count = 0
    correlations = []
    eye = torch.eye(256, dtype=torch.bool, device=args.device)
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            batch = gamma[start:start + args.batch_size].to(args.device)
            _, spikes, membrane, theta = model(batch, return_core_out=True, return_theta=True)
            phase = phase_locking_value(theta, settle=64, combine="mean")
            membrane_sync = signal_synchrony(membrane, settle=64)
            spike_sync = signal_synchrony(spikes, settle=64)
            spike_sum += float(spikes.sum())
            spike_count += spikes.numel()
            for offset in range(batch.size(0)):
                labels = truth[start + offset].flatten().to(args.device)
                foreground = labels != 0
                same = labels[:, None] == labels[None, :]
                masks = {
                    "same_object": foreground[:, None] & foreground[None, :] & same & ~eye,
                    "different_objects": foreground[:, None] & foreground[None, :] & ~same,
                    "all_offdiagonal": ~eye,
                }
                for name, affinity in (
                    ("phase_plv", phase[offset]),
                    ("membrane_synchrony", membrane_sync[offset]),
                    ("spike_synchrony", spike_sync[offset]),
                ):
                    for category, mask in masks.items():
                        values = affinity[mask]
                        totals[name][category]["sum"] += float(values.sum())
                        totals[name][category]["count"] += values.numel()
                phase_vector = phase[offset][~eye]
                spike_vector = spike_sync[offset][~eye]
                phase_centered = phase_vector - phase_vector.mean()
                spike_centered = spike_vector - spike_vector.mean()
                denominator = phase_centered.norm() * spike_centered.norm()
                correlations.append(float((phase_centered @ spike_centered) / denominator.clamp_min(1e-8)))
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "steps": 256,
        "settle": 64,
        "mean_spike_rate": spike_sum / spike_count,
        "mean_phase_to_spike_affinity_correlation": sum(correlations) / len(correlations),
        "affinity": {
            name: {
                category: entry["sum"] / entry["count"]
                for category, entry in values.items()
            }
            for name, values in totals.items()
        },
        "warning": "Ground truth only partitions pair diagnostics; it is not used for training or prediction.",
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
