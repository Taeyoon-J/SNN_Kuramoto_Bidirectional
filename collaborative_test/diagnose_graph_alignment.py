"""Check whether the learned graph favors pairs from the same CLEVR object.

Diagnostic only: instance labels never enter training or mask prediction.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))

from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.evaluate_binding import spectral_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=256, sc=None,
        gamma_drive_mode="static", num_time_steps=64,
        theta_init="gamma", gamma_phase_mode="standardize_tanh", osc_dim=4,
        freq_gain=2.0, graph_mode="learned", graph_top_k=32,
        graph_spatial_decay=0.55, k=256.0, low_n=-4.0, high_n=0.0,
        membrane_vth=0.06, membrane_low_m=-4.0, membrane_high_m=0.0,
        gate_mode="raw", spike_classify_method="spatial_components",
        spike_spatial_grid_size=(16, 16), spike_per_component=True,
    ).validate()
    model = S2NetCore(hp, device="cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    ids = list(range(args.start, args.start + args.count))
    with h5py.File(args.dataset_path, "r") as dataset:
        labels = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]

    pair_types = ("same_object", "different_objects", "background_background", "foreground_background")
    sums = {key: 0.0 for key in pair_types}
    counts = {key: 0 for key in pair_types}
    nonzero = {key: 0 for key in pair_types}
    diagnostic_predictions = []
    eye = torch.eye(256, dtype=torch.bool)
    with torch.no_grad():
        for image_index, image_id in enumerate(ids):
            graph = model.graph_generator(gamma[image_id:image_id + 1].float())[0]
            raw_groups = spectral_cluster(graph, 8)
            background_group = int(torch.bincount(raw_groups, minlength=8).argmax())
            diagnostic = torch.zeros_like(raw_groups, dtype=torch.int64)
            next_id = 1
            for group_id in range(8):
                if group_id != background_group:
                    diagnostic[raw_groups == group_id] = next_id
                    next_id += 1
            diagnostic_predictions.append(diagnostic.reshape(16, 16))
            target = labels[image_index].flatten()
            foreground = target != 0
            same = target[:, None] == target[None, :]
            masks = {
                "same_object": foreground[:, None] & foreground[None, :] & same,
                "different_objects": foreground[:, None] & foreground[None, :] & ~same,
                "background_background": ~foreground[:, None] & ~foreground[None, :],
                "foreground_background": foreground[:, None] ^ foreground[None, :],
            }
            for key, mask in masks.items():
                mask &= ~eye
                values = graph[mask]
                sums[key] += values.sum().item()
                counts[key] += values.numel()
                nonzero[key] += int((values > 0).sum())
    report = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "pair_categories": {
            key: {
                "mean_graph_weight": sums[key] / counts[key] if counts[key] else None,
                "nonzero_fraction": nonzero[key] / counts[key] if counts[key] else None,
                "pair_count": counts[key],
            }
            for key in pair_types
        },
    }
    diagnostic_scores = evaluate_patch_masks(torch.stack(diagnostic_predictions), labels)
    report["graph_only_diagnostic_not_valid_final_readout"] = {
        key: float(value) for key, value in diagnostic_scores["mean"].items()
    }
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
