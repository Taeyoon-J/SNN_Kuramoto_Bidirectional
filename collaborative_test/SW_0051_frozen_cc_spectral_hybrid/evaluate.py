"""Frozen seed-0 hybrid readout; all grouping decisions are GT-free."""
import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "collaborative_test"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0027_component_membrane_spectral"))
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0028_spatial_membrane_spectral"))
sys.path.insert(0, str(ROOT))
from evaluate_fixed_split import _core
from evaluate import correlation, score, spectral_labels
from spatial_evaluate import spatial_kernel
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from hybrid import (gamma_local_rows, generic_spectral_partition,
                    preserve_foreground, restricted_spectral_labels)


def groups_to_mask(groups, nodes=256):
    mask = np.zeros(nodes, dtype=bool)
    for group in groups:
        mask[np.asarray(group, dtype=np.int64)] = True
    return mask


def main():
    p = argparse.ArgumentParser()
    for arg in ("checkpoint", "gamma-path", "gamma-manifest", "dataset-path", "output-path"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--global-start", type=int, default=1320)
    p.add_argument("--count", type=int, default=320)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--settle", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.global_start < 1320 or args.global_start + args.count > 1640:
        raise ValueError("Only the fixed HDF5-aligned validation IDs 1320-1639 are permitted")
    if args.count < 1 or args.batch_size < 1 or not 0 <= args.settle < args.steps:
        raise ValueError("Invalid count, batch size, or rollout window")
    ids = list(range(args.global_start, args.global_start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)
    local = gamma_local_rows(len(gamma), args.global_start, args.count)
    gamma = gamma[local].float()
    manifest = json.loads(Path(args.gamma_manifest).read_text(encoding="utf-8"))
    with h5py.File(args.dataset_path, "r") as h5:
        truth = clevr_mask_patch(torch.from_numpy(h5["mask"][ids]), 8)["patch_labels"]

    model = _core(args.device, args.checkpoint, args.steps, "shared", 3, 1.5, 2.0, 0.5, 16.0, 0.35, "factorized")
    model.membrane_layer.vth = 0.06
    kernel = spatial_kernel(1.5).cpu().numpy()
    records = {key: [] for key in (
        "membrane_freeze", "spike_freeze", "membrane_restrict_k10",
        "spike_restrict_k10", "membrane_restrict_dynamic", "spike_restrict_dynamic")}
    foreground_counts, foreground_fractions = [], []

    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            stop = min(start + args.batch_size, args.count)
            _, spikes, membrane, _ = model(gamma[start:stop].to(args.device), return_core_out=True, return_theta=True)
            component_spikes = model.last_component_spikes
            # The connected-component helper builds CPU bookkeeping tensors.
            # Keep its inputs on CPU; spectral affinities below are CPU too.
            cc_groups = spike_synchrony_components(
                spikes.cpu(), synchrony_threshold=0.35, min_group_size=2,
                settle=args.settle, components=component_spikes.cpu(),
                background="largest_component")
            for bi, groups in enumerate(cc_groups):
                fg = groups_to_mask(groups)
                foreground_counts.append(len(groups))
                foreground_fractions.append(float(fg.mean()))
                cc_labels = spatial_components_to_patch_labels([groups], 16)[0].numpy().reshape(-1)
                # Connected-component mask is fixed for every hybrid variant.
                for signal, history in (("membrane", membrane), ("spike", spikes)):
                    affinity = correlation(history[bi, :, args.settle:].cpu()).abs() * torch.from_numpy(kernel)
                    full = np.asarray(spectral_labels(affinity, 10), dtype=np.int64).reshape(-1) + 1
                    frozen = preserve_foreground(full, fg)
                    records[f"{signal}_freeze"].append(frozen.reshape(16, 16))
                    fixed = restricted_spectral_labels(affinity.numpy(), fg, 10, generic_spectral_partition)
                    dynamic_k = max(2, len(groups))
                    dynamic = restricted_spectral_labels(affinity.numpy(), fg, dynamic_k, generic_spectral_partition)
                    records[f"{signal}_restrict_k10"].append(fixed.reshape(16, 16))
                    records[f"{signal}_restrict_dynamic"].append(dynamic.reshape(16, 16))
                # Baseline labels are recorded too, permitting exact mask audit.
                records.setdefault("spike_cc_baseline", []).append(cc_labels.reshape(16, 16))

    predictions = {key: torch.from_numpy(np.stack(value)).long() for key, value in records.items()}
    rows = []
    for name, pred in predictions.items():
        scores = score(pred, truth)
        rows.append({"mode": name, **scores,
                     "predicted_foreground_fraction": float((pred != 0).float().mean()),
                     "predicted_object_count_mean": float(torch.tensor([
                         torch.unique(x[x != 0]).numel() for x in pred]).float().mean()),
                     "foreground_mask_exactly_matches_spike_cc": bool(torch.equal(
                         pred != 0, predictions["spike_cc_baseline"] != 0))})
    report = {
        "experiment": "SW0051 frozen spike-CC spectral hybrids",
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "gamma": {"path": args.gamma_path, "manifest_path": args.gamma_manifest,
                  "manifest": manifest, "global_start": args.global_start},
        "target": {"path": args.dataset_path, "ids": [ids[0], ids[-1]]},
        "core": {"steps": args.steps, "settle": args.settle, "vth": 0.06,
                 "projection": "shared", "graph_spatial_decay": 0.35,
                 "geodesic_steps": 3, "kuramoto_backend": "factorized"},
        "readout": {"sigma": 1.5, "spectral_k": 10,
                    "frozen_spike_cc_threshold": 0.35,
                    "prediction_uses_ground_truth": False,
                    "foreground_component_count_mean": float(np.mean(foreground_counts)),
                    "foreground_fraction_mean": float(np.mean(foreground_fractions))},
        "rows": rows,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
