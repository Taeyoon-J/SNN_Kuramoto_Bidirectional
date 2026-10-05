#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0040_peer_transfer.evaluate import gamma_row_indices
from collaborative_test.evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.loss_function import phase_locking_value, signal_synchrony


WINDOWS = {"early": (0, 64), "middle": (96, 160), "late": (192, 256)}


class PairAccumulator:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name, affinity, labels):
        affinity = affinity.detach().float().cpu()
        labels = labels.detach().long().cpu().flatten(1)
        n = labels.size(1)
        offdiag = ~torch.eye(n, dtype=torch.bool).unsqueeze(0)
        fg = labels > 0
        same = labels.unsqueeze(2) == labels.unsqueeze(1)
        masks = {
            "same_foreground": same & fg.unsqueeze(2) & fg.unsqueeze(1) & offdiag,
            "different_foreground": (~same) & fg.unsqueeze(2) & fg.unsqueeze(1) & offdiag,
            "foreground_background": (~same) & (fg.unsqueeze(2) ^ fg.unsqueeze(1)) & offdiag,
        }
        for key, mask in masks.items():
            values = affinity[mask]
            self.sums[(name, key)] += float(values.sum())
            self.counts[(name, key)] += int(values.numel())

    def result(self):
        stages = sorted({name for name, _ in self.sums})
        out = {}
        for stage in stages:
            row = {}
            for key in ("same_foreground", "different_foreground", "foreground_background"):
                count = self.counts[(stage, key)]
                if count == 0:
                    raise ValueError(f"no pairs for {stage}/{key}")
                row[key] = self.sums[(stage, key)] / count
            row["object_margin"] = row["same_foreground"] - row["different_foreground"]
            row["foreground_background_margin"] = row["same_foreground"] - row["foreground_background"]
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"non-finite stage {stage}")
            out[stage] = row
        return out


def cosine_affinity(gamma):
    x = F.normalize(gamma.transpose(1, 2).float(), dim=-1)
    return (1.0 + torch.bmm(x, x.transpose(1, 2))) / 2.0


def normalized_graph(graph):
    return graph / graph.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)


def gate_trace(theta):
    values = []
    for t in range(theta.size(1)):
        delayed = theta[:, max(0, t - 2)]
        values.append(0.5 * (1.0 + torch.sin(delayed.mean(dim=-1))))
    return torch.stack(values, dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gamma", type=Path, required=True)
    p.add_argument("--gamma-global-start", type=int, default=1320)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--start", type=int, default=1320)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps != 256 or args.count != 32 or args.start != 1320:
        raise ValueError("initial SW0064 contract requires IDs1320-1351 and T256")
    gamma_blob = torch.load(args.gamma, map_location="cpu", weights_only=True).float()
    ids = list(range(args.start, args.start + args.count))
    rows = gamma_row_indices(ids, args.gamma_global_start, len(gamma_blob))
    gamma = gamma_blob[rows]
    with h5py.File(args.hdf5, "r") as dataset:
        labels = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, args.steps, "shared", 3, 1.5, 2.0, 0.5, 16.0, 0.35, "factorized", "raw")
    accumulator = PairAccumulator()
    activity = defaultdict(lambda: {"foreground_sum": 0.0, "foreground_count": 0, "background_sum": 0.0, "background_count": 0})
    with torch.no_grad():
        for offset in range(0, args.count, args.batch_size):
            x = gamma[offset:offset + args.batch_size].to(args.device)
            y = labels[offset:offset + args.batch_size]
            dendritic_steps = []
            handle = model.dendric_layer.register_forward_hook(lambda module, inputs, output: dendritic_steps.append(output.detach()))
            graph = model.graph_generator(x)
            _, spikes, membrane, theta = model(x, return_core_out=True, return_theta=True)
            handle.remove()
            batch = x.size(0)
            dendritic = torch.stack(dendritic_steps, dim=-1).reshape(batch, model.osc_dim, 256, args.steps).mean(dim=1)
            gate = gate_trace(theta)
            accumulator.add("gamma", cosine_affinity(x), y)
            accumulator.add("graph", normalized_graph(graph), y)
            for window, (begin, end) in WINDOWS.items():
                accumulator.add(f"kuramoto_plv_{window}", phase_locking_value(theta[:, begin:end], combine="product"), y)
                traces = {"gate": gate, "dendritic": dendritic, "membrane": membrane, "spike": spikes}
                for stage, trace in traces.items():
                    segment = trace[:, :, begin:end]
                    accumulator.add(f"{stage}_{window}", signal_synchrony(segment), y)
                    magnitude = segment.abs().mean(dim=-1).detach().cpu()
                    flat_labels = y.flatten(1)
                    fg = flat_labels > 0
                    key = f"{stage}_{window}"
                    activity[key]["foreground_sum"] += float(magnitude[fg].sum())
                    activity[key]["foreground_count"] += int(fg.sum())
                    activity[key]["background_sum"] += float(magnitude[~fg].sum())
                    activity[key]["background_count"] += int((~fg).sum())
    pair_metrics = accumulator.result()
    activity_metrics = {}
    for key, row in sorted(activity.items()):
        fg = row["foreground_sum"] / row["foreground_count"]
        bg = row["background_sum"] / row["background_count"]
        activity_metrics[key] = {"foreground": fg, "background": bg, "foreground_minus_background": fg - bg}
    late_order = ["gamma", "graph", "kuramoto_plv_late", "gate_late", "dendritic_late", "membrane_late", "spike_late"]
    propagation = []
    for previous, current in zip(late_order, late_order[1:]):
        propagation.append({"from": previous, "to": current, "object_margin_delta": pair_metrics[current]["object_margin"] - pair_metrics[previous]["object_margin"]})
    result = {
        "experiment": "SW0064 layer/time activation-flow probe",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "ids": [args.start, args.start + args.count - 1],
        "steps": args.steps,
        "windows": WINDOWS,
        "ground_truth_role": "diagnostic only after forward pass; never prediction input",
        "pair_affinity": pair_metrics,
        "activity": activity_metrics,
        "late_stage_propagation": propagation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "late_object_margins": {k: pair_metrics[k]["object_margin"] for k in late_order}}, indent=2))


if __name__ == "__main__":
    main()
