"""Evaluate adaptive slots derived from actual patch spike histories."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, evaluate_patch_masks


def dynamic_slots(traces, threshold, initial_slots, seed):
    """Grow cosine slots from observed nonconstant spike traces until assigned."""
    traces = np.asarray(traces, dtype=np.float32)
    centered = traces - traces.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    valid = np.flatnonzero(norm > 1e-8)
    labels = np.zeros(len(traces), dtype=np.int64)
    if not len(valid):
        return labels.reshape(16, 16)
    x = centered[valid] / norm[valid, None]
    rng = np.random.default_rng(seed)
    seeds = rng.choice(len(x), size=min(initial_slots, len(x)), replace=False)
    centers = x[seeds].copy()
    assigned = np.full(len(x), -1, dtype=np.int64)
    for _ in range(len(x) + 1):
        while True:
            remaining = np.flatnonzero(assigned < 0)
            if not len(remaining):
                break
            similarity = x[remaining] @ centers.T
            nearest = similarity.argmax(axis=1)
            accepted = similarity[np.arange(len(remaining)), nearest] >= threshold
            if not accepted.any():
                break
            assigned[remaining[accepted]] = nearest[accepted]
            for slot in np.unique(nearest[accepted]):
                mean = x[assigned == slot].mean(axis=0)
                magnitude = np.linalg.norm(mean)
                if magnitude > 1e-8:
                    centers[slot] = mean / magnitude
        remaining = np.flatnonzero(assigned < 0)
        if not len(remaining):
            break
        centers = np.vstack((centers, x[rng.choice(remaining)]))
    if (assigned < 0).any():
        raise RuntimeError("Slot assignment did not terminate")
    counts = np.bincount(assigned, minlength=len(centers))
    background_slot = int(counts.argmax())
    object_id = 1
    for slot, count in enumerate(counts):
        if slot != background_slot and count:
            labels[valid[assigned == slot]] = object_id
            object_id += 1
    return labels.reshape(16, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--initial-slots", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--assignment-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    if args.settle >= args.steps or min(args.initial_slots) < 1:
        raise ValueError("Invalid settle or initial slot count")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, args.steps)
    histories = []
    with torch.no_grad():
        for index in range(0, args.count, args.batch_size):
            _, spikes, _ = model(gamma[index:index + args.batch_size].to(args.device), return_core_out=True)
            histories.extend(spikes[:, :, args.settle:].cpu().numpy())
    true_counts = torch.tensor([torch.unique(t[t != 0]).numel() for t in truth])
    rows = []
    for threshold in args.thresholds:
        for slots in args.initial_slots:
            for seed in args.assignment_seeds:
                predicted = torch.from_numpy(np.stack([
                    dynamic_slots(history, threshold, slots, seed + image_id * 1009)
                    for image_id, history in zip(ids, histories)
                ]))
                scores = evaluate_patch_masks(predicted, truth)["mean"]
                counts = torch.tensor([torch.unique(p[p != 0]).numel() for p in predicted])
                rows.append({
                    "threshold": threshold, "initial_slots": slots, "assignment_seed": seed,
                    "fg_ari": float(scores["fg_ari"]),
                    "foreground_iou": float(scores["foreground_iou"]),
                    "matched_object_iou": float(scores["matched_object_iou"]),
                    "predicted_groups_mean": float(counts.float().mean()),
                    "exact_count_fraction": float((counts == true_counts).float().mean()),
                    "within_one_count_fraction": float(((counts - true_counts).abs() <= 1).float().mean()),
                    "predicted_foreground_fraction": float((predicted != 0).float().mean()),
                })
    result = {"checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
              "source": "actual core spike histories after settle",
              "background_rule": "zero traces and largest slot",
              "true_groups_mean": float(true_counts.float().mean()), "rows": rows}
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "best_fg_ari": max(rows, key=lambda r: r["fg_ari"])}, indent=2))


if __name__ == "__main__":
    main()
