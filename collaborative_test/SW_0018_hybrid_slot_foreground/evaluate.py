"""Validation test: low spike synchrony for FG, adaptive spike slots for grouping."""
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


def foreground_rank(traces, fraction):
    centered = traces - traces.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-8)
    # The peer's insight: foreground may be less synchronized with the whole image.
    degree = (normalized @ normalized.T).mean(axis=1)
    keep = np.zeros(len(degree), dtype=bool)
    keep[np.argsort(degree, kind="stable")[:round(fraction * len(degree))]] = True
    return keep


def apply_foreground(raw, keep, remove_largest):
    output = raw.reshape(-1).copy()
    output[~keep] = 0
    positive, counts = np.unique(output[output > 0], return_counts=True)
    if remove_largest and len(positive):
        output[output == positive[counts.argmax()]] = 0
    ids = np.unique(output[output > 0])
    canonical = np.zeros_like(output)
    for new_id, old_id in enumerate(ids, 1):
        canonical[output == old_id] = new_id
    return canonical.reshape(16, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gamma-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start", type=int, default=1320)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.start < 1320 or args.start + args.count > 1640:
        raise ValueError("Use fixed validation IDs 1320-1639 only")
    ids = list(range(args.start, args.start + args.count))
    gamma = torch.load(args.gamma_path, map_location="cpu", weights_only=True)[ids].float()
    with h5py.File(args.dataset_path, "r") as dataset:
        truth = clevr_mask_patch(torch.from_numpy(dataset["mask"][ids]), 8)["patch_labels"]
    model = _core(args.device, args.checkpoint, 256)
    histories = []
    with torch.no_grad():
        for index in range(0, len(ids), args.batch_size):
            _, spikes, _ = model(gamma[index:index + args.batch_size].to(args.device), return_core_out=True)
            histories.extend(spikes[:, :, 64:].cpu().numpy())
    true_counts = torch.tensor([torch.unique(t[t != 0]).numel() for t in truth])
    keep_sets = {fraction: [foreground_rank(h, fraction) for h in histories]
                 for fraction in (0.2, 0.3, 0.4, 0.5)}
    rows = []
    for threshold, slots in ((0.3, 3), (0.5, 3), (0.5, 6), (0.7, 6)):
        raw = [dynamic_slots(history, threshold, slots, image_id * 1009,
                             assign_background=False)
               for image_id, history in zip(ids, histories)]
        for fraction, keeps in keep_sets.items():
            for remove_largest in (False, True):
                prediction = torch.from_numpy(np.stack([
                    apply_foreground(p, keep, remove_largest)
                    for p, keep in zip(raw, keeps)
                ]))
                scores = evaluate_patch_masks(prediction, truth)["mean"]
                counts = torch.tensor([torch.unique(p[p != 0]).numel() for p in prediction])
                fg = prediction != 0
                actual = truth != 0
                tp = (fg & actual).sum().item()
                rows.append({
                    "threshold": threshold, "initial_slots": slots,
                    "candidate_foreground_fraction": fraction,
                    "remove_largest_slot": remove_largest,
                    "fg_ari": float(scores["fg_ari"]),
                    "foreground_iou": float(scores["foreground_iou"]),
                    "matched_object_iou": float(scores["matched_object_iou"]),
                    "predicted_groups_mean": float(counts.float().mean()),
                    "exact_count_fraction": float((counts == true_counts).float().mean()),
                    "foreground_precision": tp / max(fg.sum().item(), 1),
                    "foreground_recall": tp / max(actual.sum().item(), 1),
                })
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ids": [ids[0], ids[-1]], "checkpoint": args.checkpoint,
                                  "source": "actual spike histories", "true_groups_mean": float(true_counts.float().mean()),
                                  "rows": rows}, indent=2))
    print(json.dumps({"best_ari": max(rows, key=lambda r: r["fg_ari"]),
                      "best_iou": max(rows, key=lambda r: r["foreground_iou"])}, indent=2))


if __name__ == "__main__":
    main()
