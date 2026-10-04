"""Foreground policy test on SW_0021's membrane-derived adaptive slots."""
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


def border_template_scores(traces):
    """Cosine to mean border membrane pattern; no RGB or GT used."""
    traces = np.asarray(traces, dtype=np.float32)
    centered = traces - traces.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-8)
    border = np.zeros((16, 16), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    template = normalized[border.reshape(-1)].mean(axis=0)
    template_norm = np.linalg.norm(template)
    if template_norm < 1e-8:
        return np.zeros(256, dtype=np.float32)
    return normalized @ (template / template_norm)


def apply_background(raw, scores, policy, threshold):
    flat = raw.reshape(-1).copy()
    positive, counts = np.unique(flat[flat > 0], return_counts=True)
    largest = int(positive[counts.argmax()]) if len(positive) else -1
    by_slot = flat == largest
    by_template = scores >= threshold
    if policy == "largest":
        background = by_slot
    elif policy == "template_only":
        background = by_template
    elif policy == "largest_or_template":
        background = by_slot | by_template
    elif policy == "largest_and_template":
        background = by_slot & by_template
    else:
        raise ValueError("Unknown background policy")
    flat[background] = 0
    result = np.zeros_like(flat)
    for new_id, old_id in enumerate(np.unique(flat[flat > 0]), 1):
        result[flat == old_id] = new_id
    return result.reshape(16, 16)


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
        for start in range(0, args.count, args.batch_size):
            _, _, membrane = model(gamma[start:start + args.batch_size].to(args.device), return_core_out=True)
            histories.extend(membrane[:, :, 64:].cpu().numpy())
    raw = [dynamic_slots(history, 0.7, 6, image_id * 1009,
                         assign_background=False)
           for image_id, history in zip(ids, histories)]
    template_scores = [border_template_scores(history) for history in histories]
    true_counts = torch.tensor([torch.unique(t[t != 0]).numel() for t in truth])
    all_scores = np.stack(template_scores)
    true_foreground = truth.numpy().reshape(args.count, -1) > 0
    rows = []
    policies = [("largest", 0.0)]
    policies.extend((policy, threshold)
                    for policy in ("template_only", "largest_or_template", "largest_and_template")
                    for threshold in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98))
    for policy, threshold in policies:
        predicted = torch.from_numpy(np.stack([
            apply_background(grouping, scores, policy, threshold)
            for grouping, scores in zip(raw, template_scores)
        ]))
        metrics = evaluate_patch_masks(predicted, truth)["mean"]
        counts = torch.tensor([torch.unique(p[p != 0]).numel() for p in predicted])
        foreground = predicted != 0
        actual = truth != 0
        tp = (foreground & actual).sum().item()
        rows.append({
            "policy": policy, "threshold": threshold,
            "fg_ari": float(metrics["fg_ari"]),
            "foreground_iou": float(metrics["foreground_iou"]),
            "matched_object_iou": float(metrics["matched_object_iou"]),
            "predicted_groups_mean": float(counts.float().mean()),
            "exact_count_fraction": float((counts == true_counts).float().mean()),
            "predicted_foreground_fraction": float(foreground.float().mean()),
            "foreground_precision": tp / max(foreground.sum().item(), 1),
            "foreground_recall": tp / max(actual.sum().item(), 1),
        })
    result = {
        "checkpoint": args.checkpoint, "ids": [ids[0], ids[-1]],
        "source": "actual membrane history; border prototype from membrane only",
        "slots": "SW_0021 adaptive slots, threshold .7, six initial, seed 0",
        "true_groups_mean": float(true_counts.float().mean()),
        "true_foreground_fraction": float(true_foreground.mean()),
        "diagnostic_template_score_mean": {
            "true_background": float(all_scores[~true_foreground].mean()),
            "true_foreground": float(all_scores[true_foreground].mean()),
        },
        "rows": rows,
    }
    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(path),
                      "best_fg_ari": max(rows, key=lambda row: row["fg_ari"]),
                      "best_foreground_iou": max(rows, key=lambda row: row["foreground_iou"])}, indent=2))


if __name__ == "__main__":
    main()
