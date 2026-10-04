"""Separate adaptive-slot grouping errors from background-selection errors."""
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


def set_background(raw, policy, border_fraction=0.2):
    """Use only predicted group geometry, never GT, to choose background."""
    flat = raw.reshape(-1)
    output = flat.copy()
    group_ids, counts = np.unique(flat[flat > 0], return_counts=True)
    if not len(group_ids):
        return raw
    largest = int(group_ids[counts.argmax()])
    border = np.zeros((16, 16), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border = border.reshape(-1)
    background = set()
    if policy in ("largest", "largest_plus_border"):
        background.add(largest)
    if policy in ("border", "largest_plus_border"):
        for group, count in zip(group_ids, counts):
            border_count = int(((flat == group) & border).sum())
            if border_count >= 2 and border_count / int(count) >= border_fraction:
                background.add(int(group))
    for group in background:
        output[flat == group] = 0
    # Canonical consecutive object IDs; metric does not depend on ID names.
    ids = np.unique(output[output > 0])
    relabeled = np.zeros_like(output)
    for new_id, old_id in enumerate(ids, 1):
        relabeled[output == old_id] = new_id
    return relabeled.reshape(16, 16)


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
    truth_count = torch.tensor([torch.unique(t[t != 0]).numel() for t in truth])
    rows = []
    for threshold, initial_slots in ((0.3, 1), (0.5, 3), (0.7, 6)):
        raw = [dynamic_slots(history, threshold, initial_slots,
                             image_id * 1009, assign_background=False)
               for image_id, history in zip(ids, histories)]
        for policy in ("largest", "border", "largest_plus_border", "none"):
            for fraction in ((0.1, 0.2, 0.3) if "border" in policy else (0.2,)):
                labels = torch.from_numpy(np.stack([
                    set_background(item, policy, fraction) for item in raw
                ]))
                scores = evaluate_patch_masks(labels, truth)["mean"]
                predicted_count = torch.tensor([
                    torch.unique(item[item != 0]).numel() for item in labels
                ])
                foreground = labels != 0
                actual = truth != 0
                tp = (foreground & actual).sum().item()
                rows.append({
                    "threshold": threshold, "initial_slots": initial_slots,
                    "background_policy": policy, "border_fraction": fraction,
                    "fg_ari": float(scores["fg_ari"]),
                    "foreground_iou": float(scores["foreground_iou"]),
                    "matched_object_iou": float(scores["matched_object_iou"]),
                    "predicted_groups_mean": float(predicted_count.float().mean()),
                    "exact_count_fraction": float((predicted_count == truth_count).float().mean()),
                    "foreground_precision": tp / max(foreground.sum().item(), 1),
                    "foreground_recall": tp / max(actual.sum().item(), 1),
                })
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ids": [ids[0], ids[-1]], "checkpoint": args.checkpoint,
                                  "true_groups_mean": float(truth_count.float().mean()),
                                  "rows": rows}, indent=2))
    print(json.dumps({"output": str(output), "best_ari": max(rows, key=lambda r: r["fg_ari"])}, indent=2))


if __name__ == "__main__":
    main()
