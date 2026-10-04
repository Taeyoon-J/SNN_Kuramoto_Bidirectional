"""Measure foreground separation in actual SNN traces, without changing masks."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test"))
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch


def border_template_scores(traces):
    """Same centered-cosine border prototype as the SW_0022 evaluator."""
    centered = traces - traces.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-8)
    border = np.zeros((16, 16), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    prototype = normalized[border.reshape(-1)].mean(axis=0)
    magnitude = np.linalg.norm(prototype)
    if magnitude < 1e-8:
        return np.zeros(256, dtype=np.float32)
    return normalized @ (prototype / magnitude)


def image_auc(scores, foreground):
    positive = int(foreground.sum())
    negative = len(foreground) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[foreground].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


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
    features = {name: [] for name in (
        "spike_rate", "membrane_mean", "membrane_abs_mean", "membrane_std",
        "membrane_max", "membrane_range", "membrane_temporal_change",
        "border_template_similarity",
    )}
    with torch.no_grad():
        for start in range(0, args.count, args.batch_size):
            _, spikes, membrane = model(gamma[start:start + args.batch_size].to(args.device), return_core_out=True)
            spike = spikes[:, :, 64:].cpu().numpy()
            mem = membrane[:, :, 64:].cpu().numpy()
            batch_features = {
                "spike_rate": spike.mean(axis=-1),
                "membrane_mean": mem.mean(axis=-1),
                "membrane_abs_mean": np.abs(mem).mean(axis=-1),
                "membrane_std": mem.std(axis=-1),
                "membrane_max": mem.max(axis=-1),
                "membrane_range": np.ptp(mem, axis=-1),
                "membrane_temporal_change": np.abs(np.diff(mem, axis=-1)).mean(axis=-1),
                "border_template_similarity": np.stack([border_template_scores(t) for t in mem]),
            }
            for name, value in batch_features.items():
                features[name].extend(value)
    foreground = truth.numpy().reshape(args.count, -1) != 0
    rows = []
    for name, per_image in features.items():
        values = np.asarray(per_image)
        aucs = [image_auc(scores, mask) for scores, mask in zip(values, foreground)]
        aucs = [value for value in aucs if value is not None]
        rows.append({
            "feature": name,
            "true_foreground_mean": float(values[foreground].mean()),
            "true_background_mean": float(values[~foreground].mean()),
            "per_image_auc_mean": float(np.mean(aucs)),
            "per_image_auc_median": float(np.median(aucs)),
            "images_with_valid_auc": len(aucs),
        })
    result = {
        "checkpoint": args.checkpoint,
        "ids": [ids[0], ids[-1]],
        "true_foreground_fraction": float(foreground.mean()),
        "warning": "GT mask is used only to label this post-hoc diagnostic; no predictions or model training here.",
        "rows": rows,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
