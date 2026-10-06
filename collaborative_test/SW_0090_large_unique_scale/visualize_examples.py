"""Export pre-registered qualitative masks for the selected SW0090 checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]
from evaluate_fixed_split import _core
from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components

PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
    [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128],
], dtype=np.uint8)


def colorize(labels):
    labels = np.asarray(labels, dtype=np.int64)
    return PALETTE[labels % len(PALETTE)]


def save_mask_set(labels, directory):
    directory.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in np.unique(labels) if x != 0]
    for index, label in enumerate(ids, 1):
        mask = (labels == label).astype(np.uint8) * 255
        Image.fromarray(mask, mode="L").save(directory / f"mask_{index:02d}.png")
    return [(labels == label).astype(np.uint8) for label in ids]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gamma", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", type=int, nargs=3, default=[1320, 1321, 1322])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if any(scene < 1320 or scene > 1639 for scene in a.ids):
        raise ValueError("visualization IDs must be in the fixed held-out split")

    gamma_all = torch.load(a.gamma, map_location="cpu", weights_only=True)
    gamma = gamma_all[[scene - 1320 for scene in a.ids]].float().to(a.device)
    with h5py.File(a.dataset, "r") as data:
        originals = np.asarray(data["image"][a.ids])
        raw_masks = np.asarray(data["mask"][a.ids])
    targets = clevr_mask_patch(torch.from_numpy(raw_masks), 8)["patch_labels"]

    model = _core(a.device, a.checkpoint, 1024, "shared", 3, 1.5, 2.0,
                  .5, 16.0, .35, "factorized", "raw")
    model.membrane_layer.vth = .06
    with torch.no_grad():
        _, spikes, _ = model(gamma, return_core_out=True)
        components = model.last_component_spikes.detach().float().cpu()
        spikes = spikes.detach().float().cpu()
    groups = spike_synchrony_components(
        spikes, synchrony_threshold=.50, min_group_size=2, settle=512,
        components=components, background="largest_component",
        foreground_threshold=.15, synchrony_quantile=.35,
    )
    predictions = spatial_components_to_patch_labels(groups, 16).cpu()

    a.output.mkdir(parents=True, exist_ok=True)
    overview = plt.figure(figsize=(12, 12), constrained_layout=True)
    overview_grid = overview.subplots(3, 3)
    manifest = {"checkpoint": a.checkpoint, "ids": a.ids, "examples": []}
    for row, scene in enumerate(a.ids):
        target_patch = targets[row].cpu().numpy()
        pred_patch = predictions[row].cpu().numpy()
        target_full = np.repeat(np.repeat(target_patch, 8, axis=0), 8, axis=1)
        pred_full = np.repeat(np.repeat(pred_patch, 8, axis=0), 8, axis=1)
        scene_dir = a.output / f"image_{scene}"
        scene_dir.mkdir(exist_ok=True)
        Image.fromarray(originals[row]).save(scene_dir / "original.png")
        Image.fromarray(colorize(target_full)).save(scene_dir / "ground_truth_instance_map.png")
        Image.fromarray(colorize(pred_full)).save(scene_dir / "predicted_instance_map.png")
        gt_masks = save_mask_set(target_full, scene_dir / "ground_truth_masks")
        pred_masks = save_mask_set(pred_full, scene_dir / "predicted_masks")

        tiles = [("Original", originals[row]), ("GT instances", colorize(target_full)),
                 ("Predicted instances", colorize(pred_full))]
        tiles += [(f"GT mask {i+1}", m * 255) for i, m in enumerate(gt_masks)]
        tiles += [(f"Pred mask {i+1}", m * 255) for i, m in enumerate(pred_masks)]
        columns = 4
        rows = (len(tiles) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
        for ax, (title, tile) in zip(axes.flat, tiles):
            ax.imshow(tile, cmap="gray" if tile.ndim == 2 else None, vmin=0, vmax=255)
            ax.set_title(title); ax.axis("off")
        for ax in axes.flat[len(tiles):]: ax.axis("off")
        fig.suptitle(f"Held-out image {scene}: {len(gt_masks)} GT objects, {len(pred_masks)} predicted")
        fig.tight_layout()
        fig.savefig(scene_dir / "mask_grid.png", dpi=160)
        plt.close(fig)

        for col, (title, tile) in enumerate(tiles[:3]):
            overview_grid[row, col].imshow(tile)
            overview_grid[row, col].set_title(f"{scene} — {title}")
            overview_grid[row, col].axis("off")
        manifest["examples"].append({"id": scene, "ground_truth_objects": len(gt_masks),
                                     "predicted_objects": len(pred_masks)})
    overview.savefig(a.output / "overview_all_three.png", dpi=180)
    plt.close(overview)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

