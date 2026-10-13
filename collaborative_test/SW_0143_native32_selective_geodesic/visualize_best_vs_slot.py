"""Render supplied SW0143/Slot32 labels; this utility never loads models or GT files."""
from __future__ import annotations

import argparse
import colorsys
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_IDS = (1320, 1321, 1322)
METRICS = ("fg_ari", "foreground_iou", "matched_object_iou")
BACKGROUND_COLOR = (0.20, 0.20, 0.20)
PALETTE_GOLDEN_RATIO = 0.618033988749895


def _read_inputs(bundle_path: Path, metadata_path: Path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("seed") != 0:
        raise ValueError("visualization metadata must identify seed 0")
    if metadata.get("selection") != "first3 fixed validation IDs, not selected by quality":
        raise ValueError("metadata must state that these are the first three fixed IDs, not quality-selected")
    expected = list(EXPECTED_IDS)
    with np.load(bundle_path, allow_pickle=False) as archive:
        required = {"image_ids", "rgb", "gt32", "ours32", "slot32"}
        if not required.issubset(archive.files):
            raise ValueError(f"sample bundle is missing keys: {sorted(required - set(archive.files))}")
        image_ids = np.asarray(archive["image_ids"])
        rgb = np.asarray(archive["rgb"])
        gt = np.asarray(archive["gt32"])
        ours = np.asarray(archive["ours32"])
        slot = np.asarray(archive["slot32"])
    if image_ids.shape != (3,) or image_ids.dtype.kind not in "iu" or image_ids.tolist() != expected:
        raise ValueError(f"bundle image_ids must exactly equal {expected}")
    if rgb.dtype != np.uint8 or rgb.shape != (3, 128, 128, 3):
        raise ValueError("rgb must be uint8 [3,128,128,3]")
    for name, labels in (("gt32", gt), ("ours32", ours), ("slot32", slot)):
        if (labels.shape != (3, 32, 32) or labels.dtype.kind not in "iu"
                or np.any(labels < 0)):
            raise ValueError(f"{name} must be nonnegative integer labels [3,32,32]")

    records = metadata.get("images")
    if (not isinstance(records, list) or len(records) != len(expected)
            or any(not isinstance(row, dict) for row in records)
            or [row.get("image_id") for row in records] != expected):
        raise ValueError("metadata images must be in fixed bundle order 1320,1321,1322")
    for row in records:
        for arm in ("ours", "slot"):
            metrics = row.get(arm)
            if not isinstance(metrics, dict) or any(
                    key not in metrics or isinstance(metrics[key], bool)
                    or not math.isfinite(float(metrics[key])) for key in METRICS):
                raise ValueError(f"metadata image {row['image_id']} lacks finite {arm} metrics")
    return image_ids.astype(np.int64, copy=False), rgb, gt, ours, slot, metadata


def _label_color(label: int):
    if label == 0:
        return BACKGROUND_COLOR
    hue = (int(label) * PALETTE_GOLDEN_RATIO) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Assign stable display colors by numeric label; color is not semantic matching."""
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("colorize_labels expects nonnegative integer [32,32] labels")
    output = np.empty((*labels.shape, 3), dtype=np.float32)
    for label in np.unique(labels):
        output[labels == label] = _label_color(int(label))
    # 4x nearest-neighbor replication displays exact patch labels without smoothing.
    return np.repeat(np.repeat(output, 4, axis=0), 4, axis=1)


def _metric_text(metrics: dict) -> str:
    return (f"FG-ARI {float(metrics['fg_ari']):.3f}  |  "
            f"FG-IoU {float(metrics['foreground_iou']):.3f}  |  "
            f"Obj-IoU {float(metrics['matched_object_iou']):.3f}")


def _mask_ids(labels: np.ndarray):
    ids = sorted(int(value) for value in np.unique(labels))
    if 0 not in ids:
        ids.insert(0, 0)  # Show the registered background slot even if empty.

    return ids


def _render_mask_cell(ax, labels: np.ndarray, label: int):
    tile = (labels == label).astype(np.uint8)
    ax.imshow(np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1),
              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"ID {label} | {int(tile.sum())} patches", fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.5)


def render_image(image_id, rgb, gt, ours, slot, metrics, output: Path):
    ours_ids, slot_ids = _mask_ids(ours), _mask_ids(slot)
    mask_rows = max((len(ours_ids) + 5) // 6, (len(slot_ids) + 5) // 6)
    fig = plt.figure(figsize=(16, 7.0 + 1.5 * mask_rows), constrained_layout=True)
    grid = fig.add_gridspec(2 + mask_rows, 12,
                            height_ratios=[1.2, 0.16] + [0.55] * mask_rows)
    panels = (
        ("Original RGB", rgb),
        ("Ground truth | native 4x4 modal to 32x32", colorize_labels(gt)),
        ("Our model | SW0097 + SW0143 T", colorize_labels(ours)),
        ("Slot Attention | own 70k, seed0", colorize_labels(slot)),
    )
    for col, (title, image) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col * 3:(col + 1) * 3])
        if col == 0:
            ax.imshow(image, interpolation="nearest")
        else:
            ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for col, title in ((0, "Our model | SW0097 + SW0143 T hard masks"),
                       (6, "Slot Attention | own70k seed0 hard masks")):
        heading = fig.add_subplot(grid[1, col:col + 6])
        heading.axis("off")
        arm = "ours" if col == 0 else "slot"
        heading.set_title(title + "\n" + _metric_text(metrics[arm]),
                          loc="left", fontsize=11, fontweight="bold", pad=1)
    for ids, labels, start_col in ((ours_ids, ours, 0), (slot_ids, slot, 6)):
        for position, label in enumerate(ids):
            row = 2 + position // 6
            col = start_col + position % 6
            _render_mask_cell(fig.add_subplot(grid[row, col]), labels, label)
        for position in range(len(ids), mask_rows * 6):
            row = 2 + position // 6
            col = start_col + position % 6
            fig.add_subplot(grid[row, col]).axis("off")
    fig.suptitle(
        f"Image {image_id} | seed 0 | first three fixed validation IDs, not quality-selected\n"
        "Every label is shown; background is ID 0. Colors encode label numbers only, not matched objects. "
        "Masks are exact 32x32 patches enlarged 4x with nearest-neighbor display.",
        fontsize=12)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_overview(image_ids, rgb, gt, ours, slot, metadata, output: Path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
    column_titles = ("Original RGB", "GT | native 4x4 modal",
                     "Our model | SW0097 + SW0143 T",
                     "Slot Attention | own70k seed0")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10)
    for row, image_id in enumerate(image_ids):
        metric_row = metadata["images"][row]
        panels = (rgb[row], colorize_labels(gt[row]), colorize_labels(ours[row]),
                  colorize_labels(slot[row]))
        for col, image in enumerate(panels):
            axes[row, col].imshow(image, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"ID {int(image_id)}", fontsize=10)
        axes[row, 2].set_xlabel(_metric_text(metric_row["ours"]), fontsize=8)
        axes[row, 3].set_xlabel(_metric_text(metric_row["slot"]), fontsize=8)
    fig.suptitle(
        "Our model (SW0097 + SW0143 T) vs Slot Attention (own70k seed0)\n"
        "First three fixed validation IDs (1320-1322), not quality-selected; "
        "32x32 labels shown with nearest-neighbor enlargement. Colors encode label IDs only, not matched objects.", fontsize=12)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(bundle: Path, metadata_path: Path, output_dir: Path):
    ids, rgb, gt, ours, slot, metadata = _read_inputs(bundle, metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"sw0143_vs_slot32_seed0_id{int(image_id)}.png" for image_id in ids]
    paths.append(output_dir / "sw0143_vs_slot32_seed0_overview.png")
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"preserve existing visualizations: {collisions}")
    for index, image_id in enumerate(ids):
        render_image(int(image_id), rgb[index], gt[index], ours[index], slot[index],
                     metadata["images"][index], paths[index])
    render_overview(ids, rgb, gt, ours, slot, metadata, paths[-1])
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="root-validated NPZ containing exactly the three requested sample IDs")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="JSON with seed0 selection note and existing per-image scores")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = create_visualizations(args.bundle, args.metadata, args.output_dir)
    print(json.dumps({"status": "complete", "outputs": [str(path) for path in paths],
                      "image_ids": list(EXPECTED_IDS), "ground_truth_read_by_script": False,
                      "metrics_recomputed": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
