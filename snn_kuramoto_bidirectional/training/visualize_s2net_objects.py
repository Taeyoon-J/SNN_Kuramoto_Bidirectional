import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
    from snn_kuramoto_bidirectional.s2net_cls import S2NetClassifier
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_image_folder
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from s2net_cls import S2NetClassifier
    from training.train_gamma_initializer import load_image_folder


def list_image_paths(image_dir, max_images=None):
    image_dir = Path(image_dir)
    paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if max_images is not None:
        paths = paths[: int(max_images)]
    if not paths:
        raise ValueError(f"No images found in {image_dir}.")
    return paths


def parse_indices(indices, num_samples, dataset_size):
    if indices is not None:
        parsed = [int(value) for value in indices.split(",") if value.strip()]
    else:
        parsed = list(range(int(num_samples)))

    for index in parsed:
        if index < 0 or index >= dataset_size:
            raise ValueError(f"sample index {index} is outside dataset size {dataset_size}.")
    return parsed


def parse_pair_arg(value, name):
    if value is None:
        return None
    if len(value) == 1:
        if value[0] <= 0:
            raise ValueError(f"{name} must be positive.")
        return (int(value[0]), int(value[0]))
    if len(value) == 2:
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError(f"{name} values must be positive.")
        return (first, second)
    raise ValueError(f"{name} must receive one int or two ints.")


def infer_patch_grid_size(num_regions):
    side = int(num_regions ** 0.5)
    if side * side != num_regions:
        raise ValueError(
            "Cannot infer patch grid from num_regions. Pass --patch-grid-size H W."
        )
    return (side, side)


@torch.no_grad()
def predict_groups(model, images, sample_indices, batch_size, device):
    model.eval()
    results = {}
    for start in range(0, len(sample_indices), int(batch_size)):
        indices = sample_indices[start:start + int(batch_size)]
        batch = images[indices].to(device)
        output = model(batch, return_details=True)
        spikes = output.spikes.detach().cpu()
        core_out = output.core_out.detach().cpu()
        sc = output.sc.detach().cpu()
        for local_idx, sample_idx in enumerate(indices):
            results[int(sample_idx)] = {
                "object_groups": [
                    [int(index) for index in group]
                    for group in output.object_groups[local_idx]
                ],
                "spikes": spikes[local_idx],
                "core_out": core_out[local_idx],
                "sc": sc[local_idx],
            }
    return results


def save_sample_figure(image_path, sample_idx, object_groups, spikes, output_path, image_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("RGB").resize((int(image_size), int(image_size)))

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.35, 1.15))

    ax_image = fig.add_subplot(gs[0, 0])
    ax_image.imshow(image)
    ax_image.set_title(f"sample {sample_idx}")
    ax_image.axis("off")

    ax_spikes = fig.add_subplot(gs[0, 1])
    ax_spikes.imshow(spikes.numpy(), aspect="auto", interpolation="nearest", cmap="magma")
    ax_spikes.set_title("spike raster")
    ax_spikes.set_xlabel("time")
    ax_spikes.set_ylabel("oscillator")

    ax_text = fig.add_subplot(gs[0, 2])
    ax_text.axis("off")
    lines = [
        f"file: {image_path.name}",
        f"objects: {len(object_groups)}",
        "",
    ]
    max_groups = 12
    for group_idx, group in enumerate(object_groups[:max_groups]):
        preview = ", ".join(str(index) for index in group[:18])
        if len(group) > 18:
            preview += ", ..."
        lines.append(f"{group_idx}: [{preview}]")
    if len(object_groups) > max_groups:
        lines.append(f"... +{len(object_groups) - max_groups} more")
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=8, family="monospace")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_patch_mask_figure(
    image_path,
    sample_idx,
    object_groups,
    output_path,
    image_size,
    patch_grid_size,
    max_groups=12,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_h, grid_w = patch_grid_size
    image = Image.open(image_path).convert("RGB").resize((int(image_size), int(image_size)))
    groups = object_groups[: int(max_groups)]
    combined = groups_to_mask(object_groups, grid_h, grid_w)

    num_group_plots = max(1, len(groups))
    cols = min(4, num_group_plots)
    rows = (num_group_plots + cols - 1) // cols
    fig = plt.figure(figsize=(12, 4 + rows * 2.4))
    top = fig.add_gridspec(1, 2, left=0.04, right=0.98, top=0.95, bottom=0.58, wspace=0.08)

    ax_image = fig.add_subplot(top[0, 0])
    ax_image.imshow(image)
    ax_image.set_title(f"sample {sample_idx} original")
    ax_image.axis("off")

    ax_combined = fig.add_subplot(top[0, 1])
    ax_combined.imshow(combined, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_combined.set_title("all grouped patches")
    ax_combined.axis("off")

    lower = fig.add_gridspec(
        rows,
        cols,
        left=0.04,
        right=0.98,
        top=0.50,
        bottom=0.05,
        wspace=0.05,
        hspace=0.22,
    )
    for plot_idx in range(rows * cols):
        ax = fig.add_subplot(lower[plot_idx // cols, plot_idx % cols])
        ax.axis("off")
        if plot_idx >= len(groups):
            continue
        mask = groups_to_mask([groups[plot_idx]], grid_h, grid_w)
        ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"group {plot_idx} | {len(groups[plot_idx])} patches", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def groups_to_mask(groups, grid_h, grid_w):
    mask = torch.zeros(int(grid_h), int(grid_w), dtype=torch.float32)
    for group in groups:
        for index in group:
            index = int(index)
            if index < 0 or index >= grid_h * grid_w:
                continue
            row = index // grid_w
            col = index % grid_w
            mask[row, col] = 1.0
    return mask.numpy()


def activity_to_time_masks(activity, patch_grid_size, threshold=0.5):
    grid_h, grid_w = patch_grid_size
    if activity.dim() != 2:
        raise ValueError("activity must have shape [num_oscillators, T].")
    if activity.size(0) != grid_h * grid_w:
        raise ValueError(
            f"activity has {activity.size(0)} oscillators, but grid is {grid_h}x{grid_w}."
        )
    masks = activity.transpose(0, 1).reshape(activity.size(1), grid_h, grid_w)
    return (masks >= float(threshold)).float()


def select_activity(spikes, core_out, source):
    if source == "spikes":
        return spikes.float()
    if source == "membrane":
        return core_out.float()
    if source == "sigmoid_membrane":
        return torch.sigmoid(core_out.float())
    raise ValueError('source must be "spikes", "membrane", or "sigmoid_membrane".')


def aggregate_time_masks(time_masks, mode):
    if mode == "max":
        return time_masks.max(dim=0).values
    if mode == "mean":
        return time_masks.mean(dim=0)
    raise ValueError('mode must be "max" or "mean".')


def connected_components(binary_mask):
    if binary_mask.dim() != 2:
        raise ValueError("binary_mask must have shape [H, W].")
    height, width = binary_mask.shape
    visited = torch.zeros_like(binary_mask, dtype=torch.bool)
    components = []

    for row in range(height):
        for col in range(width):
            if visited[row, col] or binary_mask[row, col] <= 0:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component = []
            while stack:
                cur_row, cur_col = stack.pop()
                component.append((cur_row, cur_col))
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                        continue
                    if visited[next_row, next_col] or binary_mask[next_row, next_col] <= 0:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            components.append(component)
    return sorted(components, key=len, reverse=True)


def component_to_mask(component, grid_h, grid_w):
    mask = torch.zeros(int(grid_h), int(grid_w), dtype=torch.float32)
    for row, col in component:
        mask[int(row), int(col)] = 1.0
    return mask


def resize_mask(mask, image_size):
    mask_image = Image.fromarray((mask.numpy() * 255).astype("uint8"), mode="L")
    nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    mask_image = mask_image.resize((int(image_size), int(image_size)), resample=nearest)
    return torch.as_tensor(list(mask_image.getdata()), dtype=torch.float32).view(image_size, image_size) / 255.0


def masked_original(image, mask, image_size):
    resized_mask = resize_mask(mask, image_size).unsqueeze(-1)
    image_tensor = torch.as_tensor(list(image.getdata()), dtype=torch.float32)
    image_tensor = image_tensor.view(int(image_size), int(image_size), 3) / 255.0
    return image_tensor * resized_mask


def save_spike_pattern_mask_figure(
    image_path,
    sample_idx,
    spikes,
    core_out,
    output_path,
    image_size,
    patch_grid_size,
    activity_source="spikes",
    activity_threshold=0.5,
    time_aggregate="max",
    max_components=8,
    title_prefix="",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_h, grid_w = patch_grid_size
    image = Image.open(image_path).convert("RGB").resize((int(image_size), int(image_size)))
    activity = select_activity(spikes, core_out, activity_source)
    time_masks = activity_to_time_masks(
        activity,
        patch_grid_size=patch_grid_size,
        threshold=activity_threshold,
    )
    aggregate = aggregate_time_masks(time_masks, time_aggregate)
    binary_aggregate = (aggregate >= float(activity_threshold if time_aggregate == "mean" else 0.5)).float()
    components = connected_components(binary_aggregate)

    fig = plt.figure(figsize=(14, 10))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13)
    top = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.95, bottom=0.68, wspace=0.08)
    ax_image = fig.add_subplot(top[0, 0])
    ax_image.imshow(image)
    ax_image.set_title("1. original image")
    ax_image.axis("off")

    ax_mask = fig.add_subplot(top[0, 1])
    ax_mask.imshow(binary_aggregate.numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_mask.set_title(
        f"2. final spatial mask\n{activity_source}, {time_aggregate}, threshold={activity_threshold:g}"
    )
    ax_mask.axis("off")

    ax_recon = fig.add_subplot(top[0, 2])
    ax_recon.imshow(masked_original(image, binary_aggregate, image_size).numpy())
    ax_recon.set_title("3. image x mask")
    ax_recon.axis("off")

    time_count = time_masks.size(0)
    time_cols = min(8, time_count)
    time_rows = (time_count + time_cols - 1) // time_cols
    mid = fig.add_gridspec(
        time_rows,
        time_cols,
        left=0.04,
        right=0.98,
        top=0.61,
        bottom=0.39,
        wspace=0.08,
        hspace=0.25,
    )
    for idx in range(time_rows * time_cols):
        ax = fig.add_subplot(mid[idx // time_cols, idx % time_cols])
        ax.axis("off")
        if idx >= time_count:
            continue
        ax.imshow(time_masks[idx].numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"time {idx}", fontsize=9)

    shown_components = components[: int(max_components)]
    comp_count = max(1, len(shown_components))
    comp_cols = min(4, comp_count)
    comp_rows = (comp_count + comp_cols - 1) // comp_cols
    bottom = fig.add_gridspec(
        comp_rows,
        comp_cols,
        left=0.04,
        right=0.98,
        top=0.31,
        bottom=0.04,
        wspace=0.08,
        hspace=0.25,
    )
    for idx in range(comp_rows * comp_cols):
        ax = fig.add_subplot(bottom[idx // comp_cols, idx % comp_cols])
        ax.axis("off")
        if idx >= len(shown_components):
            continue
        component_mask = component_to_mask(shown_components[idx], grid_h, grid_w)
        ax.imshow(component_mask.numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"object candidate {idx} | {len(shown_components[idx])} patches", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return components


def compute_activity_diagnostics(
    spikes,
    core_out,
    patch_grid_size,
    activity_source,
    activity_threshold,
    time_aggregate,
):
    activity = select_activity(spikes, core_out, activity_source)
    time_masks = activity_to_time_masks(
        activity,
        patch_grid_size=patch_grid_size,
        threshold=activity_threshold,
    )
    aggregate = aggregate_time_masks(time_masks, time_aggregate)
    binary_aggregate = (aggregate >= float(activity_threshold if time_aggregate == "mean" else 0.5)).float()
    components = connected_components(binary_aggregate)
    active_counts_by_time = time_masks.flatten(start_dim=1).sum(dim=1)
    component_sizes = [len(component) for component in components]
    total_patches = int(binary_aggregate.numel())
    active_patches = int(binary_aggregate.sum().item())

    return {
        "active_patches": active_patches,
        "total_patches": total_patches,
        "mask_density": active_patches / total_patches if total_patches else 0.0,
        "num_components": len(components),
        "component_sizes": component_sizes,
        "largest_component_size": max(component_sizes) if component_sizes else 0,
        "active_patches_by_time": [int(value.item()) for value in active_counts_by_time],
        "mean_active_patches_by_time": float(active_counts_by_time.float().mean().item()),
        "min_active_patches_by_time": int(active_counts_by_time.min().item()) if active_counts_by_time.numel() else 0,
        "max_active_patches_by_time": int(active_counts_by_time.max().item()) if active_counts_by_time.numel() else 0,
        "spike_rate": float(spikes.float().mean().item()),
        "activity_mean": float(activity.float().mean().item()),
        "activity_std": float(activity.float().std(unbiased=False).item()),
        "activity_min": float(activity.float().min().item()),
        "activity_max": float(activity.float().max().item()),
        "binary_mask_cells": [
            [int(row), int(col)]
            for row, col in torch.nonzero(binary_aggregate, as_tuple=False).tolist()
        ],
    }


def summarize_diagnostics(sample_summaries):
    if not sample_summaries:
        return {}

    densities = [item["diagnostics"]["mask_density"] for item in sample_summaries]
    active_patches = [item["diagnostics"]["active_patches"] for item in sample_summaries]
    num_components = [item["diagnostics"]["num_components"] for item in sample_summaries]
    largest_components = [item["diagnostics"]["largest_component_size"] for item in sample_summaries]
    spike_rates = [item["diagnostics"]["spike_rate"] for item in sample_summaries]
    masks = [
        {tuple(cell) for cell in item["diagnostics"]["binary_mask_cells"]}
        for item in sample_summaries
    ]
    pairwise_ious = []
    for left_idx in range(len(masks)):
        for right_idx in range(left_idx + 1, len(masks)):
            union = masks[left_idx] | masks[right_idx]
            intersection = masks[left_idx] & masks[right_idx]
            pairwise_ious.append(len(intersection) / len(union) if union else 1.0)

    unique_masks = {frozenset(mask) for mask in masks}
    return {
        "num_samples": len(sample_summaries),
        "unique_binary_masks": len(unique_masks),
        "mean_pairwise_mask_iou": _mean(pairwise_ious),
        "max_pairwise_mask_iou": max(pairwise_ious) if pairwise_ious else None,
        "min_pairwise_mask_iou": min(pairwise_ious) if pairwise_ious else None,
        "mean_mask_density": _mean(densities),
        "min_mask_density": min(densities),
        "max_mask_density": max(densities),
        "mean_active_patches": _mean(active_patches),
        "mean_num_components": _mean(num_components),
        "mean_largest_component_size": _mean(largest_components),
        "mean_spike_rate": _mean(spike_rates),
    }


def save_diagnostics_csv(sample_summaries, output_path):
    fieldnames = [
        "sample_index",
        "active_patches",
        "total_patches",
        "mask_density",
        "num_components",
        "largest_component_size",
        "component_sizes",
        "active_patches_by_time",
        "spike_rate",
        "activity_mean",
        "activity_std",
        "activity_min",
        "activity_max",
        "spike_pattern_mask_figure_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in sample_summaries:
            diagnostics = item["diagnostics"]
            row = {
                "sample_index": item["sample_index"],
                "spike_pattern_mask_figure_path": item["spike_pattern_mask_figure_path"],
            }
            for key in fieldnames:
                if key in diagnostics:
                    value = diagnostics[key]
                    if isinstance(value, list):
                        value = json.dumps(value)
                    row[key] = value
            writer.writerow(row)


def _mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original CLEVR images with S2NetCore detected oscillator groups."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--input-encoder-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--figure-mode", default="compact", choices=["compact", "all"])
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--max-mask-groups", type=int, default=12)
    parser.add_argument("--activity-source", default="spikes", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--time-aggregate", default="max", choices=["max", "mean"])
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-feature-maps", type=int, default=8)
    parser.add_argument("--sc-sigma-color", type=float, default=0.25)
    parser.add_argument("--sc-m-min", type=float, default=0.5)
    parser.add_argument("--sc-self-connectivity", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--spike-classify-method", default="spike_interval", choices=["spike_rhythm", "spike_interval", "spatial_components"])
    parser.add_argument("--spike-rhythm-threshold", type=float, default=0.8)
    parser.add_argument("--spike-rhythm-min-group-size", type=int, default=2)
    parser.add_argument("--spike-rhythm-return-all-groups", action="store_true")
    parser.add_argument("--spike-interval-size", type=int, default=1)
    parser.add_argument("--spike-interval-threshold", type=float, default=0.5)
    parser.add_argument("--spike-interval-min-group-size", type=int, default=1)
    parser.add_argument("--no-spike-interval-include-partial", action="store_true")
    parser.add_argument("--spike-spatial-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--spike-spatial-threshold", type=float, default=None)
    parser.add_argument("--spike-spatial-min-group-size", type=int, default=2)
    parser.add_argument("--spike-spatial-activity-source", default="sigmoid_membrane", choices=["spikes", "membrane", "sigmoid_membrane"])
    parser.add_argument("--spike-spatial-time-aggregate", default="mean", choices=["max", "mean"])
    args = parser.parse_args()

    device = torch.device(args.device)
    image_paths = list_image_paths(args.image_dir, max_images=args.max_images)
    images = load_image_folder(args.image_dir, args.image_size, args.max_images)
    dataset_size = len(image_paths)
    sample_indices = parse_indices(args.sample_indices, args.num_samples, dataset_size)
    patch_grid_size = parse_pair_arg(args.patch_grid_size, "patch-grid-size")
    if patch_grid_size is None:
        patch_grid_size = (8, 8)
    spike_spatial_grid_size = parse_pair_arg(args.spike_spatial_grid_size, "spike-spatial-grid-size")
    if spike_spatial_grid_size is None and args.spike_classify_method == "spatial_components":
        spike_spatial_grid_size = patch_grid_size
    spike_spatial_threshold = (
        args.activity_threshold
        if args.spike_spatial_threshold is None
        else args.spike_spatial_threshold
    )

    hparams = S2NetHyperparameters(
        num_feature_maps=args.num_feature_maps,
        num_regions=patch_grid_size[0] * patch_grid_size[1],
        kernel_size=args.kernel_size,
        gamma_patch_grid_size=patch_grid_size,
        sc_sigma_color=args.sc_sigma_color,
        sc_m_min=args.sc_m_min,
        sc_self_connectivity=args.sc_self_connectivity,
        k=args.k,
        dt=args.dt,
        low_n=args.low_n,
        high_n=args.high_n,
        branch=args.branch,
        spike_classify_method=args.spike_classify_method,
        spike_rhythm_threshold=args.spike_rhythm_threshold,
        spike_rhythm_min_group_size=args.spike_rhythm_min_group_size,
        spike_rhythm_return_all_groups=args.spike_rhythm_return_all_groups,
        spike_interval_size=args.spike_interval_size,
        spike_interval_threshold=args.spike_interval_threshold,
        spike_interval_min_group_size=args.spike_interval_min_group_size,
        spike_interval_include_partial=not args.no_spike_interval_include_partial,
        spike_spatial_grid_size=spike_spatial_grid_size,
        spike_spatial_threshold=spike_spatial_threshold,
        spike_spatial_min_group_size=args.spike_spatial_min_group_size,
        spike_spatial_activity_source=args.spike_spatial_activity_source,
        spike_spatial_time_aggregate=args.spike_spatial_time_aggregate,
    )
    hparams.validate()

    model = S2NetClassifier(hparams, device=device).to(device)
    model.load_input_layer(args.input_encoder_path, map_location=device)
    model.load_core(args.checkpoint_path, map_location=device)

    output_dir = Path(args.output_dir)
    predictions = predict_groups(
        model,
        images,
        sample_indices=sample_indices,
        batch_size=args.batch_size,
        device=device,
    )

    summary = []
    for sample_idx in sample_indices:
        item = predictions[int(sample_idx)]
        output_path = None
        mask_output_path = None
        if args.figure_mode == "all":
            output_path = output_dir / f"s2net_objects_sample_{sample_idx:04d}.png"
            save_sample_figure(
                image_path=image_paths[sample_idx],
                sample_idx=sample_idx,
                object_groups=item["object_groups"],
                spikes=item["spikes"],
                output_path=output_path,
                image_size=args.image_size,
            )
            mask_output_path = output_dir / f"s2net_patch_masks_sample_{sample_idx:04d}.png"
            save_patch_mask_figure(
                image_path=image_paths[sample_idx],
                sample_idx=sample_idx,
                object_groups=item["object_groups"],
                output_path=mask_output_path,
                image_size=args.image_size,
                patch_grid_size=patch_grid_size,
                max_groups=args.max_mask_groups,
            )
            spike_mask_output_path = output_dir / f"s2net_spike_pattern_masks_sample_{sample_idx:04d}.png"
        else:
            spike_mask_output_path = output_dir / f"s2net_sample_{sample_idx:04d}.png"

        components = save_spike_pattern_mask_figure(
            image_path=image_paths[sample_idx],
            sample_idx=sample_idx,
            spikes=item["spikes"],
            core_out=item["core_out"],
            output_path=spike_mask_output_path,
            image_size=args.image_size,
            patch_grid_size=patch_grid_size,
            activity_source=args.activity_source,
            activity_threshold=args.activity_threshold,
            time_aggregate=args.time_aggregate,
            max_components=args.max_components,
            title_prefix=(
                f"S2Net spatial reconstruction | sample {sample_idx} | "
                f"{patch_grid_size[0]}x{patch_grid_size[1]} patch oscillators"
            ),
        )
        diagnostics = compute_activity_diagnostics(
            spikes=item["spikes"],
            core_out=item["core_out"],
            patch_grid_size=patch_grid_size,
            activity_source=args.activity_source,
            activity_threshold=args.activity_threshold,
            time_aggregate=args.time_aggregate,
        )
        summary.append({
            "sample_index": int(sample_idx),
            "image_path": str(image_paths[sample_idx]),
            "figure_path": str(output_path) if output_path is not None else None,
            "mask_figure_path": str(mask_output_path) if mask_output_path is not None else None,
            "spike_pattern_mask_figure_path": str(spike_mask_output_path),
            "spike_pattern_components": [
                [[int(row), int(col)] for row, col in component]
                for component in components
            ],
            "object_groups": item["object_groups"],
            "diagnostics": diagnostics,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s2net_object_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    diagnostics_summary = summarize_diagnostics(summary)
    diagnostics_summary_path = output_dir / "diagnostics_summary.json"
    diagnostics_summary_path.write_text(
        json.dumps(diagnostics_summary, indent=2),
        encoding="utf-8",
    )
    diagnostics_csv_path = output_dir / "sample_diagnostics.csv"
    save_diagnostics_csv(summary, diagnostics_csv_path)
    print(f"saved {len(summary)} visualizations to: {output_dir}")
    print(f"saved summary: {summary_path}")
    print(f"saved diagnostics summary: {diagnostics_summary_path}")
    print(f"saved diagnostics csv: {diagnostics_csv_path}")


if __name__ == "__main__":
    main()
