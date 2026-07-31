import argparse
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
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from s2net_cls import S2NetCore


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
def predict_groups(core, gamma_seq, sample_indices, batch_size, device):
    core.eval()
    results = {}
    for start in range(0, len(sample_indices), int(batch_size)):
        indices = sample_indices[start:start + int(batch_size)]
        batch = gamma_seq[indices].to(device)
        object_groups, spikes = core(batch)
        spikes = spikes.detach().cpu()
        for local_idx, sample_idx in enumerate(indices):
            results[int(sample_idx)] = {
                "object_groups": [
                    [int(index) for index in group]
                    for group in object_groups[local_idx]
                ],
                "spikes": spikes[local_idx],
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original CLEVR images with S2NetCore detected oscillator groups."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--sc-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-indices", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--max-mask-groups", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--low-n", type=float, default=0.0)
    parser.add_argument("--high-n", type=float, default=4.0)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--spike-classify-method", default="spike_interval", choices=["spike_rhythm", "spike_interval"])
    parser.add_argument("--spike-rhythm-threshold", type=float, default=0.8)
    parser.add_argument("--spike-rhythm-min-group-size", type=int, default=2)
    parser.add_argument("--spike-rhythm-return-all-groups", action="store_true")
    parser.add_argument("--spike-interval-size", type=int, default=1)
    parser.add_argument("--spike-interval-threshold", type=float, default=0.5)
    parser.add_argument("--spike-interval-min-group-size", type=int, default=1)
    parser.add_argument("--no-spike-interval-include-partial", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    gamma_seq = torch.load(args.gamma_seq_path, map_location="cpu").float()
    if gamma_seq.dim() != 3:
        raise ValueError(f"gamma_seq must have shape [B, T, N], but got {tuple(gamma_seq.shape)}.")
    sc = torch.load(args.sc_path, map_location="cpu").float()
    image_paths = list_image_paths(args.image_dir, max_images=args.max_images)

    dataset_size = min(len(image_paths), gamma_seq.size(0))
    if dataset_size == 0:
        raise ValueError("No aligned image/gamma samples found.")
    image_paths = image_paths[:dataset_size]
    gamma_seq = gamma_seq[:dataset_size]
    sample_indices = parse_indices(args.sample_indices, args.num_samples, dataset_size)
    patch_grid_size = parse_pair_arg(args.patch_grid_size, "patch-grid-size")
    if patch_grid_size is None:
        patch_grid_size = infer_patch_grid_size(gamma_seq.size(2))

    hparams = S2NetHyperparameters(
        num_feature_maps=gamma_seq.size(1),
        num_regions=gamma_seq.size(2),
        kernel_size=args.kernel_size,
        sc=sc,
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
    )
    hparams.validate()

    core = S2NetCore(hparams, device=device).to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    core.load_state_dict(state_dict)

    output_dir = Path(args.output_dir)
    predictions = predict_groups(
        core,
        gamma_seq,
        sample_indices=sample_indices,
        batch_size=args.batch_size,
        device=device,
    )

    summary = []
    for sample_idx in sample_indices:
        item = predictions[int(sample_idx)]
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
        summary.append({
            "sample_index": int(sample_idx),
            "image_path": str(image_paths[sample_idx]),
            "figure_path": str(output_path),
            "mask_figure_path": str(mask_output_path),
            "object_groups": item["object_groups"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "s2net_object_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {len(summary)} visualizations to: {output_dir}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
