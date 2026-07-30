"""Visualize outputs from a fully trained S2Net checkpoint.

This module is intentionally separate from the training loop. It loads a
finished checkpoint, runs inference on selected dataset images, and shows the
original image, reconstruction, normalized masks, and normalized masked RGB
components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch
from PIL import Image, ImageDraw
from torch import Tensor
import torch.nn.functional as F

from .hyperparameter import S2NetHyperparameters
from .s2net_cls import S2NetClassifier, S2NetOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained S2Net checkpoint after training."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Inference seed. Defaults to the training checkpoint seed.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical stability value for display-only normalization.",
    )
    return parser.parse_args()


def _load_images(
    path: Path,
    dataset_key: str,
    start_index: int,
    num_images: int,
    image_size: tuple[int, int],
) -> Tensor:
    with h5py.File(path, "r") as file:
        data = file[dataset_key][start_index : start_index + num_images]

    images = torch.from_numpy(data)
    if images.shape[-1] in {1, 3, 4}:
        if images.shape[-1] == 1:
            images = images.repeat(1, 1, 1, 3)
        elif images.shape[-1] == 4:
            images = images[..., :3]
        images = images.permute(0, 3, 1, 2)
    elif images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] == 4:
        images = images[:, :3]

    images = images.contiguous().float()
    if data.dtype.kind in {"u", "i"}:
        images = images / 255.0
    if tuple(images.shape[-2:]) != image_size:
        images = F.interpolate(
            images,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
    return images.clamp(0.0, 1.0)


def _build_model(
    config: dict,
    device: torch.device,
) -> S2NetClassifier:
    num_feature_maps = int(config["num_feature_maps"])
    num_oscillators = int(config["num_oscillators"])
    image_size = int(config["image_size"])

    hparams = S2NetHyperparameters(
        num_feature_maps=num_feature_maps,
        num_regions=num_oscillators,
        sc=None,
        in_channels=3,
        kernel_size=int(config["kernel_size"]),
        gamma_dropout=float(config["gamma_dropout"]),
        k=float(config["kuramoto_k"]),
        dt=float(config["kuramoto_dt"]),
        low_n=float(config["dendritic_low"]),
        high_n=float(config["dendritic_high"]),
        branch=int(config["dendritic_branches"]),
    )
    return S2NetClassifier(
        hparams=hparams,
        device=device,
        num_objects=config.get("num_objects"),
        image_size=(image_size, image_size),
        decoder_broadcast_size=(
            int(config["decoder_broadcast_size"]),
            int(config["decoder_broadcast_size"]),
        ),
        decoder_hidden_channels=tuple(config["decoder_hidden_channels"]),
        rgb_activation="sigmoid",
        sc_momentum=float(config["sc_momentum"]),
        sc_eps=float(config["sc_eps"]),
    )


def _display_outputs(
    output: S2NetOutput,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Return display-brightened masks and masked RGB components.

    Each mask is divided by its own spatial maximum. The same factor is used
    for its masked RGB component. This removes the global 1/K dimming while
    preserving each mask's relative spatial structure. These tensors are only
    used for visualization and never alter model outputs.
    """

    mask_max = output.masks.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    display_masks = output.masks / mask_max
    display_objects = output.masks * output.object_rgb / mask_max
    return display_masks, display_objects


def _tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().float().clamp(0.0, 1.0).cpu()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    array = (
        image.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _labeled_tile(image: Image.Image, label: str) -> Image.Image:
    label_height = 34
    tile = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        "white",
    )
    tile.paste(image, (0, label_height))
    ImageDraw.Draw(tile).text((4, 9), label, fill="black")
    return tile


def _horizontal_sheet(
    images: list[Image.Image],
    width: int | None = None,
) -> Image.Image:
    sheet_width = width or sum(image.width for image in images)
    sheet_height = max(image.height for image in images)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet


def _sample_sheet(
    original: Tensor,
    output: S2NetOutput,
    display_masks: Tensor,
    display_objects: Tensor,
    sample_index: int,
    dataset_index: int,
) -> Image.Image:
    num_objects = output.masks.shape[1]
    tile_width = original.shape[-1]
    full_width = num_objects * tile_width

    summary = _horizontal_sheet(
        [
            _labeled_tile(_tensor_to_pil(original), "Original"),
            _labeled_tile(
                _tensor_to_pil(output.reconstruction[sample_index]),
                "Reconstruction",
            ),
        ],
        width=full_width,
    )

    mask_tiles = []
    object_tiles = []
    for object_index in range(num_objects):
        mask = output.masks[sample_index, object_index]
        statistics = (
            f"M{object_index}: "
            f"{mask.min().item():.3f}/"
            f"{mask.mean().item():.3f}/"
            f"{mask.max().item():.3f}"
        )
        mask_tiles.append(
            _labeled_tile(
                _tensor_to_pil(
                    display_masks[sample_index, object_index]
                ),
                statistics,
            )
        )
        object_tiles.append(
            _labeled_tile(
                _tensor_to_pil(
                    display_objects[sample_index, object_index]
                ),
                f"Mask {object_index} * RGB",
            )
        )

    masks = _horizontal_sheet(mask_tiles, width=full_width)
    objects = _horizontal_sheet(object_tiles, width=full_width)
    title_height = 34
    sheet = Image.new(
        "RGB",
        (
            full_width,
            title_height + summary.height + masks.height + objects.height,
        ),
        "white",
    )
    ImageDraw.Draw(sheet).text(
        (4, 9),
        f"Dataset image {dataset_index} | "
        "mask labels show raw min/mean/max; images are display-normalized",
        fill="black",
    )
    y = title_height
    for row in (summary, masks, objects):
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


@torch.no_grad()
def visualize(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    device = torch.device(args.device)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = _build_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(config["image_size"])
    images = _load_images(
        path=args.dataset_path,
        dataset_key=args.hdf5_key,
        start_index=args.start_index,
        num_images=args.num_images,
        image_size=(image_size, image_size),
    ).to(device)
    output = model(images)
    display_masks, display_objects = _display_outputs(output, args.eps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_sheets = []
    for sample_index in range(images.shape[0]):
        dataset_index = args.start_index + sample_index
        sheet = _sample_sheet(
            original=images[sample_index],
            output=output,
            display_masks=display_masks,
            display_objects=display_objects,
            sample_index=sample_index,
            dataset_index=dataset_index,
        )
        sheet.save(
            args.output_dir / f"sample_{dataset_index:06d}.png"
        )
        sample_sheets.append(sheet)

    combined_width = max(sheet.width for sheet in sample_sheets)
    combined_height = sum(sheet.height for sheet in sample_sheets)
    combined = Image.new(
        "RGB",
        (combined_width, combined_height),
        "white",
    )
    y = 0
    for sheet in sample_sheets:
        combined.paste(sheet, (0, y))
        y += sheet.height
    combined_path = args.output_dir / "all_samples.png"
    combined.save(combined_path)

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Images: {images.shape[0]}")
    print(f"Inference seed: {seed}")
    print(f"Visualization: {combined_path}")


def main() -> None:
    args = parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
