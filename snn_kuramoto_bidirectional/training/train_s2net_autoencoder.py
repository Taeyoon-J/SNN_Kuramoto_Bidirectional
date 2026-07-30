"""Train the complete S2Net object-centric autoencoder end to end.

Example:
    python -m snn_kuramoto_bidirectional.training.train_s2net_autoencoder \
        --dataset-path /path/to/clevr_10-full.hdf5 \
        --output-dir /path/to/output \
        --num-images 500 \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import h5py
import torch
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ..hyperparameter import DEFAULT_HYPERPARAMETERS, S2NetHyperparameters
from ..loss_function import s2net_total_loss
from ..s2net_cls import S2NetClassifier, S2NetOutput


class HDF5ImageDataset(Dataset):
    """Load one contiguous HDF5 image subset into host memory."""

    def __init__(
        self,
        path: Path,
        dataset_key: str,
        start_index: int,
        num_images: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.dataset_key = str(dataset_key)
        self.start_index = int(start_index)
        self.num_images = int(num_images)
        self.image_size = int(image_size)

        with h5py.File(self.path, "r") as file:
            data = file[self.dataset_key][
                self.start_index : self.start_index + self.num_images
            ]

        images = torch.from_numpy(data)
        if images.ndim != 4:
            raise ValueError(
                f"Expected four-dimensional image data, got {images.shape}."
            )

        # HDF5 CLEVR images are normally [N,H,W,C].
        if images.shape[-1] in {1, 3, 4}:
            if images.shape[-1] == 1:
                images = images.repeat(1, 1, 1, 3)
            elif images.shape[-1] == 4:
                images = images[..., :3]
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] not in {1, 3, 4}:
            raise ValueError(
                "Images must have channels in the first or final dimension."
            )
        elif images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] == 4:
            images = images[:, :3]

        images = images.contiguous().float()
        if data.dtype.kind in {"u", "i"}:
            images.div_(255.0)
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        self.images = images.clamp_(0.0, 1.0)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.images[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the complete S2Net reconstruction model."
    )

    # Dataset and outputs
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)

    # Model dimensions
    parser.add_argument(
        "--num-feature-maps",
        type=int,
        default=DEFAULT_HYPERPARAMETERS.num_feature_maps,
    )
    parser.add_argument(
        "--num-oscillators",
        type=int,
        default=DEFAULT_HYPERPARAMETERS.num_regions,
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=None,
        help="Number of soft object vectors. Defaults to num-feature-maps.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=DEFAULT_HYPERPARAMETERS.kernel_size,
    )
    parser.add_argument(
        "--gamma-dropout",
        type=float,
        default=DEFAULT_HYPERPARAMETERS.gamma_dropout,
    )

    # Kuramoto and SNN
    parser.add_argument(
        "--kuramoto-k",
        type=float,
        default=DEFAULT_HYPERPARAMETERS.k,
    )
    parser.add_argument(
        "--kuramoto-dt",
        type=float,
        default=DEFAULT_HYPERPARAMETERS.dt,
    )
    parser.add_argument(
        "--dendritic-low",
        type=float,
        default=DEFAULT_HYPERPARAMETERS.low_n,
    )
    parser.add_argument(
        "--dendritic-high",
        type=float,
        default=DEFAULT_HYPERPARAMETERS.high_n,
    )
    parser.add_argument(
        "--dendritic-branches",
        type=int,
        default=DEFAULT_HYPERPARAMETERS.branch,
    )

    # Dynamic SC
    parser.add_argument("--sc-momentum", type=float, default=0.99)
    parser.add_argument("--sc-eps", type=float, default=1e-8)

    # Decoder
    parser.add_argument(
        "--decoder-broadcast-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--decoder-hidden-channels",
        type=int,
        nargs="+",
        default=(64, 64, 64, 64, 64),
    )

    # Optimization and runtime
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Set to 0 to disable clipping.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Use CUDA_VISIBLE_DEVICES in bash to select a physical GPU.",
    )

    # Saving and resuming
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint produced by this script from which to resume.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "num_images",
        "image_size",
        "num_feature_maps",
        "num_oscillators",
        "kernel_size",
        "dendritic_branches",
        "decoder_broadcast_size",
        "epochs",
        "batch_size",
        "checkpoint_every",
        "preview_every",
        "preview_images",
    )
    for name in positive_integer_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")

    if args.num_objects is not None:
        if args.num_objects <= 0:
            raise ValueError("--num-objects must be positive.")
        if args.num_objects > args.num_oscillators:
            raise ValueError(
                "--num-objects cannot exceed --num-oscillators because "
                "classifier centers are initialized from oscillator "
                "embeddings."
            )
    if args.image_size < args.kernel_size:
        raise ValueError("--image-size must be at least --kernel-size.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1).")
    if not 0.0 <= args.gamma_dropout < 1.0:
        raise ValueError("--gamma-dropout must be in [0, 1).")
    if not 0.0 <= args.sc_momentum < 1.0:
        raise ValueError("--sc-momentum must be in [0, 1).")
    if args.sc_eps <= 0.0:
        raise ValueError("--sc-eps must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative.")
    if args.gradient_clip < 0.0:
        raise ValueError("--gradient-clip must be non-negative.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if any(channel <= 0 for channel in args.decoder_hidden_channels):
        raise ValueError("--decoder-hidden-channels must be positive.")
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")


def inspect_hdf5(
    path: Path,
    dataset_key: str,
    num_images: int,
    seed: int,
) -> tuple[int, int]:
    with h5py.File(path, "r") as file:
        if dataset_key not in file:
            raise KeyError(f"HDF5 key '{dataset_key}' was not found in {path}.")
        dataset = file[dataset_key]
        if dataset.ndim != 4:
            raise ValueError(
                f"HDF5 images must have four dimensions, got {dataset.shape}."
            )
        total_images = int(dataset.shape[0])

    if total_images < num_images:
        raise ValueError(
            f"Dataset contains {total_images} images, "
            f"but {num_images} were requested."
        )
    start_index = random.Random(seed).randrange(total_images - num_images + 1)
    return total_images, start_index


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, int, int]:
    total_images, start_index = inspect_hdf5(
        args.dataset_path,
        args.hdf5_key,
        args.num_images,
        args.seed,
    )
    dataset = HDF5ImageDataset(
        path=args.dataset_path,
        dataset_key=args.hdf5_key,
        start_index=start_index,
        num_images=args.num_images,
        image_size=args.image_size,
    )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    num_validation = max(
        1,
        int(round(len(dataset) * args.validation_fraction)),
    )
    if num_validation >= len(dataset):
        raise ValueError(
            "Training split is empty. Increase --num-images or reduce "
            "--validation-fraction."
        )
    validation_indices = indices[:num_validation]
    training_indices = indices[num_validation:]

    common_loader_args = {
        "batch_size": min(args.batch_size, len(training_indices)),
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    training_loader = DataLoader(
        Subset(dataset, training_indices),
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        batch_size=min(args.batch_size, len(validation_indices)),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return training_loader, validation_loader, total_images, start_index


def build_model(args: argparse.Namespace, device: torch.device) -> S2NetClassifier:
    hparams = S2NetHyperparameters(
        num_feature_maps=args.num_feature_maps,
        num_regions=args.num_oscillators,
        sc=None,
        in_channels=3,
        kernel_size=args.kernel_size,
        gamma_dropout=args.gamma_dropout,
        k=args.kuramoto_k,
        dt=args.kuramoto_dt,
        low_n=args.dendritic_low,
        high_n=args.dendritic_high,
        branch=args.dendritic_branches,
    )
    return S2NetClassifier(
        hparams=hparams,
        device=device,
        num_objects=args.num_objects,
        image_size=(args.image_size, args.image_size),
        decoder_broadcast_size=(
            args.decoder_broadcast_size,
            args.decoder_broadcast_size,
        ),
        decoder_hidden_channels=tuple(args.decoder_hidden_channels),
        rgb_activation="sigmoid",
        sc_momentum=args.sc_momentum,
        sc_eps=args.sc_eps,
    )


def train_one_epoch(
    model: S2NetClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        loss = s2net_total_loss(
            reconstruction=output.reconstruction,
            target_image=images,
            masks=output.masks,
            membrane_history=output.membrane,
            object_vectors=output.object_vectors,
            hparams=model.hparams,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        loss_sum += loss.detach().item() * images.shape[0]
        sample_count += images.shape[0]

    return loss_sum / sample_count


@torch.no_grad()
def evaluate(
    model: S2NetClassifier,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        loss = s2net_total_loss(
            reconstruction=output.reconstruction,
            target_image=images,
            masks=output.masks,
            membrane_history=output.membrane,
            object_vectors=output.object_vectors,
            hparams=model.hparams,
        )
        loss_sum += loss.item() * images.shape[0]
        sample_count += images.shape[0]

    return loss_sum / sample_count


@torch.no_grad()
def save_preview(
    model: S2NetClassifier,
    loader: DataLoader,
    device: torch.device,
    path: Path,
    max_images: int,
) -> None:
    model.eval()
    images = next(iter(loader))[:max_images].to(device)
    output = model(images)

    rows = []
    for index in range(images.shape[0]):
        original = _tensor_to_pil(images[index])
        reconstruction = _tensor_to_pil(output.reconstruction[index])
        object_tiles = [
            _tensor_to_pil(
                output.object_rgb[index, object_index]
                * output.masks[index, object_index]
            )
            for object_index in range(output.object_rgb.shape[1])
        ]
        mask_tiles = [
            _tensor_to_pil(
                output.masks[index, object_index].expand(3, -1, -1)
            )
            for object_index in range(output.masks.shape[1])
        ]
        rows.append(
            _horizontal_sheet(
                [
                    original,
                    reconstruction,
                    *object_tiles,
                    *mask_tiles,
                ]
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    _vertical_sheet(rows).save(path)


def _tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().float().clamp(0.0, 1.0).cpu()
    array = (
        image.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _horizontal_sheet(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        sheet.paste(image, (offset, 0))
        offset += image.width
    return sheet


def _vertical_sheet(images: list[Image.Image]) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        sheet.paste(image, (0, offset))
        offset += image.height
    return sheet


def save_checkpoint(
    path: Path,
    model: S2NetClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, float]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": float(best_validation_loss),
            "history": history,
            "config": _json_config(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: S2NetClassifier,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_validation_loss"]),
        list(checkpoint.get("history", [])),
    )


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "train_loss", "validation_loss"),
        )
        writer.writeheader()
        writer.writerows(history)


def _json_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
        elif isinstance(value, tuple):
            config[key] = list(value)
    return config


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, total_images, start_index = build_loaders(
        args,
        device,
    )
    model = build_model(args, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []
    if args.resume is not None:
        start_epoch, best_validation_loss, history = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
        )

    config_path = args.output_dir / "config.json"
    config_path.write_text(
        json.dumps(_json_config(args), indent=2),
        encoding="utf-8",
    )

    print(f"Dataset: {args.dataset_path}", flush=True)
    print(f"Selected images: {args.num_images} / {total_images}", flush=True)
    print(
        f"Selected HDF5 range: [{start_index}, "
        f"{start_index + args.num_images})",
        flush=True,
    )
    print(
        f"Train/validation: "
        f"{len(train_loader.dataset)}/{len(validation_loader.dataset)}",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Objects: {model.num_objects} | "
        f"Oscillators: {model.num_oscillators}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.gradient_clip,
        )
        validation_loss = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        write_history(args.output_dir / "loss.csv", history)

        print(
            f"Epoch {epoch:04d}/{args.epochs:04d} | "
            f"train={train_loss:.8f} | "
            f"validation={validation_loss:.8f}",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_validation_loss,
                history,
                args,
            )

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                best_validation_loss,
                history,
                args,
            )
        if epoch % args.preview_every == 0:
            save_preview(
                model,
                validation_loader,
                device,
                args.output_dir / "previews" / f"epoch_{epoch:04d}.png",
                args.preview_images,
            )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_validation_loss,
            history,
            args,
        )

    print(f"Best checkpoint: {args.output_dir / 'best.pt'}", flush=True)
    print(f"Last checkpoint: {args.output_dir / 'last.pt'}", flush=True)
    print(f"Loss history: {args.output_dir / 'loss.csv'}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
