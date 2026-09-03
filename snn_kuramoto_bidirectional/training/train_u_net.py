"""Train the shared multi-scale U-Net for binary CLEVR segmentation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.u_net import SharedMultiScaleUNet
except ModuleNotFoundError:
    from u_net import SharedMultiScaleUNet


def load_clevr_binary_segmentation(
    dataset_path,
    *,
    image_key="image",
    mask_key="mask",
    num_images=None,
    image_size=128,
):
    """Load normalized RGB images and combined foreground masks from HDF5."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to load the CLEVR HDF5 file.") from error

    dataset_path = Path(dataset_path)
    with h5py.File(dataset_path, "r") as file:
        available = len(file[image_key])
        count = available if num_images is None else min(int(num_images), available)
        images = torch.from_numpy(file[image_key][:count])
        instance_masks = torch.from_numpy(file[mask_key][:count])

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Images must have shape [B, H, W, 3].")
    if instance_masks.ndim != 4 or instance_masks.shape[-1] != 1:
        raise ValueError("Masks must have shape [B, H, W, 1].")

    images = images.permute(0, 3, 1, 2).float() / 255.0
    masks = (instance_masks.permute(0, 3, 1, 2) > 0).float()

    output_size = (int(image_size), int(image_size))
    if images.shape[-2:] != output_size:
        images = F.interpolate(
            images,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        masks = F.interpolate(masks, size=output_size, mode="nearest")

    return images.contiguous(), masks.contiguous(), available


def binary_dice_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Calculate soft Dice loss from binary-mask logits and targets."""
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
):
    """Return weighted BCE-plus-Dice loss and its detached components."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = binary_dice_loss(logits, target)
    total = float(bce_weight) * bce + float(dice_weight) * dice
    return total, {"bce": bce, "dice": dice, "total": total}


@torch.no_grad()
def binary_metrics(logits: Tensor, target: Tensor, threshold: float = 0.5):
    """Return sample-averaged Dice and IoU for thresholded predictions."""
    prediction = torch.sigmoid(logits) >= float(threshold)
    target = target >= 0.5
    intersection = (prediction & target).sum(dim=(1, 2, 3)).float()
    prediction_area = prediction.sum(dim=(1, 2, 3)).float()
    target_area = target.sum(dim=(1, 2, 3)).float()
    union = (prediction | target).sum(dim=(1, 2, 3)).float()
    dice = (2.0 * intersection + 1e-6) / (
        prediction_area + target_area + 1e-6
    )
    iou = (intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item(), iou.mean().item()


def run_epoch(
    model,
    dataloader,
    *,
    device,
    optimizer=None,
    bce_weight=1.0,
    dice_weight=1.0,
    threshold=0.5,
):
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "dice": 0.0, "iou": 0.0}
    sample_count = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)["segmentation_output"]
            loss, parts = segmentation_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dice, iou = binary_metrics(logits, masks, threshold=threshold)
        batch_size = images.size(0)
        sample_count += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["bce"] += parts["bce"].item() * batch_size
        totals["dice_loss"] += parts["dice"].item() * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size

    return {name: value / sample_count for name, value in totals.items()}


@torch.no_grad()
def save_preview(model, dataset, indices, output_path, device, threshold=0.5):
    """Save RGB inputs, target masks, probabilities, and binary predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    images = torch.stack([dataset[index][0] for index in indices]).to(device)
    targets = torch.stack([dataset[index][1] for index in indices])
    logits = model(images)["segmentation_output"].cpu()
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= float(threshold)

    figure, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
    for row in range(len(indices)):
        axes[row, 0].imshow(images[row].cpu().permute(1, 2, 0).numpy())
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(targets[row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Target foreground")
        axes[row, 2].imshow(probabilities[row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted probability")
        axes[row, 3].imshow(predictions[row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"Prediction >= {threshold:g}")
        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_checkpoint(model, optimizer, epoch, metrics, config, path):
    """Save model state and the complete experiment configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SharedMultiScaleUNet on combined CLEVR foreground masks."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--preview-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, masks, available = load_clevr_binary_segmentation(
        args.dataset_path,
        image_key=args.image_key,
        mask_key=args.mask_key,
        num_images=args.num_images,
        image_size=args.image_size,
    )
    dataset = TensorDataset(images, masks)
    validation_size = max(1, round(len(dataset) * args.validation_fraction))
    training_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SharedMultiScaleUNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = vars(args).copy()
    config.update({
        "available_images": available,
        "selected_images": len(dataset),
        "training_images": training_size,
        "validation_images": validation_size,
        "target_definition": "instance_mask > 0",
        "model": "SharedMultiScaleUNet",
        "smp_decoder": True,
    })
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    history = []
    best_validation_loss = float("inf")
    preview_count = min(args.preview_images, len(validation_dataset))
    preview_indices = list(range(preview_count))

    print(f"Dataset: {args.dataset_path}")
    print(f"Selected images: {len(dataset)} / {available}")
    print(f"Train/validation: {training_size}/{validation_size}")
    print(f"Image shape: {tuple(images.shape)}")
    print(f"Binary mask shape: {tuple(masks.shape)}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            threshold=args.threshold,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device=device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            threshold=args.threshold,
        )
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)

        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:04d}/{args.epochs:04d} | "
            f"train={train_metrics['loss']:.6f} "
            f"val={validation_metrics['loss']:.6f} | "
            f"val_dice={validation_metrics['dice']:.6f} "
            f"val_iou={validation_metrics['iou']:.6f}",
            flush=True,
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
                output_dir / "best.pt",
            )

        if args.preview_every > 0 and (
            epoch % args.preview_every == 0 or epoch == args.epochs
        ):
            save_preview(
                model,
                validation_dataset,
                preview_indices,
                output_dir / "previews" / f"epoch_{epoch:04d}.png",
                device,
                threshold=args.threshold,
            )

    save_checkpoint(
        model,
        optimizer,
        args.epochs,
        history[-1],
        config,
        output_dir / "last.pt",
    )
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Last checkpoint: {output_dir / 'last.pt'}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
