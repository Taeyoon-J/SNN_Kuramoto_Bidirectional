"""Train four independent S2Net cores from frozen multi-scale U-Net features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
    from snn_kuramoto_bidirectional.loss_function import UnsupervisedS2NetLoss
    from snn_kuramoto_bidirectional.s2net_cls import S2NetClassifier
    from snn_kuramoto_bidirectional.training.train_input_layer_generator import (
        load_training_images,
    )
except ModuleNotFoundError:
    from hyperparameter import S2NetHyperparameters
    from loss_function import UnsupervisedS2NetLoss
    from s2net_cls import S2NetClassifier
    from training.train_input_layer_generator import load_training_images


def _unpack_image_batch(batch):
    return batch if torch.is_tensor(batch) else batch[0]


def _select_loss_signal(spikes, membrane, loss_signal):
    if loss_signal == "spikes":
        return spikes
    if loss_signal == "membrane":
        return membrane
    if loss_signal == "sigmoid_membrane":
        return torch.sigmoid(membrane)
    raise ValueError('loss_signal must be "spikes", "membrane", or "sigmoid_membrane".')


def _build_level_criterion(hparams, grid_size, core):
    """Build one loss instance using the existing project hyperparameters."""
    return UnsupervisedS2NetLoss(
        spike_rate_weight=hparams.spike_rate_weight,
        spike_smooth_weight=hparams.spike_smooth_weight,
        spike_diversity_weight=hparams.spike_diversity_weight,
        structural_weight=hparams.structural_weight,
        object_overlap_weight=hparams.object_overlap_weight,
        sample_diversity_weight=hparams.sample_diversity_weight,
        spatial_compactness_weight=hparams.spatial_compactness_weight,
        temporal_balance_weight=hparams.temporal_balance_weight,
        edge_membrane_weight=hparams.edge_membrane_weight,
        edge_membrane_margin=hparams.edge_membrane_margin,
        dense_magnitude_weight=hparams.dense_magnitude_weight,
        dense_magnitude_target=hparams.dense_magnitude_target,
        dense_positive_weight=hparams.dense_positive_weight,
        dense_positive_target=hparams.dense_positive_target,
        dense_positive_temperature=hparams.dense_positive_temperature,
        dendritic_cancellation_weight=hparams.dendritic_cancellation_weight,
        spike_v_th=core.membrane_layer.vth,
        patch_grid_size=grid_size,
    )


def _loss_weights(criterion):
    return {
        "spike_rate": criterion.spike_rate_weight,
        "spike_smooth": criterion.spike_smooth_weight,
        "spike_diversity": criterion.spike_diversity_weight,
        "structural": criterion.structural_weight,
        "object_overlap": criterion.object_overlap_weight,
        "sample_diversity": criterion.sample_diversity_weight,
        "spatial_compactness": criterion.spatial_compactness_weight,
        "temporal_balance": criterion.temporal_balance_weight,
        "edge_membrane": criterion.edge_membrane_weight,
        "dense_magnitude_loss": criterion.dense_magnitude_weight,
        "dense_positive_loss": criterion.dense_positive_weight,
        "dendritic_cancellation_loss": criterion.dendritic_cancellation_weight,
    }


def _save_metrics(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for name in row:
            if name not in columns:
                columns.append(name)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(model, optimizer, epoch, metrics, hparams, unet_path):
    return {
        "epoch": int(epoch),
        "level_cores_state_dict": model.level_cores.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "hparams": asdict(hparams),
        "level_grid_sizes": [list(grid) for grid in model.level_grid_sizes],
        "level_num_regions": list(model.level_num_regions),
        "unet_checkpoint_path": str(Path(unet_path).resolve()),
    }


def train_multilevel_s2net_core(
    model,
    dataloader,
    *,
    hparams,
    unet_checkpoint_path,
    output_dir,
    epochs=100,
    lr=1e-3,
    loss_signal="sigmoid_membrane",
    grad_clip_norm=1.0,
    checkpoint_every=10,
    device=None,
    verbose=False,
):
    """Train each level core with its own grid-aware unsupervised objective."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model = model.to(device)
    model.gamma_generator.set_unet_trainable(False)
    model.gamma_generator.unet.eval()

    criteria = [
        _build_level_criterion(hparams, grid_size, core).to(device)
        for grid_size, core in zip(model.level_grid_sizes, model.level_cores)
    ]
    optimizer = torch.optim.Adam(model.level_cores.parameters(), lr=float(lr))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "epoch_metrics.csv"
    history = []
    best_loss = float("inf")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        model.gamma_generator.unet.eval()
        epoch_sums = {}
        sample_count = 0

        for batch in dataloader:
            images = _unpack_image_batch(batch).to(device)
            output = model(images, return_details=True, classify=False)
            level_losses = []

            for level_index, criterion in enumerate(criteria):
                activity = _select_loss_signal(
                    output.spike_levels[level_index],
                    output.core_out_levels[level_index],
                    loss_signal,
                )
                level_loss, parts = criterion(
                    spikes=activity,
                    object_groups=None,
                    sc=output.sc_levels[level_index],
                    core_out=output.core_out_levels[level_index],
                    images=images,
                    dense_i=output.dense_i_levels[level_index],
                    dendritic_h=output.dendritic_h_levels[level_index],
                )
                level_losses.append(level_loss)

                prefix = f"level_{level_index + 1}"
                for name, value in parts.items():
                    key = f"{prefix}_{name}"
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value.detach()) * images.size(0)

                spikes = output.spike_levels[level_index].detach()
                dense_i = output.dense_i_levels[level_index].detach()
                statistics = {
                    f"{prefix}_actual_spike_rate": float((spikes != 0).float().mean()),
                    f"{prefix}_spikes_per_image": float((spikes != 0).sum()) / images.size(0),
                    f"{prefix}_active_oscillators_per_image": float(
                        (spikes != 0).any(dim=2).sum()
                    ) / images.size(0),
                    f"{prefix}_mean_dense": float(dense_i.mean()),
                    f"{prefix}_absolute_mean_dense": float(dense_i.abs().mean()),
                }
                for key, value in statistics.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + value * images.size(0)

            # Summation preserves the single-core loss scale for every independent core.
            total_loss = torch.stack(level_losses).sum()
            optimizer.zero_grad()
            total_loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                for core in model.level_cores:
                    torch.nn.utils.clip_grad_norm_(
                        core.parameters(),
                        float(grad_clip_norm),
                    )
            optimizer.step()

            epoch_sums["total_loss"] = epoch_sums.get("total_loss", 0.0) + float(total_loss.detach()) * images.size(0)
            sample_count += images.size(0)

        row = {"epoch": epoch}
        row.update({name: value / sample_count for name, value in epoch_sums.items()})
        for level_index, criterion in enumerate(criteria, start=1):
            for name, weight in _loss_weights(criterion).items():
                raw_key = f"level_{level_index}_{name}"
                if raw_key in row:
                    row[f"level_{level_index}_weighted_{name}"] = row[raw_key] * weight
        history.append(row)
        _save_metrics(history, metrics_path)

        checkpoint = _checkpoint(
            model, optimizer, epoch, row, hparams, unet_checkpoint_path
        )
        torch.save(checkpoint, output_dir / "last.pt")
        if row["total_loss"] < best_loss:
            best_loss = row["total_loss"]
            torch.save(checkpoint, output_dir / "best.pt")
        if checkpoint_every > 0 and epoch % int(checkpoint_every) == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")

        if verbose:
            level_text = " ".join(
                f"L{index}={row[f'level_{index}_total']:.6f}"
                for index in range(1, len(model.level_cores) + 1)
            )
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | "
                f"total={row['total_loss']:.8f} | {level_text}",
                flush=True,
            )

    return model, history


def _parse_args():
    defaults = S2NetHyperparameters()
    parser = argparse.ArgumentParser(
        description="Train four independent U-Net multi-level S2Net cores."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir")
    source.add_argument("--dataset-path")
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--unet-checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--num-feature-maps", type=int, default=defaults.num_feature_maps)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loss-signal",
        choices=("spikes", "membrane", "sigmoid_membrane"),
        default="sigmoid_membrane",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    images = load_training_images(
        image_dir=args.image_dir,
        dataset_path=args.dataset_path,
        hdf5_key=args.hdf5_key,
        image_size=args.image_size,
        max_images=args.max_images,
    ).float()
    dataloader = DataLoader(
        TensorDataset(images),
        batch_size=min(int(args.batch_size), len(images)),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    hparams = S2NetHyperparameters(num_feature_maps=args.num_feature_maps).validate()
    device = torch.device(args.device)
    model = S2NetClassifier(hparams, device=device).to(device)
    model.load_unet(
        args.unet_checkpoint_path,
        map_location=device,
        trainable=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["hparams"] = asdict(hparams)
    config["level_grid_sizes"] = [list(grid) for grid in model.level_grid_sizes]
    config["level_num_regions"] = list(model.level_num_regions)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)

    print(f"Images: {tuple(images.shape)}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"U-Net checkpoint: {args.unet_checkpoint_path}", flush=True)
    print(f"Level grids: {model.level_grid_sizes}", flush=True)
    print(f"Level oscillators: {model.level_num_regions}", flush=True)
    print(f"Loss signal: {args.loss_signal}", flush=True)

    train_multilevel_s2net_core(
        model,
        dataloader,
        hparams=hparams,
        unet_checkpoint_path=args.unet_checkpoint_path,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        loss_signal=args.loss_signal,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_every=args.checkpoint_every,
        device=device,
        verbose=args.verbose,
    )
    print(f"Best checkpoint: {output_dir / 'best.pt'}", flush=True)
    print(f"Last checkpoint: {output_dir / 'last.pt'}", flush=True)
    print(f"Epoch metrics: {output_dir / 'epoch_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
