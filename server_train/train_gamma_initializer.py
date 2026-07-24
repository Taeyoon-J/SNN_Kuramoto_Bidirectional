"""Train the gamma initializer from pretrained input-layer feature maps."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from server_train.train_input_layer import build_dataset
from snn_kuramoto_bidirectional.input_layer_generator import CNNFeatureEncoder
from snn_kuramoto_bidirectional.training.train_gamma_initializer import (
    train_gamma_initializer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create feature maps and train the gamma initializer."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--input-encoder-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-kernels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-osci", type=int, default=90)
    parser.add_argument(
        "--hidden-channels",
        type=int,
        nargs="+",
        default=(16, 32, 64),
        help="Deprecated compatibility option; ignored by the direct-linear encoder.",
    )
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def validate_args(args):
    positive_fields = (
        "num_images",
        "image_size",
        "num_kernels",
        "kernel_size",
        "num_osci",
        "decoder_hidden_dim",
        "epochs",
        "batch_size",
        "feature_batch_size",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if any(channel <= 0 for channel in args.hidden_channels):
        raise ValueError("--hidden-channels values must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("--dropout must be in [0, 1).")
    if not args.input_encoder_path.is_file():
        raise FileNotFoundError(
            f"Input-layer encoder checkpoint not found: {args.input_encoder_path}"
        )


@torch.no_grad()
def create_feature_maps(args, device):
    dataset, total_images, selected_samples = build_dataset(
        args.dataset_path,
        args.num_images,
        args.image_size,
        args.seed,
        args.hdf5_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(args.feature_batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    encoder = CNNFeatureEncoder(
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        in_channels=3,
        bias=True,
    ).to(device)
    state_dict = torch.load(
        args.input_encoder_path,
        map_location=device,
        weights_only=True,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()

    batches = []
    for images in loader:
        batches.append(encoder(images.to(device, non_blocking=True)).cpu())
    feature_maps = torch.cat(batches, dim=0)
    return feature_maps, total_images, selected_samples


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    device = torch.device(args.device)

    print(f"Dataset: {args.dataset_path}", flush=True)
    print(f"Input encoder: {args.input_encoder_path}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}", flush=True)

    feature_maps, total_images, selected_samples = create_feature_maps(args, device)
    print(f"Selected images: {feature_maps.size(0)} / {total_images}", flush=True)
    print(f"Feature maps: {tuple(feature_maps.shape)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / "gamma_initializer_encoder.pt"
    decoder_path = args.output_dir / "gamma_initializer_decoder.pt"

    trained_encoder, _, loss_history = train_gamma_initializer(
        feature_maps=feature_maps,
        num_osci=args.num_osci,
        hidden_channels=tuple(args.hidden_channels),
        decoder_hidden_dim=args.decoder_hidden_dim,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=device,
        save_path=encoder_path,
        decoder_save_path=decoder_path,
        verbose=True,
    )

    loss_path = args.output_dir / "gamma_initializer_loss.csv"
    config_path = args.output_dir / "gamma_initializer_config.json"
    selected_path = args.output_dir / "selected_images.txt"
    gamma_sequences_path = args.output_dir / "gamma_sequences.pt"

    trained_encoder.eval()
    gamma_batches = []
    flat_feature_maps = feature_maps.reshape(-1, 1, *feature_maps.shape[-2:])
    with torch.no_grad():
        for start in range(0, flat_feature_maps.size(0), 256):
            batch = flat_feature_maps[start : start + 256].to(device)
            gamma_batches.append(trained_encoder(batch).cpu())
    gamma_sequences = torch.cat(gamma_batches, dim=0).reshape(
        feature_maps.size(0), feature_maps.size(1), args.num_osci
    )
    torch.save(gamma_sequences, gamma_sequences_path)
    with loss_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "mse_loss"])
        writer.writerows(enumerate(loss_history, start=1))
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_path": str(args.dataset_path.resolve()),
                "hdf5_key": args.hdf5_key,
                "input_encoder_path": str(args.input_encoder_path.resolve()),
                "num_images": args.num_images,
                "image_size": args.image_size,
                "num_kernels": args.num_kernels,
                "kernel_size": args.kernel_size,
                "num_osci": args.num_osci,
                "gamma_encoder_type": "direct_linear",
                "hidden_channels": list(args.hidden_channels),
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "feature_map_shape": list(feature_maps.shape),
                "gamma_sequence_shape": list(gamma_sequences.shape),
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
            },
            file,
            indent=2,
        )
    selected_path.write_text("\n".join(selected_samples), encoding="utf-8")

    print(f"Encoder: {encoder_path}", flush=True)
    print(f"Decoder: {decoder_path}", flush=True)
    print(f"Loss: {loss_path}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Gamma sequences: {gamma_sequences_path}", flush=True)


def main():
    args = parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
