"""Train the input-layer autoencoder on a subset of server-side images."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from snn_kuramoto_bidirectional.input_layer_generator import CNNAutoEncoder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size):
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNNFeatureEncoder and its decoder on image files."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("server_train/outputs"))
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-kernels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Use CUDA_VISIBLE_DEVICES in bash to select a physical GPU.",
    )
    return parser.parse_args()


def find_image_paths(dataset_dir, num_images, seed):
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    paths = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(paths) < num_images:
        raise ValueError(
            f"Found {len(paths)} images, but --num-images={num_images} was requested."
        )

    random.Random(seed).shuffle(paths)
    return paths[:num_images], len(paths)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    device = torch.device(args.device)

    selected_paths, total_images = find_image_paths(
        args.dataset_dir, args.num_images, args.seed
    )
    dataset = ImagePathDataset(selected_paths, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = CNNAutoEncoder(
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        channels=3,
        bias=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_history = []

    print(f"Dataset directory: {args.dataset_dir}", flush=True)
    print(f"Selected images: {len(selected_paths)} / {total_images}", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"Visible GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        sample_count = 0
        for images in loader:
            images = images.to(device, non_blocking=True)
            reconstruction = model(images)
            loss = F.mse_loss(reconstruction, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            sample_count += images.size(0)

        mean_loss = epoch_loss / sample_count
        loss_history.append(mean_loss)
        print(
            f"Epoch {epoch:04d}/{args.epochs:04d} | loss={mean_loss:.8f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = args.output_dir / "input_layer_encoder.pt"
    decoder_path = args.output_dir / "input_layer_decoder.pt"
    loss_path = args.output_dir / "input_layer_loss.csv"
    config_path = args.output_dir / "input_layer_config.json"
    selected_paths_file = args.output_dir / "selected_images.txt"

    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    with loss_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "mse_loss"])
        writer.writerows(enumerate(loss_history, start=1))
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_dir": str(args.dataset_dir.resolve()),
                "num_images": args.num_images,
                "image_size": args.image_size,
                "num_kernels": args.num_kernels,
                "kernel_size": args.kernel_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
            },
            file,
            indent=2,
        )
    with selected_paths_file.open("w", encoding="utf-8") as file:
        file.write("\n".join(str(path.resolve()) for path in selected_paths))

    print(f"Encoder: {encoder_path}", flush=True)
    print(f"Decoder: {decoder_path}", flush=True)
    print(f"Loss: {loss_path}", flush=True)
    print(f"Config: {config_path}", flush=True)


def validate_args(args):
    positive_integer_fields = (
        "num_images",
        "image_size",
        "num_kernels",
        "kernel_size",
        "epochs",
        "batch_size",
    )
    for name in positive_integer_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.image_size < args.kernel_size:
        raise ValueError("--image-size must be at least --kernel-size.")


def main():
    args = parse_args()
    validate_args(args)
    train(args)


if __name__ == "__main__":
    main()
