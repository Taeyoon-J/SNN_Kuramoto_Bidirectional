"""Generate spatial patch gamma sequences from pretrained feature maps."""

import argparse
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
    from snn_kuramoto_bidirectional.gamma_initializer import (
        feature_maps_to_patch_gamma,
    )
    from snn_kuramoto_bidirectional.training.train_input_layer_generator import (
        load_input_encoder,
    )
except ModuleNotFoundError:
    from gamma_initializer import feature_maps_to_patch_gamma
    from training.train_input_layer_generator import load_input_encoder


def load_image_folder(image_dir, image_size=128, max_images=None):
    """Load RGB images from a folder into a [B, 3, H, W] float tensor."""
    image_dir = Path(image_dir)
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if max_images is not None:
        paths = paths[: int(max_images)]
    if not paths:
        raise ValueError(f"No images found in {image_dir}.")

    size = (int(image_size), int(image_size))
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize(size)
        tensor = torch.as_tensor(list(image.getdata()), dtype=torch.float32)
        tensor = tensor.view(size[1], size[0], 3).permute(2, 0, 1) / 255.0
        images.append(tensor)
    return torch.stack(images)


@torch.no_grad()
def images_to_feature_maps(
    images,
    input_encoder_checkpoint,
    num_kernels=8,
    kernel_size=3,
    channels=3,
    batch_size=32,
    device=None,
):
    """Convert image tensors [B, 3, H, W] to [B, T, H', W']."""
    if not torch.is_tensor(images):
        images = torch.as_tensor(images, dtype=torch.float32)
    images = images.float()
    device = torch.device(device) if device is not None else images.device

    encoder = load_input_encoder(
        input_encoder_checkpoint,
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        channels=channels,
        device=device,
    )
    encoder.eval()

    feature_maps = []
    for start in range(0, images.size(0), int(batch_size)):
        batch = images[start : start + int(batch_size)].to(device)
        feature_maps.append(encoder(batch).cpu())
    return torch.cat(feature_maps, dim=0)


def normalize_feature_maps(feature_maps, mode="none", clip=None, eps=1e-6):
    """Normalize feature-map values before patch pooling."""
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.as_tensor(feature_maps, dtype=torch.float32)
    feature_maps = feature_maps.float()

    mode = str(mode)
    if mode == "none":
        normalized = feature_maps
        stats = {"mode": mode}
    elif mode == "standardize":
        mean = feature_maps.mean()
        std = feature_maps.std(unbiased=False).clamp_min(float(eps))
        normalized = (feature_maps - mean) / std
        stats = {
            "mode": mode,
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        }
    elif mode == "per_map_standardize":
        mean = feature_maps.mean(dim=(-2, -1), keepdim=True)
        std = feature_maps.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        ).clamp_min(float(eps))
        normalized = (feature_maps - mean) / std
        stats = {"mode": mode}
    elif mode == "minmax":
        min_value = feature_maps.amin()
        max_value = feature_maps.amax()
        normalized = (feature_maps - min_value) / (
            max_value - min_value + float(eps)
        )
        stats = {
            "mode": mode,
            "min": min_value.detach().cpu(),
            "max": max_value.detach().cpu(),
        }
    else:
        raise ValueError(
            'mode must be "none", "standardize", '
            '"per_map_standardize", or "minmax".'
        )

    if clip is not None:
        normalized = normalized.clamp(-float(clip), float(clip))
        stats["clip"] = float(clip)
    return normalized, stats


def save_preprocessing_stats(stats, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, save_path)


def save_patch_gamma_config(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(config, save_path)


def _parse_pair_arg(value, name):
    if value is None:
        return None
    if len(value) == 1:
        if value[0] <= 0:
            raise ValueError(f"{name} must be positive.")
        return int(value[0])
    if len(value) == 2:
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError(f"{name} values must be positive.")
        return (first, second)
    raise ValueError(f"{name} must receive one int or two ints.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate patch gamma from pretrained image features."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--input-encoder-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--gamma-seq-save-path", required=True)
    parser.add_argument("--preprocess-save-path", default=None)
    parser.add_argument("--num-kernels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--patch-size", type=int, nargs="+", default=None)
    parser.add_argument("--patch-stride", type=int, nargs="+", default=None)
    parser.add_argument(
        "--patch-reduction",
        default="mean",
        choices=["mean", "max"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--feature-normalize",
        default="none",
        choices=["none", "standardize", "per_map_standardize", "minmax"],
    )
    parser.add_argument("--feature-clip", type=float, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    grid_size = _parse_pair_arg(args.patch_grid_size, "patch-grid-size")
    patch_size = _parse_pair_arg(args.patch_size, "patch-size")
    patch_stride = _parse_pair_arg(args.patch_stride, "patch-stride")
    if grid_size is not None and patch_size is not None:
        raise ValueError("Use --patch-grid-size or --patch-size, not both.")
    if grid_size is None and patch_size is None:
        grid_size = 8

    images = load_image_folder(
        args.image_dir,
        image_size=args.image_size,
        max_images=args.max_images,
    )
    feature_maps = images_to_feature_maps(
        images,
        input_encoder_checkpoint=args.input_encoder_path,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    feature_maps, stats = normalize_feature_maps(
        feature_maps,
        mode=args.feature_normalize,
        clip=args.feature_clip,
    )
    gamma_seq = feature_maps_to_patch_gamma(
        feature_maps,
        grid_size=grid_size,
        patch_size=patch_size,
        stride=patch_stride,
        reduction=args.patch_reduction,
        device=args.device,
    )

    config = {
        "gamma_mode": "patch",
        "feature_map_shape": tuple(feature_maps.shape),
        "gamma_seq_shape": tuple(gamma_seq.shape),
        "grid_size": grid_size,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "patch_reduction": args.patch_reduction,
        "feature_preprocessing": stats,
    }

    gamma_seq_path = Path(args.gamma_seq_save_path)
    gamma_seq_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gamma_seq, gamma_seq_path)
    save_patch_gamma_config(config, args.save_path)
    if args.preprocess_save_path is not None:
        save_preprocessing_stats(stats, args.preprocess_save_path)

    print(f"feature maps: {tuple(feature_maps.shape)}")
    print(f"patch gamma_seq: {tuple(gamma_seq.shape)}")
    print(f"saved patch gamma_seq to {gamma_seq_path}")
    print(f"saved patch gamma config to {args.save_path}")


if __name__ == "__main__":
    main()
