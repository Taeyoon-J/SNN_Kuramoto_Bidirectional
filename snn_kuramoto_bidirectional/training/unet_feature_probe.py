import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(image_path, image_size):
    image = Image.open(image_path).convert("RGB").resize((int(image_size), int(image_size)))
    tensor = torch.as_tensor(list(image.getdata()), dtype=torch.float32)
    tensor = tensor.view(int(image_size), int(image_size), 3).permute(2, 0, 1) / 255.0
    return image, tensor.unsqueeze(0)


def build_smp_unet(encoder_name, encoder_weights, device):
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "segmentation_models_pytorch is required for this probe. Install it with: "
            "pip install segmentation-models-pytorch"
        ) from error

    weights = None if str(encoder_weights).lower() in {"none", "null"} else encoder_weights
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_unet_encoder_features(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    normalized = (image_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    features = model.encoder(normalized)
    return [feature.detach().cpu() for feature in features]


@torch.no_grad()
def predict_unet_decoder_output(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    normalized = (image_tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    logits = model(normalized)
    probability = torch.sigmoid(logits[:, :1])
    return probability.detach().cpu()


def choose_feature_levels(features, num_levels):
    candidates = []
    for idx, feature in enumerate(features):
        if feature.dim() != 4:
            continue
        if feature.size(2) <= 1 or feature.size(3) <= 1:
            continue
        candidates.append((idx, feature))

    if not candidates:
        raise ValueError("No spatial encoder features were returned by the U-Net encoder.")

    if len(candidates) <= int(num_levels):
        return candidates

    # Keep levels spread from high-resolution local features to deep semantic features.
    positions = torch.linspace(0, len(candidates) - 1, steps=int(num_levels)).round().long().tolist()
    selected = []
    seen = set()
    for pos in positions:
        if pos in seen:
            continue
        selected.append(candidates[pos])
        seen.add(pos)
    return selected


def feature_to_heatmap(feature):
    if feature.dim() != 4 or feature.size(0) != 1:
        raise ValueError("feature must have shape [1, C, H, W].")
    heatmap = feature.abs().mean(dim=1, keepdim=True)
    heatmap = heatmap - heatmap.amin(dim=(-2, -1), keepdim=True)
    denom = heatmap.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (heatmap / denom).squeeze(0).squeeze(0)


def pooled_patch_heatmap(heatmap, grid_size):
    grid_h, grid_w = grid_size
    pooled = F.adaptive_avg_pool2d(
        heatmap.view(1, 1, heatmap.size(0), heatmap.size(1)),
        output_size=(int(grid_h), int(grid_w)),
    )
    return pooled.squeeze(0).squeeze(0)


def parse_grid_sizes(values):
    if values is None:
        return [(16, 16), (8, 8), (4, 4), (1, 1)]
    return [(int(value), int(value)) for value in values]


def save_feature_probe_figure(image, level_items, grid_sizes, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(level_items)
    cols = 1 + len(grid_sizes)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)

    for row, item in enumerate(level_items):
        level_idx = item["level_index"]
        heatmap = item["heatmap"]

        ax = axes[row][0]
        ax.imshow(image)
        ax.imshow(
            F.interpolate(
                heatmap.view(1, 1, heatmap.size(0), heatmap.size(1)),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy(),
            cmap="magma",
            alpha=0.45,
        )
        ax.set_title(f"level {level_idx} feature heatmap")
        ax.axis("off")

        for col, grid_size in enumerate(grid_sizes, start=1):
            pooled = pooled_patch_heatmap(heatmap, grid_size)
            axes[row][col].imshow(pooled.numpy(), cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
            axes[row][col].set_title(f"{grid_size[0]}x{grid_size[1]} patch view")
            axes[row][col].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_individual_heatmaps(image, level_items, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for item in level_items:
        level_idx = item["level_index"]
        heatmap = item["heatmap"]
        resized = F.interpolate(
            heatmap.view(1, 1, heatmap.size(0), heatmap.size(1)),
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        axes[0].imshow(image)
        axes[0].set_title("image")
        axes[0].axis("off")
        axes[1].imshow(resized, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"level {level_idx} heatmap")
        axes[1].axis("off")
        axes[2].imshow(image)
        axes[2].imshow(resized, cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[2].set_title("overlay")
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"unet_level_{level_idx:02d}_heatmap.png", dpi=170)
        plt.close(fig)


def save_decoder_output_figure(image, decoder_output, output_path, threshold=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probability = decoder_output.squeeze(0).squeeze(0).float()
    binary = (probability >= float(threshold)).float()
    image_tensor = torch.as_tensor(list(image.getdata()), dtype=torch.float32)
    image_tensor = image_tensor.view(image.height, image.width, 3) / 255.0
    masked = image_tensor * binary.unsqueeze(-1)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")
    axes[1].imshow(probability.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("decoder probability")
    axes[1].axis("off")
    axes[2].imshow(binary.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(f"threshold >= {threshold:g}")
    axes[2].axis("off")
    axes[3].imshow(masked.numpy())
    axes[3].set_title("image x decoder mask")
    axes[3].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summarize_decoder_output(decoder_output, threshold):
    probability = decoder_output.squeeze(0).squeeze(0).float()
    binary = probability >= float(threshold)
    return {
        "shape": list(probability.shape),
        "mean": float(probability.mean().item()),
        "std": float(probability.std(unbiased=False).item()),
        "min": float(probability.min().item()),
        "max": float(probability.max().item()),
        "threshold": float(threshold),
        "binary_mask_density": float(binary.float().mean().item()),
    }


def make_summary(
    image_path,
    encoder_name,
    encoder_weights,
    level_items,
    grid_sizes,
    decoder_output_summary=None,
):
    levels = []
    for item in level_items:
        feature = item["feature"]
        heatmap = item["heatmap"]
        levels.append(
            {
                "level_index": int(item["level_index"]),
                "feature_shape": list(feature.shape),
                "heatmap_shape": list(heatmap.shape),
                "heatmap_mean": float(heatmap.mean().item()),
                "heatmap_std": float(heatmap.std(unbiased=False).item()),
                "heatmap_min": float(heatmap.min().item()),
                "heatmap_max": float(heatmap.max().item()),
            }
        )
    return {
        "image_path": str(image_path),
        "backend": "segmentation_models_pytorch.Unet",
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "num_selected_levels": len(level_items),
        "patch_grid_views": [[int(h), int(w)] for h, w in grid_sizes],
        "levels": levels,
        "decoder_output": decoder_output_summary,
        "decoder_note": (
            "This is the U-Net decoder output, not an RGB reconstruction. "
            "With segmentation_models_pytorch, encoder_weights initializes only "
            "the encoder; the segmentation decoder is not trained for CLEVR/RGB "
            "reconstruction unless separately trained."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe multi-scale U-Net encoder features on one image."
    )
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-name", default="resnet34")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--decoder-threshold", type=float, default=0.5)
    parser.add_argument("--skip-decoder-output", action="store_true")
    parser.add_argument(
        "--patch-grid-views",
        type=int,
        nargs="+",
        default=None,
        help="Square patch grid sizes to preview. Default: 16 8 4 1.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    image, image_tensor = load_image(args.image_path, args.image_size)
    model = build_smp_unet(args.encoder_name, args.encoder_weights, device)
    features = extract_unet_encoder_features(model, image_tensor, device)
    decoder_output = None
    if not args.skip_decoder_output:
        decoder_output = predict_unet_decoder_output(model, image_tensor, device)
    selected_features = choose_feature_levels(features, args.num_levels)
    grid_sizes = parse_grid_sizes(args.patch_grid_views)

    level_items = []
    for level_index, feature in selected_features:
        level_items.append(
            {
                "level_index": int(level_index),
                "feature": feature,
                "heatmap": feature_to_heatmap(feature),
            }
        )

    save_feature_probe_figure(
        image=image,
        level_items=level_items,
        grid_sizes=grid_sizes,
        output_path=output_dir / "unet_multilevel_feature_probe.png",
    )
    save_individual_heatmaps(image, level_items, output_dir)
    decoder_output_summary = None
    if decoder_output is not None:
        save_decoder_output_figure(
            image=image,
            decoder_output=decoder_output,
            output_path=output_dir / "unet_decoder_output_probe.png",
            threshold=args.decoder_threshold,
        )
        decoder_output_summary = summarize_decoder_output(
            decoder_output,
            threshold=args.decoder_threshold,
        )

    summary = make_summary(
        image_path=args.image_path,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        level_items=level_items,
        grid_sizes=grid_sizes,
        decoder_output_summary=decoder_output_summary,
    )
    with (output_dir / "unet_feature_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"saved U-Net feature probe to: {output_dir}")
    print(f"saved summary: {output_dir / 'unet_feature_summary.json'}")


if __name__ == "__main__":
    main()
