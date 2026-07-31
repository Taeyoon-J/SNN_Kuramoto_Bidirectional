import torch
import torch.nn.functional as F
import sys
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.gamma_initializer import (
        FeatureMapAutoEncoder,
        FeatureMapCNNEncoder,
        _prepare_feature_maps,
        feature_maps_to_patch_gamma,
    )
    from snn_kuramoto_bidirectional.training.train_input_layer_generator import load_input_encoder
except ModuleNotFoundError:
    from gamma_initializer import (
        FeatureMapAutoEncoder,
        FeatureMapCNNEncoder,
        _prepare_feature_maps,
        feature_maps_to_patch_gamma,
    )
    from training.train_input_layer_generator import load_input_encoder


def train_gamma_initializer(
    feature_maps,
    num_osci,
    hidden_channels=(16, 32, 64),
    decoder_hidden_dim=256,
    epochs=200,
    lr=1e-3,
    batch_size=32,
    dropout=0.0,
    device=None,
    optimizer_cls=torch.optim.Adam,
    save_path=None,
    decoder_save_path=None,
    verbose=False,
):
    """
    Pretrain FeatureMapCNNEncoder with its paired decoder.

    Objective:
        feature_map -> FeatureMapCNNEncoder -> vector -> decoder -> reconstructed feature_map
        minimize MSE(reconstruction, feature_map)

    Args:
        feature_maps:
            Tensor shaped [B, T, H, W].
        save_path:
            Encoder checkpoint path. When provided, the decoder is also saved.
        decoder_save_path:
            Optional decoder checkpoint path. If omitted while save_path is
            provided, ``<encoder_stem>_decoder<suffix>`` is used.

    Returns:
        trained_encoder, autoencoder, loss_history
    """
    samples, _ = _prepare_feature_maps(feature_maps)
    samples = samples.detach()
    device = torch.device(device) if device is not None else samples.device
    samples = samples.to(device)

    autoencoder = FeatureMapAutoEncoder(
        input_size=samples.shape[-2:],
        num_osci=num_osci,
        hidden_channels=hidden_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
    ).to(device)

    dataset = TensorDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
    )
    optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)
    loss_history = []

    autoencoder.train()
    for epoch in range(1, int(epochs) + 1):
        epoch_loss = 0.0
        sample_count = 0
        for (batch,) in loader:
            reconstruction = autoencoder(batch)
            loss = F.mse_loss(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            sample_count += batch.size(0)
        mean_loss = epoch_loss / sample_count
        loss_history.append(mean_loss)
        if verbose:
            print(
                f"Epoch {epoch:04d}/{int(epochs):04d} | loss={mean_loss:.8f}",
                flush=True,
            )

    if save_path is not None:
        save_gamma_initializer(autoencoder.encoder, save_path)
        if decoder_save_path is None:
            decoder_save_path = _decoder_path_from_encoder_path(save_path)
        save_gamma_decoder(autoencoder.decoder, decoder_save_path)
    elif decoder_save_path is not None:
        save_gamma_decoder(autoencoder.decoder, decoder_save_path)

    return autoencoder.encoder, autoencoder, loss_history


def copy_trained_gamma_initializer(gamma_generator, trained_encoder):
    """Copy pretrained FeatureMapCNNEncoder weights into GammaGenerator.gamma_initializer."""
    gamma_generator.gamma_initializer.load_state_dict(trained_encoder.state_dict())
    return gamma_generator


def save_gamma_initializer(encoder, save_path):
    """Save a trained FeatureMapCNNEncoder state_dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path)


def save_gamma_decoder(decoder, save_path):
    """Save a trained FeatureMapAutoEncoder decoder state_dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), save_path)


def _decoder_path_from_encoder_path(encoder_path):
    """Derive a decoder checkpoint path next to the encoder checkpoint."""
    encoder_path = Path(encoder_path)
    return encoder_path.with_name(
        f"{encoder_path.stem}_decoder{encoder_path.suffix}"
    )


def load_gamma_initializer(
    checkpoint_path,
    num_osci,
    in_channels=1,
    hidden_channels=(16, 32, 64),
    dropout=0.0,
    device=None,
):
    """Load a FeatureMapCNNEncoder from a saved state_dict."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    encoder = FeatureMapCNNEncoder(
        num_osci=num_osci,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def load_image_folder(image_dir, image_size=128, max_images=None):
    """Load RGB images from a folder into a [B, 3, H, W] float tensor."""
    image_dir = Path(image_dir)
    paths = sorted(
        path for path in image_dir.iterdir()
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
    """Convert image tensors [B, 3, H, W] into feature maps [B, T, H', W']."""
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
        batch = images[start:start + int(batch_size)].to(device)
        feature_maps.append(encoder(batch).cpu())
    return torch.cat(feature_maps, dim=0)


def normalize_feature_maps(feature_maps, mode="none", clip=None, eps=1e-6):
    """Normalize feature-map values before gamma autoencoder training."""
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
        stats = {"mode": mode, "mean": mean.detach().cpu(), "std": std.detach().cpu()}
    elif mode == "per_map_standardize":
        mean = feature_maps.mean(dim=(-2, -1), keepdim=True)
        std = feature_maps.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(float(eps))
        normalized = (feature_maps - mean) / std
        stats = {"mode": mode}
    elif mode == "minmax":
        min_value = feature_maps.amin()
        max_value = feature_maps.amax()
        normalized = (feature_maps - min_value) / (max_value - min_value + float(eps))
        stats = {
            "mode": mode,
            "min": min_value.detach().cpu(),
            "max": max_value.detach().cpu(),
        }
    else:
        raise ValueError(
            'mode must be one of "none", "standardize", "per_map_standardize", or "minmax".'
        )

    if clip is not None:
        normalized = normalized.clamp(-float(clip), float(clip))
        stats["clip"] = float(clip)
    return normalized, stats


@torch.no_grad()
def encode_feature_maps_to_gamma_seq(autoencoder, feature_maps, batch_size=32, device=None):
    """Encode feature maps [B, T, H, W] into gamma_seq [B, T, num_osci]."""
    samples, restore_shape = _prepare_feature_maps(feature_maps)
    device = torch.device(device) if device is not None else next(autoencoder.parameters()).device
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    vectors = []
    for start in range(0, samples.size(0), int(batch_size)):
        batch = samples[start:start + int(batch_size)].to(device)
        vectors.append(autoencoder.encoder(batch).cpu())
    vectors = torch.cat(vectors, dim=0)
    return vectors.view(restore_shape["batch_size"], restore_shape["num_maps"], -1)


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
    import argparse

    parser = argparse.ArgumentParser(
        description="Train FeatureMapCNNEncoder from pretrained input-layer feature maps."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--input-encoder-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--decoder-save-path", default=None)
    parser.add_argument("--gamma-seq-save-path", default=None)
    parser.add_argument("--preprocess-save-path", default=None)
    parser.add_argument("--gamma-mode", default="autoencoder", choices=["autoencoder", "patch"])
    parser.add_argument("--num-kernels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--num-osci", type=int, default=90)
    parser.add_argument("--patch-grid-size", type=int, nargs="+", default=None)
    parser.add_argument("--patch-size", type=int, nargs="+", default=None)
    parser.add_argument("--patch-stride", type=int, nargs="+", default=None)
    parser.add_argument("--patch-reduction", default="mean", choices=["mean", "max"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--feature-normalize",
        default="none",
        choices=["none", "standardize", "per_map_standardize", "minmax"],
    )
    parser.add_argument("--feature-clip", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
    print(f"feature maps: {tuple(feature_maps.shape)}")
    print(f"feature preprocessing: {stats}")

    if args.gamma_mode == "patch":
        grid_size = _parse_pair_arg(args.patch_grid_size, "patch-grid-size")
        patch_size = _parse_pair_arg(args.patch_size, "patch-size")
        patch_stride = _parse_pair_arg(args.patch_stride, "patch-stride")
        if grid_size is None and patch_size is None:
            grid_size = 10

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

        if args.gamma_seq_save_path is None:
            raise ValueError("--gamma-seq-save-path is required when --gamma-mode patch.")
        gamma_seq_path = Path(args.gamma_seq_save_path)
        gamma_seq_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gamma_seq, gamma_seq_path)
        print(f"saved patch gamma_seq: {tuple(gamma_seq.shape)} to {gamma_seq_path}")

        save_patch_gamma_config(config, args.save_path)
        print(f"saved patch gamma config: {args.save_path}")

        if args.preprocess_save_path is not None:
            save_preprocessing_stats(stats, args.preprocess_save_path)
            print(f"saved preprocessing stats: {args.preprocess_save_path}")
        return

    _, autoencoder, losses = train_gamma_initializer(
        feature_maps,
        num_osci=args.num_osci,
        decoder_hidden_dim=args.decoder_hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=args.device,
        save_path=args.save_path,
        decoder_save_path=args.decoder_save_path,
        verbose=args.verbose,
    )

    if args.gamma_seq_save_path is not None:
        gamma_seq = encode_feature_maps_to_gamma_seq(
            autoencoder,
            feature_maps,
            batch_size=args.batch_size,
            device=args.device,
        )
        gamma_seq_path = Path(args.gamma_seq_save_path)
        gamma_seq_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gamma_seq, gamma_seq_path)
        print(f"saved gamma_seq: {tuple(gamma_seq.shape)} to {gamma_seq_path}")

    if args.preprocess_save_path is not None:
        save_preprocessing_stats(stats, args.preprocess_save_path)
        print(f"saved preprocessing stats: {args.preprocess_save_path}")

    print(f"trained gamma initializer: {args.save_path}")
    if args.decoder_save_path is not None:
        print(f"trained gamma decoder: {args.decoder_save_path}")
    print(f"loss: {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
