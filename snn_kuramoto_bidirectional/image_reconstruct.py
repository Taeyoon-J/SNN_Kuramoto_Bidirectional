import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.gamma_initializer import (
        FeatureMapAutoEncoder,
        FeatureMapCNNEncoder,
    )
except ModuleNotFoundError:
    from gamma_initializer import FeatureMapAutoEncoder, FeatureMapCNNEncoder


@torch.no_grad()
def decode_oscillator_features(
    decoder,
    gamma_samples,
    sigma_steps=(-2.0, -1.0, 0.0, 1.0, 2.0),
    one_hot_value=1.0,
    eps=1e-8,
    device=None,
):
    """Decode the feature-map pattern represented by every oscillator.

    Args:
        decoder:
            Trained decoder from ``FeatureMapAutoEncoder``.
        gamma_samples:
            Representative gamma vectors shaped ``[..., num_osci]``. Their
            dataset mean and standard deviation define the traversal region.
        sigma_steps:
            Multiples of each oscillator's standard deviation used for latent
            traversal. The default produces mean - 2 sigma through mean + 2
            sigma.
        one_hot_value:
            Active value used for pure one-hot decoding.
        eps:
            Minimum standard deviation used for a constant dimension.
        device:
            Inference device. Defaults to the decoder's current device.

    Returns:
        A dictionary containing CPU tensors:

        ``mean_gamma``:
            Dataset mean gamma, shaped ``[num_osci]``.
        ``std_gamma``:
            Dataset standard deviation, shaped ``[num_osci]``.
        ``sigma_steps``:
            Traversal multipliers, shaped ``[num_steps]``.
        ``mean_feature_map``:
            Decoder output at mean gamma, shaped ``[1, H, W]``.
        ``one_hot_feature_maps``:
            Pure one-hot outputs, shaped ``[num_osci, 1, H, W]``.
        ``traversal_feature_maps``:
            Decoded mean-based traversal, shaped
            ``[num_osci, num_steps, 1, H, W]``.
        ``difference_from_mean``:
            Traversal outputs minus the mean output, with the same shape.
    """
    if not torch.is_tensor(gamma_samples):
        gamma_samples = torch.as_tensor(gamma_samples, dtype=torch.float32)
    if gamma_samples.dim() < 2:
        raise ValueError("gamma_samples must have shape [..., num_osci].")
    if gamma_samples.numel() == 0:
        raise ValueError("gamma_samples must not be empty.")

    flat_gamma = gamma_samples.float().reshape(-1, gamma_samples.shape[-1])
    if flat_gamma.size(0) < 2:
        raise ValueError("At least two gamma samples are required for std.")

    model_device = next(decoder.parameters()).device
    device = torch.device(device) if device is not None else model_device
    decoder = decoder.to(device)
    flat_gamma = flat_gamma.to(device)
    steps = torch.as_tensor(sigma_steps, dtype=flat_gamma.dtype, device=device)
    if steps.dim() != 1 or steps.numel() == 0:
        raise ValueError("sigma_steps must be a non-empty one-dimensional sequence.")

    was_training = decoder.training
    decoder.eval()
    try:
        mean_gamma = flat_gamma.mean(dim=0)
        std_gamma = flat_gamma.std(dim=0, unbiased=False).clamp_min(float(eps))
        num_osci = mean_gamma.numel()

        mean_feature_map = decoder(mean_gamma.unsqueeze(0))
        _validate_decoded_feature_maps(mean_feature_map, 1)

        one_hot_gamma = torch.eye(
            num_osci, dtype=flat_gamma.dtype, device=device
        ) * float(one_hot_value)
        one_hot_feature_maps = decoder(one_hot_gamma)
        _validate_decoded_feature_maps(one_hot_feature_maps, num_osci)

        traversal_gamma = mean_gamma.view(1, 1, num_osci).expand(
            num_osci, steps.numel(), num_osci
        ).clone()
        oscillator_indices = torch.arange(num_osci, device=device)
        traversal_gamma[oscillator_indices, :, oscillator_indices] += (
            steps.unsqueeze(0) * std_gamma.unsqueeze(1)
        )
        traversal_feature_maps = decoder(
            traversal_gamma.reshape(-1, num_osci)
        )
        _validate_decoded_feature_maps(
            traversal_feature_maps, num_osci * steps.numel()
        )
        traversal_feature_maps = traversal_feature_maps.reshape(
            num_osci, steps.numel(), *traversal_feature_maps.shape[1:]
        )
        difference_from_mean = traversal_feature_maps - mean_feature_map.view(
            1, 1, *mean_feature_map.shape[1:]
        )

        return {
            "mean_gamma": mean_gamma.cpu(),
            "std_gamma": std_gamma.cpu(),
            "sigma_steps": steps.cpu(),
            "mean_feature_map": mean_feature_map.squeeze(0).cpu(),
            "one_hot_feature_maps": one_hot_feature_maps.cpu(),
            "traversal_feature_maps": traversal_feature_maps.cpu(),
            "difference_from_mean": difference_from_mean.cpu(),
        }
    finally:
        decoder.train(was_training)


def decode_oscillator_features_from_checkpoint(
    decoder_checkpoint_path,
    input_size,
    gamma_samples,
    decoder_hidden_dim=None,
    device=None,
    **decode_kwargs,
):
    """Load a trained decoder checkpoint and decode oscillator patterns.

    The checkpoint must be the decoder ``state_dict`` saved by
    ``save_gamma_decoder``. ``num_osci`` and the hidden size are inferred from
    the weights; ``input_size`` disambiguates the flattened output dimensions.
    """
    height, width = _validate_input_size(input_size)
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state_dict = torch.load(decoder_checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint must contain a decoder state_dict.")

    first_weight = state_dict.get("0.weight")
    output_weight = state_dict.get("2.weight")
    if first_weight is None or output_weight is None:
        raise ValueError(
            "The checkpoint does not match a FeatureMapAutoEncoder decoder."
        )
    num_osci = int(first_weight.shape[1])
    inferred_hidden_dim = int(first_weight.shape[0])
    if decoder_hidden_dim is not None and int(decoder_hidden_dim) != inferred_hidden_dim:
        raise ValueError(
            f"decoder_hidden_dim={decoder_hidden_dim} does not match checkpoint "
            f"dimension {inferred_hidden_dim}."
        )
    if int(output_weight.shape[0]) != height * width:
        raise ValueError(
            "input_size does not match the decoder checkpoint output size."
        )

    decoder = nn.Sequential(
        nn.Linear(num_osci, inferred_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(inferred_hidden_dim, height * width),
        nn.Unflatten(dim=1, unflattened_size=(1, height, width)),
    ).to(device)
    decoder.load_state_dict(state_dict)
    return decode_oscillator_features(
        decoder,
        gamma_samples=gamma_samples,
        device=device,
        **decode_kwargs,
    )


def decode_oscillator_features_from_autoencoder_checkpoint(
    autoencoder_checkpoint_path,
    gamma_samples,
    device=None,
    **decode_kwargs,
):
    """Load a full ``FeatureMapAutoEncoder`` checkpoint and decode features.

    Supports checkpoints saved either as:
        {"state_dict": ..., "input_size": ..., "num_osci": ...}

    or plain ``FeatureMapAutoEncoder.state_dict()`` when ``input_size`` and
    ``num_osci`` are inferable from decoder/encoder weights.
    """
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(autoencoder_checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must be a dictionary.")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        input_size = checkpoint["input_size"]
        num_osci = int(checkpoint["num_osci"])
        hidden_channels = tuple(checkpoint.get("hidden_channels", (16, 32, 64)))
        decoder_hidden_dim = int(checkpoint.get("decoder_hidden_dim", 256))
        dropout = float(checkpoint.get("dropout", 0.0))
    else:
        state_dict = checkpoint
        decoder_first_weight = state_dict.get("decoder.0.weight")
        decoder_output_weight = state_dict.get("decoder.2.weight")
        projection_weight = state_dict.get("encoder.projection.2.weight")
        if decoder_first_weight is None or decoder_output_weight is None or projection_weight is None:
            raise ValueError("The checkpoint does not match a FeatureMapAutoEncoder.")
        num_osci = int(projection_weight.shape[0])
        decoder_hidden_dim = int(decoder_first_weight.shape[0])
        flat_output = int(decoder_output_weight.shape[0])
        side = int(flat_output ** 0.5)
        if side * side != flat_output:
            raise ValueError("Cannot infer square decoder output size from checkpoint.")
        input_size = (side, side)
        hidden_channels = _infer_hidden_channels_from_autoencoder_state_dict(state_dict)
        dropout = 0.0

    autoencoder = FeatureMapAutoEncoder(
        input_size=input_size,
        num_osci=num_osci,
        hidden_channels=hidden_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
    ).to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return decode_oscillator_features(
        autoencoder.decoder,
        gamma_samples=gamma_samples,
        device=device,
        **decode_kwargs,
    )


def _infer_hidden_channels_from_autoencoder_state_dict(state_dict):
    conv_weights = [
        value
        for key, value in state_dict.items()
        if key.startswith("encoder.cnn.") and key.endswith(".weight") and value.dim() == 4
    ]
    if not conv_weights:
        return (16, 32, 64)
    return tuple(int(weight.shape[0]) for weight in conv_weights)


def _validate_decoded_feature_maps(feature_maps, expected_batch_size):
    if feature_maps.dim() != 4 or feature_maps.size(0) != expected_batch_size:
        raise ValueError(
            "decoder must return feature maps shaped [B, 1, H, W]."
        )
    if feature_maps.size(1) != 1:
        raise ValueError("decoder output must contain exactly one channel.")


def _validate_input_size(input_size):
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
        raise ValueError("input_size must be a (height, width) pair.")
    height, width = int(input_size[0]), int(input_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("input_size values must be positive.")
    return height, width


def maximize_oscillator_images_from_checkpoint(
    checkpoint_path,
    input_size,
    device=None,
    **maximization_kwargs,
):
    """Load a saved gamma initializer and maximize all oscillator activations.

    ``checkpoint_path`` must contain the direct-linear encoder ``state_dict``
    saved by ``save_gamma_initializer``. Model dimensions are inferred from
    its projection weight and checked against ``input_size``.
    Remaining keyword arguments are forwarded to
    :func:`maximize_oscillator_images`.
    """
    device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint must contain an encoder state_dict.")

    projection_weight = state_dict.get("projection.2.weight")
    if projection_weight is None or projection_weight.dim() != 2:
        raise ValueError(
            "The checkpoint does not match a direct-linear gamma encoder "
            "state_dict."
        )

    num_osci = int(projection_weight.shape[0])
    height, width = _validate_input_size(input_size)
    expected_features = height * width
    if int(projection_weight.shape[1]) != expected_features:
        raise ValueError(
            f"input_size {(height, width)} requires {expected_features} linear "
            f"features, but the checkpoint expects {projection_weight.shape[1]}."
        )

    model = FeatureMapCNNEncoder(
        num_osci=num_osci,
        in_channels=1,
        input_size=(height, width),
    ).to(device)
    model.load_state_dict(state_dict)

    return maximize_oscillator_images(
        model,
        input_size=input_size,
        device=device,
        **maximization_kwargs,
    )


def maximize_oscillator_images(
    gamma_initializer,
    input_size,
    steps=500,
    lr=0.05,
    tv_weight=1e-3,
    l2_weight=1e-4,
    other_weight=0.1,
    init_mean=0.0,
    init_std=0.05,
    value_range=None,
    device=None,
):
    """Create one activation-maximizing feature-map image per oscillator.

    The gamma initializer consumes a single feature map, not an RGB image.
    Consequently, the returned images visualize feature-map patterns that
    maximize each gamma dimension.

    Args:
        gamma_initializer:
            Trained ``FeatureMapCNNEncoder`` module.
        input_size:
            ``(height, width)`` of the feature maps used during training.
        steps:
            Number of gradient-ascent optimization steps.
        lr:
            Adam learning rate for the generated feature-map images.
        tv_weight:
            Weight of total-variation regularization, which suppresses noisy
            high-frequency patterns.
        l2_weight:
            Weight of the mean squared pixel-value penalty.
        other_weight:
            Weight used to suppress non-target oscillator activations. Set to
            zero to maximize only the target dimension.
        init_mean, init_std:
            Mean and standard deviation of the initial random images.
        value_range:
            Optional ``(minimum, maximum)`` used to clamp optimized values.
            Use the observed feature-map range when it is known. By default,
            values are not clamped because feature maps may be signed.
        device:
            Optimization device. Defaults to the model's current device.

    Returns:
        Tensor shaped ``[num_osci, 1, height, width]`` on CPU. Item ``d`` is
        the feature-map image optimized for oscillator ``d``.
    """
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
        raise ValueError("input_size must be a (height, width) pair.")
    height, width = (int(input_size[0]), int(input_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("input_size values must be positive.")
    if int(steps) <= 0:
        raise ValueError("steps must be positive.")
    if float(lr) <= 0:
        raise ValueError("lr must be positive.")
    if value_range is not None:
        if len(value_range) != 2 or value_range[0] >= value_range[1]:
            raise ValueError("value_range must be a (minimum, maximum) pair.")

    model_device = next(gamma_initializer.parameters()).device
    device = torch.device(device) if device is not None else model_device
    gamma_initializer = gamma_initializer.to(device)

    was_training = gamma_initializer.training
    original_requires_grad = [
        parameter.requires_grad for parameter in gamma_initializer.parameters()
    ]
    gamma_initializer.eval()
    gamma_initializer.requires_grad_(False)

    try:
        with torch.no_grad():
            probe = torch.zeros(1, 1, height, width, device=device)
            num_osci = gamma_initializer(probe).shape[-1]

        images = torch.empty(
            num_osci, 1, height, width, device=device
        ).normal_(mean=float(init_mean), std=float(init_std))
        images.requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=float(lr))
        targets = torch.arange(num_osci, device=device)

        for _ in range(int(steps)):
            gamma = gamma_initializer(images)
            if gamma.shape != (num_osci, num_osci):
                raise ValueError(
                    "gamma_initializer must return [B, num_osci] for input "
                    "shaped [B, 1, H, W]."
                )

            target_activation = gamma[targets, targets].mean()
            if num_osci > 1 and float(other_weight) != 0.0:
                non_target_sum = gamma.sum(dim=1) - gamma[targets, targets]
                non_target_activation = (non_target_sum / (num_osci - 1)).mean()
            else:
                non_target_activation = gamma.new_zeros(())

            total_variation = _total_variation(images)
            l2_penalty = images.square().mean()
            objective = (
                target_activation
                - float(other_weight) * non_target_activation
                - float(tv_weight) * total_variation
                - float(l2_weight) * l2_penalty
            )

            optimizer.zero_grad()
            (-objective).backward()
            optimizer.step()

            if value_range is not None:
                with torch.no_grad():
                    images.clamp_(float(value_range[0]), float(value_range[1]))

        return images.detach().cpu()
    finally:
        for parameter, requires_grad in zip(
            gamma_initializer.parameters(), original_requires_grad
        ):
            parameter.requires_grad_(requires_grad)
        gamma_initializer.train(was_training)


def _total_variation(images):
    """Return mean anisotropic total variation for ``[B, C, H, W]``."""
    vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
    horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
    return vertical.abs().mean() + horizontal.abs().mean()


def save_decoded_feature_images(decoded, output_dir, max_dims=None):
    """Save decoded oscillator feature maps as PNG files."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    one_hot = decoded["one_hot_feature_maps"].squeeze(1)
    num_dims = one_hot.size(0) if max_dims is None else min(int(max_dims), one_hot.size(0))
    for dim in range(num_dims):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(one_hot[dim].numpy(), cmap="viridis")
        ax.set_title(f"one-hot gamma dim {dim}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        fig.tight_layout()
        fig.savefig(output_dir / f"gamma_dim_{dim:03d}_one_hot.png", dpi=160)
        plt.close(fig)

    cols = min(10, num_dims)
    rows = (num_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= num_dims:
            continue
        ax.imshow(one_hot[idx].numpy(), cmap="viridis")
        ax.set_title(str(idx), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_one_hot_grid.png", dpi=160)
    plt.close(fig)

    mean_feature = decoded["mean_feature_map"].squeeze(0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mean_feature.numpy(), cmap="viridis")
    ax.set_title("decoded mean gamma")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_mean_feature_map.png", dpi=160)
    plt.close(fig)


def save_feature_map_grid(feature_maps, output_path, title, columns=10):
    """Save all oscillator feature maps in one signed-color grid PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not torch.is_tensor(feature_maps):
        feature_maps = torch.as_tensor(feature_maps)
    feature_maps = feature_maps.detach().cpu().float()
    if feature_maps.dim() == 4 and feature_maps.size(1) == 1:
        feature_maps = feature_maps.squeeze(1)
    if feature_maps.dim() != 3:
        raise ValueError("feature_maps must have shape [D, H, W] or [D, 1, H, W].")

    num_dims = feature_maps.size(0)
    columns = min(int(columns), num_dims)
    rows = (num_dims + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.1, rows * 2.1),
        squeeze=False,
    )
    for dim, ax in enumerate(axes.flat):
        ax.axis("off")
        if dim >= num_dims:
            continue
        values = feature_maps[dim].numpy()
        limit = max(float(abs(values).max()), 1e-8)
        ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"dim {dim}", fontsize=8)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_all_dimension_visualizations(
    encoder_checkpoint_path,
    decoder_checkpoint_path,
    gamma_samples,
    input_size,
    output_dir,
    sigma_step=2.0,
    activation_steps=500,
    activation_lr=0.05,
    device=None,
):
    """Save one 90-dim grid for each of the two visualization methods."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_maps = maximize_oscillator_images_from_checkpoint(
        checkpoint_path=encoder_checkpoint_path,
        input_size=input_size,
        steps=activation_steps,
        lr=activation_lr,
        device=device,
    )
    activation_path = save_feature_map_grid(
        activation_maps,
        output_dir / "activation_maximization_all_dimensions.png",
        title="Activation maximization: all gamma dimensions",
    )

    decoded = decode_oscillator_features_from_checkpoint(
        decoder_checkpoint_path=decoder_checkpoint_path,
        input_size=input_size,
        gamma_samples=gamma_samples,
        sigma_steps=(float(sigma_step),),
        device=device,
    )
    difference_maps = decoded["difference_from_mean"][:, 0]
    decoder_path = save_feature_map_grid(
        difference_maps,
        output_dir / "decoder_difference_all_dimensions.png",
        title=f"Decoder difference from mean: +{float(sigma_step):g} sigma",
    )
    return activation_path, decoder_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize all gamma dimensions in two grid PNG files."
    )
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--decoder-path", required=True)
    parser.add_argument("--gamma-seq-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-size", type=int, nargs=2, required=True)
    parser.add_argument("--sigma-step", type=float, default=2.0)
    parser.add_argument("--activation-steps", type=int, default=500)
    parser.add_argument("--activation-lr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gamma_seq = torch.load(args.gamma_seq_path, map_location="cpu").float()
    activation_path, decoder_path = save_all_dimension_visualizations(
        encoder_checkpoint_path=args.encoder_path,
        decoder_checkpoint_path=args.decoder_path,
        gamma_samples=gamma_seq,
        input_size=tuple(args.input_size),
        output_dir=args.output_dir,
        sigma_step=args.sigma_step,
        activation_steps=args.activation_steps,
        activation_lr=args.activation_lr,
        device=args.device,
    )
    print(f"Activation maximization grid: {activation_path}")
    print(f"Decoder difference grid: {decoder_path}")


if __name__ == "__main__":
    main()
