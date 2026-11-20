import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from snn_kuramoto_bidirectional.training.train_gamma_initializer import load_gamma_autoencoder
except ModuleNotFoundError:
    from training.train_gamma_initializer import load_gamma_autoencoder


def _parse_dims(value, num_dims):
    if value == "all":
        return list(range(num_dims))
    dims = [int(item.strip()) for item in value.split(",") if item.strip()]
    for dim in dims:
        if dim < 0 or dim >= num_dims:
            raise ValueError(f"dimension {dim} is outside [0, {num_dims}).")
    return dims


@torch.no_grad()
def decode_gamma_dimensions(
    autoencoder_path,
    output_dir,
    dims="all",
    gamma_seq_path=None,
    activation_scale=2.0,
    device=None,
):
    import matplotlib.pyplot as plt

    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    autoencoder = load_gamma_autoencoder(autoencoder_path, device=device)
    num_dims = autoencoder.encoder.projection[-1].out_features
    selected_dims = _parse_dims(dims, num_dims)

    if gamma_seq_path is None:
        baseline = torch.zeros(num_dims, device=device)
        scale = torch.ones(num_dims, device=device)
    else:
        gamma_seq = torch.load(gamma_seq_path, map_location=device).float()
        flat = gamma_seq.reshape(-1, gamma_seq.size(-1))
        baseline = flat.mean(dim=0)
        scale = flat.std(dim=0).clamp_min(1e-6)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors = baseline.unsqueeze(0).repeat(len(selected_dims), 1)
    for row, dim in enumerate(selected_dims):
        vectors[row, dim] = baseline[dim] + float(activation_scale) * scale[dim]

    decoded = autoencoder.decoder(vectors).squeeze(1).cpu()

    for row, dim in enumerate(selected_dims):
        feature = decoded[row]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(feature.numpy(), cmap="viridis")
        ax.set_title(f"Decoded gamma dim {dim}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        fig.tight_layout()
        fig.savefig(output_dir / f"gamma_dim_{dim:03d}.png", dpi=160)
        plt.close(fig)

    cols = min(10, len(selected_dims))
    rows = (len(selected_dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= len(selected_dims):
            continue
        ax.imshow(decoded[idx].numpy(), cmap="viridis")
        ax.set_title(str(selected_dims[idx]), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_dimension_grid.png", dpi=160)
    plt.close(fig)

    return selected_dims


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decode individual gamma dimensions into feature-map images.")
    parser.add_argument("--autoencoder-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dims", default="all", help='Comma-separated dimensions, e.g. "0,1,2", or "all".')
    parser.add_argument("--gamma-seq-path", default=None)
    parser.add_argument("--activation-scale", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dims = decode_gamma_dimensions(
        args.autoencoder_path,
        output_dir=args.output_dir,
        dims=args.dims,
        gamma_seq_path=args.gamma_seq_path,
        activation_scale=args.activation_scale,
        device=args.device,
    )
    print(f"decoded {len(dims)} gamma dimensions to: {args.output_dir}")


if __name__ == "__main__":
    main()
