import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

from snn_kuramoto_bidirectional.input_layer_generator import (
    CNNAutoEncoder,
    CNNFeatureEncoder,
)


def load_hdf5_images(
    dataset_path,
    hdf5_key="image",
    image_size=128,
    max_images=None,
):
    """Load HDF5 RGB images as a float tensor shaped [B, 3, H, W]."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to load an HDF5 image dataset.") from error

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found: {dataset_path}")

    with h5py.File(dataset_path, "r") as file:
        if hdf5_key not in file:
            raise KeyError(f"HDF5 key {hdf5_key!r} was not found in {dataset_path}.")
        dataset = file[hdf5_key]
        count = len(dataset) if max_images is None else min(int(max_images), len(dataset))
        if count <= 0:
            raise ValueError("The selected HDF5 image set is empty.")
        images = torch.from_numpy(dataset[:count])

    if images.ndim != 4:
        raise ValueError(f"HDF5 images must be 4D, but got shape {tuple(images.shape)}.")
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    elif images.shape[1] != 3:
        raise ValueError(
            "HDF5 images must have shape [B, H, W, 3] or [B, 3, H, W]."
        )

    was_integer = not images.is_floating_point()
    images = images.float()
    if was_integer:
        images = images / 255.0

    target_size = (int(image_size), int(image_size))
    if tuple(images.shape[-2:]) != target_size:
        images = F.interpolate(
            images,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return images.contiguous()


def load_training_images(
    *,
    image_dir=None,
    dataset_path=None,
    hdf5_key="image",
    image_size=128,
    max_images=None,
):
    """Load training images from exactly one folder or HDF5 source."""
    if (image_dir is None) == (dataset_path is None):
        raise ValueError("Provide exactly one of image_dir or dataset_path.")
    if dataset_path is not None:
        return load_hdf5_images(
            dataset_path,
            hdf5_key=hdf5_key,
            image_size=image_size,
            max_images=max_images,
        )

    from snn_kuramoto_bidirectional.training.train_gamma_initializer import (
        load_image_folder,
    )
    return load_image_folder(
        image_dir,
        image_size=image_size,
        max_images=max_images,
    )


def train_input_layer_generator(
    images,
    num_kernels,
    kernel_size,
    channels=3,
    epochs=500,
    lr=1e-3,
    batch_size=32,
    bias=True,
    device=None,
    optimizer_cls=torch.optim.Adam,
    save_path=None,
):
    """
    Pretrain CNNFeatureEncoder with its paired decoder.

    Objective:
        image -> CNNFeatureEncoder -> CNNFeatureDecoder -> reconstructed image
        minimize MSE(reconstruction, image)

    Args:
        images:
            Tensor shaped [B, 3, H, W].

    Returns:
        trained_encoder, autoencoder, loss_history
    """
    if not torch.is_tensor(images):
        images = torch.as_tensor(images, dtype=torch.float32)
    images = images.float()

    autoencoder = CNNAutoEncoder(
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        channels=channels,
        bias=bias,
    )
    target = autoencoder.encoder._prepare_image(images)
    autoencoder.encoder._validate_image_size(target)

    device = torch.device(device) if device is not None else target.device
    autoencoder = autoencoder.to(device)
    target = target.to(device)

    dataset = TensorDataset(target)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
    )
    optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)
    loss_history = []

    autoencoder.train()
    for _ in range(int(epochs)):
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
        loss_history.append(epoch_loss / sample_count)

    if save_path is not None:
        save_input_encoder(autoencoder.encoder, save_path)

    return autoencoder.encoder, autoencoder, loss_history


def copy_trained_input_encoder(gamma_generator, trained_encoder):
    """Copy pretrained CNNFeatureEncoder weights into GammaGenerator.input_layer."""
    gamma_generator.input_layer.load_state_dict(trained_encoder.state_dict())
    return gamma_generator


def save_input_encoder(encoder, save_path):
    """Save a trained CNNFeatureEncoder state_dict."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), save_path)


def load_input_encoder(
    checkpoint_path,
    num_kernels,
    kernel_size,
    channels=3,
    bias=True,
    device=None,
):
    """Load a CNNFeatureEncoder from a saved state_dict."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    encoder = CNNFeatureEncoder(
        num_kernels=num_kernels,
        kernel_size=kernel_size,
        in_channels=channels,
        bias=bias,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CNNFeatureEncoder from an image folder or HDF5 dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir")
    source.add_argument("--dataset-path")
    parser.add_argument("--hdf5-key", default="image")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--num-kernels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    images = load_training_images(
        image_dir=args.image_dir,
        dataset_path=args.dataset_path,
        hdf5_key=args.hdf5_key,
        image_size=args.image_size,
        max_images=args.max_images,
    )
    _, _, losses = train_input_layer_generator(
        images=images,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save_path,
    )
    print(f"Input images: {tuple(images.shape)}", flush=True)
    print(f"Input encoder: {args.save_path}", flush=True)
    print(f"Loss: {losses[0]:.8f} -> {losses[-1]:.8f}", flush=True)


if __name__ == "__main__":
    main()
