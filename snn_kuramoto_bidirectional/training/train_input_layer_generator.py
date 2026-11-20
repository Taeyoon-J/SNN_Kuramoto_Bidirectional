import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from snn_kuramoto_bidirectional.input_layer_generator import (
    CNNAutoEncoder,
    CNNFeatureEncoder,
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
