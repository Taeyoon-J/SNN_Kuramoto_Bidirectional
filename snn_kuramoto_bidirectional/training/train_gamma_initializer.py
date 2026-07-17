import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from snn_kuramoto_bidirectional.gamma_initializer import (
    FeatureMapAutoEncoder,
    FeatureMapCNNEncoder,
    _prepare_feature_maps,
)


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
