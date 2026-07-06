import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FeatureMapCNNEncoder(nn.Module):
    """
    CNN encoder that turns each 2D feature map into one feature vector.

    Expected input:
        [B, 1, H, W]

    Output:
        [B, num_osci]
    """

    def __init__(
        self,
        num_osci,
        in_channels=1,
        hidden_channels=(16, 32, 64),
        dropout=0.0,
    ):
        super().__init__()

        layers = []
        current_channels = int(in_channels)
        for out_channels in hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, int(out_channels), kernel_size=3, padding=1),
                    nn.BatchNorm2d(int(out_channels)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_channels = int(out_channels)

        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(current_channels, num_osci),
        )

    def forward(self, feature_maps):
        x = self.cnn(feature_maps)
        x = self.pool(x)
        return self.projection(x)


class FeatureMapAutoEncoder(nn.Module):
    """
    Unsupervised trainer for FeatureMapCNNEncoder.

    It learns a compact vector by reconstructing each input feature map:
        feature map -> CNN encoder -> vector -> decoder -> reconstructed feature map
    """

    def __init__(
        self,
        input_size,
        num_osci,
        hidden_channels=(16, 32, 64),
        decoder_hidden_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        height, width = input_size
        self.input_size = (height, width)
        self.encoder = FeatureMapCNNEncoder(
            num_osci=num_osci,
            in_channels=1,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_osci, int(decoder_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(decoder_hidden_dim), height * width),
            nn.Unflatten(dim=1, unflattened_size=(1, height, width)),
        )

    def forward(self, feature_maps):
        vectors = self.encoder(feature_maps)
        reconstruction = self.decoder(vectors)
        return reconstruction

    def encode(self, feature_maps):
        return self.encoder(feature_maps)


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
):
    """
    Train one CNN model that converts each feature map into a feature vector.

    Args:
        feature_maps:
            [B, T, H, W]. Each feature map is treated as one training sample.
        num_osci:
            Length of the output vector for each feature map.

    Returns:
        trained_encoder, autoencoder, loss_history
    """
    samples, _ = _prepare_feature_maps(feature_maps)
    samples = samples.detach()
    device = _resolve_device(device, samples)
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
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_history = []

    autoencoder.train()
    for _ in range(int(epochs)):
        total_loss = 0.0
        total_count = 0
        for (batch,) in loader:
            reconstruction = autoencoder(batch)
            loss = F.mse_loss(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_count += batch.size(0)
        loss_history.append(total_loss / total_count)

    return autoencoder.encoder, autoencoder, loss_history


@torch.no_grad()
def feature_maps_to_vectors(feature_maps, encoder, device=None):
    """
    Convert feature maps to vectors with a trained FeatureMapCNNEncoder.

    Shape behavior:
        [B, T, H, W] -> [B, T, num_osci]
    """
    samples, restore_shape = _prepare_feature_maps(feature_maps)
    device = _resolve_device(device, samples, encoder)
    samples = samples.to(device)
    encoder = encoder.to(device)
    encoder.eval()

    vectors = encoder(samples).cpu()
    batch_size = restore_shape["batch_size"]
    num_maps = restore_shape["num_maps"]
    return vectors.view(batch_size, num_maps, -1)


def _prepare_feature_maps(feature_maps):
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.as_tensor(feature_maps, dtype=torch.float32)
    feature_maps = feature_maps.float()

    if feature_maps.dim() == 4:
        batch_size, num_maps, height, width = feature_maps.shape
        return (
            feature_maps.reshape(batch_size * num_maps, 1, height, width),
            {"batch_size": batch_size, "num_maps": num_maps},
        )
    raise ValueError("feature_maps must have shape [B, T, H, W]. Use B=1 for one sample.")


def _resolve_device(device, tensor, module=None):
    if device is not None:
        return torch.device(device)
    if module is not None:
        return next(module.parameters()).device
    return tensor.device


if __name__ == "__main__":
    D, height, width = 6, 24, 24
    num_osci = 8
    feature_maps = torch.randn(1, D, height, width)

    encoder, autoencoder, losses = train_gamma_initializer(
        feature_maps,
        num_osci=num_osci,
        epochs=20,
        lr=1e-3,
        batch_size=3,
    )
    vectors = feature_maps_to_vectors(feature_maps, encoder)
    samples, _ = _prepare_feature_maps(feature_maps)
    reconstruction = autoencoder(samples).detach().cpu()

    print(f"initial-to-final reconstruction loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"input feature maps: {feature_maps.shape}")
    print(f"feature vectors: {vectors.shape}")
    print(f"reconstruction: {reconstruction.shape}")
