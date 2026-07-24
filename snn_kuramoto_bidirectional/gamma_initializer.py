import torch
import torch.nn as nn


class FeatureMapCNNEncoder(nn.Module):
    """
    Convert each feature map independently into one position-aware gamma vector.

    A fixed 2D Fourier position encoding is concatenated with a single feature
    map before nonlinear CNN processing. Local max-pooling reduces resolution
    while preserving a spatial grid. The grid is flattened, rather than
    globally averaged, so every gamma dimension can learn different weights for
    different spatial locations.

    Expected input:
        [B, in_channels, H, W]

    Output:
        [B, num_osci]
    """

    def __init__(
        self,
        num_osci,
        in_channels=1,
        hidden_channels=(16, 32, 64),
        dropout=0.0,
        input_size=None,
        position_frequencies=(1.0, 2.0, 4.0, 8.0),
    ):
        super().__init__()
        self.num_osci = int(num_osci)
        self.in_channels = int(in_channels)
        self.hidden_channels = tuple(int(channel) for channel in hidden_channels)
        self.input_size = _validate_optional_input_size(input_size)
        self.position_frequencies = tuple(
            float(frequency) for frequency in position_frequencies
        )

        if self.num_osci <= 0:
            raise ValueError("num_osci must be positive.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if not self.hidden_channels or any(
            channel <= 0 for channel in self.hidden_channels
        ):
            raise ValueError("hidden_channels must contain positive integers.")
        if any(frequency <= 0 for frequency in self.position_frequencies):
            raise ValueError("position_frequencies must contain positive values.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        position_channels = 2 + 4 * len(self.position_frequencies)
        layers = []
        current_channels = self.in_channels + position_channels
        for out_channels in self.hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_channels = out_channels
        self.spatial_encoder = nn.Sequential(*layers)

        if self.input_size is None:
            projection = nn.LazyLinear(self.num_osci)
        else:
            spatial_height, spatial_width = _downsampled_size(
                self.input_size,
                num_stages=len(self.hidden_channels),
            )
            projection = nn.Linear(
                current_channels * spatial_height * spatial_width,
                self.num_osci,
            )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            projection,
        )

    def forward(self, feature_maps):
        spatial_features = self.encode_spatial(feature_maps)
        return self.projection(spatial_features)

    def forward_with_spatial_features(self, feature_maps):
        """Return gamma together with the pre-flatten spatial representation."""
        spatial_features = self.encode_spatial(feature_maps)
        gamma = self.projection(spatial_features)
        return gamma, spatial_features

    def encode_spatial(self, feature_maps):
        self._validate_feature_maps(feature_maps)
        position = make_2d_fourier_position_encoding(
            height=feature_maps.size(-2),
            width=feature_maps.size(-1),
            frequencies=self.position_frequencies,
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        position = position.expand(feature_maps.size(0), -1, -1, -1)
        position_aware_features = torch.cat([feature_maps, position], dim=1)
        return self.spatial_encoder(position_aware_features)

    def _validate_feature_maps(self, feature_maps):
        if feature_maps.dim() != 4:
            raise ValueError("feature_maps must have shape [B, C, H, W].")
        if feature_maps.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got "
                f"{feature_maps.size(1)}."
            )
        if (
            self.input_size is not None
            and tuple(feature_maps.shape[-2:]) != self.input_size
        ):
            raise ValueError(
                f"Expected feature-map size {self.input_size}, but got "
                f"{tuple(feature_maps.shape[-2:])}."
            )


def make_2d_fourier_position_encoding(
    height,
    width,
    frequencies=(1.0, 2.0, 4.0, 8.0),
    device=None,
    dtype=torch.float32,
):
    """
    Create fixed absolute 2D coordinates and Fourier features.

    With four frequencies the returned tensor has 18 channels:
        x, y, and four sin/cos channels per frequency.
    """
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")

    frequencies = tuple(float(frequency) for frequency in frequencies)
    if any(frequency <= 0 for frequency in frequencies):
        raise ValueError("frequencies must contain positive values.")

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    channels = [xx, yy]
    for frequency in frequencies:
        scaled_x = torch.pi * frequency * xx
        scaled_y = torch.pi * frequency * yy
        channels.extend(
            [
                torch.sin(scaled_x),
                torch.cos(scaled_x),
                torch.sin(scaled_y),
                torch.cos(scaled_y),
            ]
        )
    return torch.stack(channels, dim=0).unsqueeze(0)


def _validate_optional_input_size(input_size):
    if input_size is None:
        return None
    if not isinstance(input_size, (tuple, list)) or len(input_size) != 2:
        raise ValueError("input_size must be a (height, width) pair.")
    height, width = int(input_size[0]), int(input_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("input_size values must be positive.")
    return height, width


def _downsampled_size(input_size, num_stages):
    height, width = input_size
    for _ in range(int(num_stages)):
        height //= 2
        width //= 2
        if height <= 0 or width <= 0:
            raise ValueError(
                "input_size is too small for the requested number of pooling stages."
            )
    return height, width


# ---------------------------------------------------------------------------
# Not particularly in use
# Retained only for optional standalone feature-map-to-gamma conversion.
# The end-to-end GammaGenerator path does not call these helpers.
# ---------------------------------------------------------------------------


@torch.no_grad()
def feature_maps_to_vectors(feature_maps, encoder, device=None):
    """
    Convert feature maps independently into position-aware gamma vectors.

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
