import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeaturePatchGammaInitializer(nn.Module):
    """
    Convert feature maps into gamma vectors by treating spatial patches as
    oscillators.

    Shape behavior:
        [B, T, H, W] -> [B, T, num_patches]

    Each gamma dimension corresponds to one spatial patch position in the
    feature map. This keeps oscillator identity tied to image/feature-map
    location instead of a learned latent dimension.
    """

    def __init__(
        self,
        grid_size=None,
        patch_size=None,
        stride=None,
        reduction="mean",
    ):
        super().__init__()
        if grid_size is None and patch_size is None:
            raise ValueError("Either grid_size or patch_size must be provided.")
        if grid_size is not None and patch_size is not None:
            raise ValueError("Use either grid_size or patch_size, not both.")
        if reduction not in {"mean", "max"}:
            raise ValueError('reduction must be "mean" or "max".')

        self.grid_size = _pair_or_none(grid_size, "grid_size")
        self.patch_size = _pair_or_none(patch_size, "patch_size")
        self.stride = _pair_or_none(stride, "stride") if stride is not None else None
        self.reduction = str(reduction)

    def forward(self, feature_maps):
        samples, restore_shape = _prepare_feature_maps(feature_maps)
        if self.grid_size is not None:
            pooled = F.adaptive_avg_pool2d(samples, self.grid_size)
        else:
            stride = self.stride if self.stride is not None else self.patch_size
            if self.reduction == "mean":
                pooled = F.avg_pool2d(samples, kernel_size=self.patch_size, stride=stride)
            else:
                pooled = F.max_pool2d(samples, kernel_size=self.patch_size, stride=stride)

        vectors = pooled.flatten(start_dim=1)
        return vectors.view(restore_shape["batch_size"], restore_shape["num_maps"], -1)

    def num_oscillators(self, feature_map_size):
        height, width = _pair(feature_map_size, "feature_map_size")
        if self.grid_size is not None:
            return int(self.grid_size[0] * self.grid_size[1])
        stride_h, stride_w = self.stride if self.stride is not None else self.patch_size
        patch_h, patch_w = self.patch_size
        out_h = (height - patch_h) // stride_h + 1
        out_w = (width - patch_w) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("patch_size is larger than feature_map_size.")
        return int(out_h * out_w)


@torch.no_grad()
def feature_maps_to_patch_gamma(
    feature_maps,
    grid_size=None,
    patch_size=None,
    stride=None,
    reduction="mean",
    device=None,
):
    """
    Convert feature maps directly to gamma sequences using spatial patches.

    Unlike ``feature_maps_to_vectors``, this path has no learned gamma
    autoencoder. The oscillator identity is the patch location.
    """
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.as_tensor(feature_maps, dtype=torch.float32)
    feature_maps = feature_maps.float()
    device = torch.device(device) if device is not None else feature_maps.device

    initializer = FeaturePatchGammaInitializer(
        grid_size=grid_size,
        patch_size=patch_size,
        stride=stride,
        reduction=reduction,
    ).to(device)
    return initializer(feature_maps.to(device)).cpu()


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


def _pair_or_none(value, name):
    if value is None:
        return None
    return _pair(value, name)


def _pair(value, name):
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError(f"{name} values must be positive.")
        return (first, second)
    raise ValueError(f"{name} must be an int or a pair of ints.")


def _resolve_device(device, tensor, module=None):
    if device is not None:
        return torch.device(device)
    if module is not None:
        return next(module.parameters()).device
    return tensor.device
