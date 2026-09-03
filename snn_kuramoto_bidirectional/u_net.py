"""Shared-weight multi-scale encoder and minimal U-Net decoder prototype."""

from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import Tensor


class SharedConvBlock(nn.Sequential):
    """Apply one 3x3 convolution followed by ReLU without changing shape."""

    def __init__(self, channels: int = 8) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=False),
        )


class SharedMultiScaleEncoder(nn.Module):
    """Encode an RGB image at four scales using two shared CNN blocks."""

    def __init__(self, feature_channels: int = 8) -> None:
        super().__init__()
        if feature_channels != 8:
            raise ValueError("This prototype requires exactly 8 feature channels.")

        self.feature_channels = feature_channels
        self.input_projection = nn.Conv2d(
            in_channels=3,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cnn_a = SharedConvBlock(feature_channels)
        self.cnn_b = SharedConvBlock(feature_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, image: Tensor) -> list[Tensor]:
        """Return raw differentiable features from 128, 64, 32, and 16 scales."""
        feature = self.input_projection(image)
        features = []

        for level_index in range(4):
            feature = self.cnn_a(feature)
            feature = self.cnn_b(feature)
            features.append(feature)

            if level_index < 3:
                feature = self.pool(feature)

        return features


class SMPUNetDecoder(nn.Module):
    """Adapt four equal-channel encoder features to SMP's U-Net decoder."""

    def __init__(self, feature_channels: int = 8) -> None:
        super().__init__()
        self.decoder = UnetDecoder(
            encoder_channels=(3,) + (feature_channels,) * 4,
            decoder_channels=(feature_channels,) * 4,
            n_blocks=4,
            use_norm=False,
            attention_type=None,
            add_center_block=False,
            interpolation_mode="bilinear",
        )
        self.segmentation_head = SegmentationHead(
            in_channels=feature_channels,
            out_channels=1,
            kernel_size=1,
            activation=None,
            upsampling=1,
        )

    def forward(self, image: Tensor, encoder_features: list[Tensor]) -> Tensor:
        feature = self.decoder([image, *encoder_features])
        return self.segmentation_head(feature)


class SharedMultiScaleUNet(nn.Module):
    """Expose the shared encoder features and one-channel decoder output."""

    def __init__(self, feature_channels: int = 8) -> None:
        super().__init__()
        self.encoder = SharedMultiScaleEncoder(feature_channels)
        self.decoder = SMPUNetDecoder(feature_channels)

    def forward(self, image: Tensor) -> dict[str, object]:
        encoder_features = self.encoder(image)
        segmentation_output = self.decoder(image, encoder_features)
        return {
            "encoder_features": encoder_features,
            "segmentation_output": segmentation_output,
        }


def run_structural_probe() -> None:
    """Run shape, reuse, pooling-count, gradient, and trainability checks."""
    model = SharedMultiScaleUNet()
    input_tensor = torch.randn(1, 3, 128, 128, requires_grad=True)

    call_counts = {"cnn_a": 0, "cnn_b": 0, "pool": 0}

    def count_call(name):
        def hook(_module, _inputs, _output):
            call_counts[name] += 1

        return hook

    handles = [
        model.encoder.cnn_a.register_forward_hook(count_call("cnn_a")),
        model.encoder.cnn_b.register_forward_hook(count_call("cnn_b")),
        model.encoder.pool.register_forward_hook(count_call("pool")),
    ]
    try:
        result = model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()

    features = result["encoder_features"]
    segmentation_output = result["segmentation_output"]
    expected_shapes = (
        (1, 8, 128, 128),
        (1, 8, 64, 64),
        (1, 8, 32, 32),
        (1, 8, 16, 16),
    )

    assert tuple(input_tensor.shape) == (1, 3, 128, 128)
    assert tuple(tuple(feature.shape) for feature in features) == expected_shapes
    assert tuple(segmentation_output.shape) == (1, 1, 128, 128)
    assert call_counts == {"cnn_a": 4, "cnn_b": 4, "pool": 3}
    assert model.encoder.cnn_a is not model.encoder.cnn_b
    assert all(feature.requires_grad for feature in features)
    assert all(parameter.requires_grad for parameter in model.parameters())

    segmentation_output.mean().backward()
    assert input_tensor.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    print("input shape:", list(input_tensor.shape))
    for level_index, feature in enumerate(features, start=1):
        print(f"Level {level_index} shape:", list(feature.shape))
    print("decoder output shape:", list(segmentation_output.shape))
    print("CNN_A and CNN_B are different modules:", model.encoder.cnn_a is not model.encoder.cnn_b)
    print("shared CNN_A call count:", call_counts["cnn_a"])
    print("shared CNN_B call count:", call_counts["cnn_b"])
    print("AvgPool2d call count:", call_counts["pool"])
    print("all parameters trainable:", all(parameter.requires_grad for parameter in model.parameters()))
    print("backward gradients verified: True")


if __name__ == "__main__":
    run_structural_probe()
