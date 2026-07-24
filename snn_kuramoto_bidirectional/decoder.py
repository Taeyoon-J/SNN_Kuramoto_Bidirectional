"""Object-wise spatial broadcast decoder.

This module follows the decoding idea used by Slot Attention:

1. Decode every object vector independently with shared parameters.
2. Produce an RGB reconstruction and a mask logit for every object.
3. Normalize masks across objects and combine the object reconstructions.

Expected input:
    object_vectors: [batch, num_objects, object_dim]

For the current S2Net design, ``object_dim`` is normally the number of
oscillators (90), and ``num_objects`` is the number of temporal intervals
returned by ``soft_classifier``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DecoderOutput(NamedTuple):
    """Outputs produced by :class:`SpatialBroadcastDecoder`."""

    reconstruction: Tensor
    object_rgb: Tensor
    masks: Tensor
    mask_logits: Tensor


class SpatialBroadcastDecoder(nn.Module):
    """Decode continuous object vectors into an image and object masks.

    Each object vector is projected to ``hidden_channels``, spatially
    broadcast over a small grid, augmented with a learnable projection of
    fixed 2D coordinates, and decoded by a CNN shared across all objects.

    Args:
        object_dim:
            Dimension of each object vector. This is normally 90.
        image_size:
            Reconstructed image size as ``(height, width)``.
        broadcast_size:
            Initial spatial size used for broadcasting object vectors.
        hidden_channels:
            Channel widths of the upsampling CNN. The first value is also
            the projected object-vector dimension.
        rgb_activation:
            ``"sigmoid"`` for images normalized to [0, 1], ``"tanh"`` for
            [-1, 1], or ``"identity"`` for unrestricted RGB output.
    """

    def __init__(
        self,
        object_dim: int = 90,
        image_size: tuple[int, int] = (128, 128),
        broadcast_size: tuple[int, int] = (8, 8),
        hidden_channels: Sequence[int] = (64, 64, 64, 64, 64),
        rgb_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        self.object_dim = int(object_dim)
        self.image_size = _pair(image_size, "image_size")
        self.broadcast_size = _pair(broadcast_size, "broadcast_size")
        self.hidden_channels = tuple(int(value) for value in hidden_channels)
        self.rgb_activation = str(rgb_activation).lower()

        if self.object_dim <= 0:
            raise ValueError("object_dim must be positive.")
        if not self.hidden_channels or any(value <= 0 for value in self.hidden_channels):
            raise ValueError("hidden_channels must contain positive integers.")
        if self.rgb_activation not in {"sigmoid", "tanh", "identity"}:
            raise ValueError(
                "rgb_activation must be 'sigmoid', 'tanh', or 'identity'."
            )

        first_channels = self.hidden_channels[0]
        self.object_projection = nn.Sequential(
            nn.LayerNorm(self.object_dim),
            nn.Linear(self.object_dim, first_channels),
            nn.ReLU(inplace=True),
        )

        # Fixed coordinates [x, y, 1-x, 1-y] are projected into the same
        # channel space as the broadcast object representation.
        self.position_projection = nn.Conv2d(
            in_channels=4,
            out_channels=first_channels,
            kernel_size=1,
        )
        self.register_buffer(
            "position_grid",
            _make_position_grid(*self.broadcast_size),
            persistent=False,
        )

        decoder_layers: list[nn.Module] = []
        in_channels = first_channels
        for out_channels in self.hidden_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        self.decoder_cnn = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Conv2d(
            in_channels,
            4,
            kernel_size=3,
            padding=1,
        )

    def forward(self, object_vectors: Tensor) -> DecoderOutput:
        """Decode ``[B, K, object_dim]`` vectors into an RGB image.

        Returns:
            ``DecoderOutput`` containing:

            - ``reconstruction``: ``[B, 3, H, W]``
            - ``object_rgb``: ``[B, K, 3, H, W]``
            - ``masks``: ``[B, K, 1, H, W]``
            - ``mask_logits``: ``[B, K, 1, H, W]``
        """

        self._validate_input(object_vectors)
        batch_size, num_objects, _ = object_vectors.shape

        flat_vectors = object_vectors.reshape(
            batch_size * num_objects,
            self.object_dim,
        )
        projected = self.object_projection(flat_vectors)

        height, width = self.broadcast_size
        broadcast = projected[:, :, None, None].expand(-1, -1, height, width)
        position = self.position_projection(
            self.position_grid.to(
                device=object_vectors.device,
                dtype=object_vectors.dtype,
            )
        )
        decoded = self.decoder_cnn(broadcast + position)
        decoded = self.output_layer(decoded)

        # This keeps the public output size fixed even when a different
        # broadcast size or number of upsampling blocks is selected.
        if decoded.shape[-2:] != self.image_size:
            decoded = F.interpolate(
                decoded,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )

        decoded = decoded.view(
            batch_size,
            num_objects,
            4,
            *self.image_size,
        )
        object_rgb = self._activate_rgb(decoded[:, :, :3])
        mask_logits = decoded[:, :, 3:4]

        # Objects compete to explain every pixel. Thus masks sum to one over
        # the object dimension for every image location.
        masks = torch.softmax(mask_logits, dim=1)
        reconstruction = torch.sum(object_rgb * masks, dim=1)

        return DecoderOutput(
            reconstruction=reconstruction,
            object_rgb=object_rgb,
            masks=masks,
            mask_logits=mask_logits,
        )

    def _activate_rgb(self, rgb: Tensor) -> Tensor:
        if self.rgb_activation == "sigmoid":
            return torch.sigmoid(rgb)
        if self.rgb_activation == "tanh":
            return torch.tanh(rgb)
        return rgb

    def _validate_input(self, object_vectors: Tensor) -> None:
        if object_vectors.ndim != 3:
            raise ValueError(
                "object_vectors must have shape "
                "[batch, num_objects, object_dim]."
            )
        if object_vectors.shape[1] <= 0:
            raise ValueError("object_vectors must contain at least one object.")
        if object_vectors.shape[2] != self.object_dim:
            raise ValueError(
                f"Expected object dimension {self.object_dim}, "
                f"but received {object_vectors.shape[2]}."
            )
        if not object_vectors.is_floating_point():
            raise TypeError("object_vectors must be a floating-point tensor.")


# Shorter name for use by the complete S2Net autoencoder.
ObjectDecoder = SpatialBroadcastDecoder


def _make_position_grid(height: int, width: int) -> Tensor:
    """Return a fixed coordinate grid shaped ``[1, 4, H, W]``."""

    y = torch.linspace(0.0, 1.0, steps=height)
    x = torch.linspace(0.0, 1.0, steps=width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack(
        (grid_x, grid_y, 1.0 - grid_x, 1.0 - grid_y),
        dim=0,
    ).unsqueeze(0)


def _pair(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive.")
    return result
