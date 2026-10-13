"""Zero-initialized spatial residual wrapper for the registered RGB encoder."""
from __future__ import annotations

import torch
from torch import nn


class SourcePreservingDeepEncoder(nn.Module):
    """Add a wider-receptive-field residual while preserving the source at init.

    ``base_encoder`` must map NCHW RGB 128x128 to [B,8,126,126].  The wrapper
    freezes it by default; callers may explicitly unfreeze it for a registered
    training recipe. Inputs are float32 RGB in [0,1].
    """

    def __init__(self, base_encoder: nn.Module, *, freeze_base: bool = True):
        super().__init__()
        if not isinstance(base_encoder, nn.Module):
            raise TypeError("base_encoder must be a torch module")
        self.base_encoder = base_encoder
        self.residual = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.SiLU(),
            nn.Conv2d(32, 8, kernel_size=1),
        )
        final = self.residual[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if freeze_base:
            self.base_encoder.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(rgb) or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 128, 128):
            raise ValueError("RGB input must have shape [B,3,128,128]")
        if rgb.dtype != torch.float32:
            raise TypeError("RGB input must be float32")
        if not torch.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("RGB input must be finite and in [0,1]")
        base = self.base_encoder(rgb)
        if base.ndim != 4 or tuple(base.shape[1:]) != (8, 126, 126):
            raise ValueError("base encoder must return [B,8,126,126]")
        residual = self.residual(rgb)[:, :, 1:-1, 1:-1]
        return base + residual

