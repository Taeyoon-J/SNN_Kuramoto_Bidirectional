from __future__ import annotations
import torch
from torch import nn

class ContextualFeatureEncoder(nn.Module):
    """Frozen source CNN plus a zero-initialized, trainable contextual residual.

    The wrapped source encoder must map NCHW RGB 128x128 to [B,8,126,126].
    Inputs are the existing float32 RGB values in [0,1]; normalization, clipping,
    and native32 pooling remain the responsibility of the registered foundation.
    """

    def __init__(self, base_encoder: nn.Module, *, seed: int = 171001):
        super().__init__()
        if not isinstance(base_encoder, nn.Module):
            raise TypeError("base_encoder must be a torch module")
        parameters = list(base_encoder.parameters())
        if not parameters:
            raise ValueError("base encoder must expose its registered kernel parameters")
        if tuple(parameters[0].shape) != (8, 3, 3, 3):
            raise ValueError("source encoder must use the registered 8x3x3x3 kernels")
        device, dtype = parameters[0].device, parameters[0].dtype
        if dtype != torch.float32:
            raise TypeError("registered source encoder kernels must be float32")
        self.base_encoder = base_encoder
        self.base_encoder.requires_grad_(False)
        # Seed only the CPU generator used by these CPU-created convolutions.
        # fork_rng restores the caller's CPU state and does not touch CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.context = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, dilation=1),
                nn.SiLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),
                nn.SiLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4),
                nn.SiLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=8, dilation=8),
                nn.SiLU(),
                nn.Conv2d(32, 8, kernel_size=1),
            )
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)
        self.context.to(device=device, dtype=dtype)
        self.seed = int(seed)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(rgb) or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 128, 128):
            raise ValueError("RGB input must have shape [B,3,128,128]")
        if rgb.dtype != torch.float32:
            raise TypeError("RGB input must be float32")
        if not bool(torch.isfinite(rgb).all()) or bool((rgb < 0).any()) or bool((rgb > 1).any()):
            raise ValueError("RGB input must be finite and in [0,1]")
        base = self.base_encoder(rgb)
        if base.ndim != 4 or tuple(base.shape[1:]) != (8, 126, 126):
            raise ValueError("source encoder must return [B,8,126,126]")
        residual = self.context(rgb)[:, :, 1:-1, 1:-1]
        if residual.shape != base.shape:
            raise AssertionError("context residual crop does not align with source feature maps")
        return base + residual
