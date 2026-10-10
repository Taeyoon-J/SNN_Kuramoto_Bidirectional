"""SW0139 zero-output multiscale context residual adapter."""
from __future__ import annotations

import torch
from torch import nn


class ContextResidualEncoder(nn.Module):
    """Contextualize the registered 8x126x126 map without adding labels/XY."""
    DILATIONS = (1, 4, 16, 32)

    def __init__(self, seed: int = 139):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.branches = nn.ModuleList([
                nn.Conv2d(8, 16, kernel_size=3, padding=d, dilation=d)
                for d in self.DILATIONS
            ])
        self.activation = nn.GELU()
        self.projection = nn.Conv2d(64, 8, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 8 or tuple(features.shape[-2:]) != (126, 126):
            raise ValueError("context adapter expects standardized [B,8,126,126]")
        residual = self.projection(torch.cat(
            [self.activation(branch(features)) for branch in self.branches], dim=1))
        return (features + residual).clamp(-3.0, 3.0)
