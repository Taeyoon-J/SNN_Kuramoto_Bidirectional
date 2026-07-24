"""Dynamic structural-connectivity generation from gamma vectors."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class SCOutput(NamedTuple):
    """Connectivity matrices produced during one forward pass."""

    sc: Tensor
    batch_sc: Tensor
    running_sc: Tensor


class DynamicSCGenerator(nn.Module):
    """Compute oscillator connectivity from the current gamma batch.

    Gamma dimensions are treated as oscillators. Batch and temporal/feature-map
    axes are combined into one sample axis, and absolute Pearson correlation is
    calculated between oscillator columns.

    During training, the differentiable batch SC is returned for immediate use
    by S2NetCore, while a detached exponential moving average is accumulated.
    During evaluation, the accumulated running SC is returned so the result
    does not depend on evaluation batch composition. If no training update has
    occurred yet, evaluation falls back to the current batch SC.

    Args:
        num_oscillators:
            Number of gamma dimensions.
        momentum:
            EMA weight assigned to the previous running SC.
        eps:
            Numerical stability value for zero-variance gamma dimensions.
    """

    def __init__(
        self,
        num_oscillators: int,
        momentum: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_oscillators = int(num_oscillators)
        self.momentum = float(momentum)
        self.eps = float(eps)

        if self.num_oscillators <= 0:
            raise ValueError("num_oscillators must be positive.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.register_buffer(
            "running_sc",
            torch.zeros(self.num_oscillators, self.num_oscillators),
        )
        self.register_buffer(
            "num_updates",
            torch.zeros((), dtype=torch.long),
        )

    def forward(self, gamma: Tensor) -> SCOutput:
        """Return SC matrices for ``gamma [B,T,N]``."""

        self._validate_gamma(gamma)
        batch_sc = pearson_cor_sc(gamma, eps=self.eps)

        if self.training:
            self._update_running_sc(batch_sc)
            effective_sc = batch_sc
        elif self.num_updates.item() > 0:
            effective_sc = self.running_sc.to(
                device=gamma.device,
                dtype=gamma.dtype,
            )
        else:
            effective_sc = batch_sc

        return SCOutput(
            sc=effective_sc,
            batch_sc=batch_sc,
            running_sc=self.running_sc.to(
                device=gamma.device,
                dtype=gamma.dtype,
            ).clone(),
        )

    @torch.no_grad()
    def _update_running_sc(self, batch_sc: Tensor) -> None:
        detached = batch_sc.detach().to(
            device=self.running_sc.device,
            dtype=self.running_sc.dtype,
        )
        if self.num_updates.item() == 0:
            self.running_sc.copy_(detached)
        else:
            self.running_sc.mul_(self.momentum).add_(
                detached,
                alpha=1.0 - self.momentum,
            )
        self.num_updates.add_(1)

    def reset_running_sc(self) -> None:
        """Clear accumulated SC statistics."""

        self.running_sc.zero_()
        self.num_updates.zero_()

    def _validate_gamma(self, gamma: Tensor) -> None:
        if gamma.ndim != 3:
            raise ValueError(
                "gamma must have shape [batch, num_steps, num_oscillators]."
            )
        if gamma.shape[2] != self.num_oscillators:
            raise ValueError(
                f"Expected {self.num_oscillators} gamma dimensions, "
                f"got {gamma.shape[2]}."
            )
        if gamma.shape[0] * gamma.shape[1] < 2:
            raise ValueError(
                "At least two gamma samples across batch and time are "
                "required to calculate Pearson correlation."
            )
        if not gamma.is_floating_point():
            raise TypeError("gamma must be a floating-point tensor.")


def pearson_cor_sc(gamma: Tensor, eps: float = 1e-8) -> Tensor:
    """Calculate absolute Pearson SC without detaching the computation graph.

    Args:
        gamma:
            Tensor shaped ``[B,T,N]`` or pre-flattened ``[S,N]``.
        eps:
            Numerical stability value used for zero-variance dimensions.

    Returns:
        Symmetric tensor ``[N,N]`` with connectivity strengths in ``[0,1]``.
    """

    if not torch.is_tensor(gamma):
        gamma = torch.as_tensor(gamma, dtype=torch.float32)
    if not gamma.is_floating_point():
        gamma = gamma.float()

    if gamma.ndim == 3:
        samples = gamma.flatten(0, 1)
    elif gamma.ndim == 2:
        samples = gamma
    else:
        raise ValueError("gamma must have shape [B,T,N] or [S,N].")
    if samples.shape[0] < 2:
        raise ValueError("At least two gamma samples are required.")

    centered = samples - samples.mean(dim=0, keepdim=True)
    column_norms = torch.linalg.vector_norm(
        centered,
        dim=0,
        keepdim=True,
    )
    normalized = centered / column_norms.clamp_min(float(eps))
    correlation = normalized.transpose(0, 1) @ normalized

    # Preserve the previous SC definition: positive and negative Pearson
    # correlations both represent connectivity strength.
    return correlation.abs().clamp(0.0, 1.0)
