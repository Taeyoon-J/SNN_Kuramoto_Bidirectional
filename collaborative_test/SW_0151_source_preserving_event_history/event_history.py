"""Standalone adaptive event-history kernel for prospective SW0151 work.

This module does not alter the native rollout or its membrane/reset equation.
It only computes events and emitted spikes from an already-computed native
membrane value, native gate, and the caller's existing positive threshold.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp


COMPONENTS = 4
INITIAL_LAMBDA = 0.95


def _component_values(value: torch.Tensor, *, name: str, ref: torch.Tensor) -> torch.Tensor:
    if value.shape not in (torch.Size([COMPONENTS]), torch.Size([1, COMPONENTS, 1])):
        raise ValueError(f"{name} must have shape [4] or [1,4,1]")
    value = value.reshape(1, COMPONENTS, 1).to(device=ref.device, dtype=ref.dtype)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


class AdaptiveEventHistory(nn.Module):
    """Threshold-history event kernel with four componentwise lambda/kappa.

    Inputs use [B,4,N]. ``base_threshold`` is the caller's existing native
    ``0.06 * exp(b)`` value. ``scale`` is a fixed positive TRAIN calibration.
    ``previous_event`` and ``history`` are recurrent states from the prior
    step. Returned next states remain connected by default across live rollout
    steps; pass ``detach_state=True`` at the registered truncation boundary.
    """

    def __init__(self, *, initial_lambda: float = INITIAL_LAMBDA):
        super().__init__()
        if not math.isfinite(float(initial_lambda)) or not 0.0 < float(initial_lambda) < 1.0:
            raise ValueError("initial_lambda must be finite and strictly between zero and one")
        logit = math.log(float(initial_lambda) / (1.0 - float(initial_lambda)))
        self.lambda_logit = nn.Parameter(torch.full((COMPONENTS,), logit, dtype=torch.float32))
        # Projected nonnegative; exactly zero initialization preserves native events.
        self.kappa_raw = nn.Parameter(torch.zeros(COMPONENTS, dtype=torch.float32))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logit)

    @property
    def kappa(self) -> torch.Tensor:
        return self.kappa_raw.clamp_min(0.0)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.kappa_raw.clamp_(min=0.0)

    def forward(
        self,
        membrane: torch.Tensor,
        gate: torch.Tensor,
        base_threshold: torch.Tensor,
        scale: float | torch.Tensor,
        previous_event: torch.Tensor,
        history: torch.Tensor,
        *,
        detach_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        if membrane.ndim != 3 or membrane.shape[1] != COMPONENTS or membrane.shape[0] < 1 or membrane.shape[2] < 1:
            raise ValueError("membrane must have nonempty shape [B,4,N]")
        if membrane.dtype != torch.float32 or not bool(torch.isfinite(membrane).all()):
            raise ValueError("native membrane must be finite float32")
        expected = membrane.shape
        if gate.shape not in (expected, torch.Size([expected[0], COMPONENTS, 1])):
            raise ValueError("gate must have shape [B,4,N] or [B,4,1]")
        if gate.device != membrane.device or gate.dtype != torch.float32 or not bool(torch.isfinite(gate).all()):
            raise ValueError("native gate must be finite float32 on the membrane device")
        if bool((gate < 0).any()):
            raise ValueError("gate must be nonnegative")
        if previous_event.shape != expected or history.shape != expected:
            raise ValueError("previous_event and history must match membrane [B,4,N]")
        if previous_event.device != membrane.device or history.device != membrane.device:
            raise ValueError("recurrent states must be on the membrane device")
        if not previous_event.is_floating_point() or not history.is_floating_point():
            raise ValueError("recurrent states must be floating point")
        if previous_event.dtype != membrane.dtype or history.dtype != membrane.dtype:
            raise ValueError("recurrent states must use the membrane dtype")
        if not bool(torch.isfinite(previous_event).all()) or not bool(torch.isfinite(history).all()):
            raise ValueError("recurrent states must be finite")
        if not bool(torch.isfinite(self.lambda_logit).all()) or not bool(torch.isfinite(self.kappa_raw).all()):
            raise ValueError("event-history parameters must be finite")

        threshold = _component_values(base_threshold, name="base_threshold", ref=membrane)
        if bool((threshold <= 0).any()):
            raise ValueError("base_threshold must be strictly positive")
        scale_t = torch.as_tensor(scale)
        if scale_t.requires_grad or scale_t.ndim > 1 or (scale_t.ndim == 1 and scale_t.numel() != COMPONENTS):
            raise ValueError("scale must be a fixed scalar or fixed four-vector")
        scale_t = scale_t.to(device=membrane.device, dtype=membrane.dtype)
        if scale_t.numel() not in (1, COMPONENTS) or not bool(torch.isfinite(scale_t).all()) or bool((scale_t <= 0).any()):
            raise ValueError("scale must be a finite positive scalar or four-vector")
        scale_t = scale_t.reshape(1, -1, 1)
        if scale_t.size(1) == 1:
            scale_t = scale_t.expand(1, COMPONENTS, 1)

        lam = self.decay.to(device=membrane.device, dtype=membrane.dtype).view(1, COMPONENTS, 1)
        kappa = self.kappa.to(device=membrane.device, dtype=membrane.dtype).view(1, COMPONENTS, 1)
        gate_full = gate.expand_as(membrane) if gate.size(2) == 1 else gate
        active = gate_full != 0

        # History uses the previous native event. Exact zero gates hold both
        # auxiliary states, matching native membrane hold semantics.
        updated_history = lam * history + (1.0 - lam) * previous_event
        next_history = torch.where(active, updated_history, history)
        adaptive_threshold = threshold + kappa * scale_t * next_history
        event = act_fun_adp(membrane - adaptive_threshold)
        emitted = event * gate_full
        next_event = torch.where(active, event, previous_event)

        if detach_state:
            next_event = next_event.detach()
            next_history = next_history.detach()
        return {
            "event": event,
            "spike": emitted,
            "history": next_history,
            "previous_event": next_event,
            "threshold": adaptive_threshold,
        }
