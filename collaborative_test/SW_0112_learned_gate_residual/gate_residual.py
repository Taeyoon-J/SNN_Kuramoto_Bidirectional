"""Nine-parameter zero-initialized residual for the registered raw scalar gate."""
from __future__ import annotations

import contextlib
import importlib

import torch
from torch import nn


class SharedGateResidual(nn.Module):
    def __init__(self, components: int = 4):
        super().__init__()
        if components != 4:
            raise ValueError("SW0112 is registered for exactly four phase components")
        self.w = nn.Parameter(torch.zeros(2 * components))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, delayed_phase, base_gate):
        if delayed_phase.shape[-1] != 4 or base_gate.shape != delayed_phase.shape[:-1]:
            raise ValueError("gate residual expects delayed phase [...,4] and scalar gate [...]")
        features = torch.cat((torch.sin(delayed_phase), torch.cos(delayed_phase)), dim=-1)
        u = torch.tanh((features * self.w).sum(dim=-1) + self.b)
        corrected = base_gate + base_gate * (1.0 - base_gate) * u
        return corrected


def attach_gate_residual(core):
    if getattr(core, "gate_mode", None) != "raw":
        raise ValueError("SW0112 requires the registered raw gate mode")
    if int(getattr(core, "phase_delay_steps", -1)) != 2:
        raise ValueError("SW0112 requires registered phase delay of two steps")
    if hasattr(core, "gate_residual"):
        raise ValueError("gate residual already attached")
    reference = next(core.parameters())
    core.gate_residual = SharedGateResidual().to(device=reference.device, dtype=reference.dtype)
    return core


@contextlib.contextmanager
def actual_gate_binding(core):
    """Scope the adapter at the sinusoidal_gating binding used by S2NetCore."""
    module = importlib.import_module(type(core).__module__)
    original = module.sinusoidal_gating
    residual = getattr(core, "gate_residual", None)
    if residual is None:
        raise ValueError("candidate core has no explicit gate_residual module")

    def wrapped(theta_hist, t, phase_delay_steps, gate_mode="sigmoid"):
        drive, base_gate = original(theta_hist, t, phase_delay_steps, gate_mode=gate_mode)
        if gate_mode != "raw":
            raise ValueError(f"SW0112 wrapper received unexpected gate mode {gate_mode!r}")
        delayed = theta_hist[max(0, t - phase_delay_steps)]
        gate = residual(delayed, base_gate)
        drive = torch.sin(theta_hist[t]) * gate.unsqueeze(-1)
        return drive, gate

    module.sinusoidal_gating = wrapped
    try:
        yield
    finally:
        module.sinusoidal_gating = original
