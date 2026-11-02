"""Opt-in SW0123 membrane dynamics; production S2Net modules stay untouched."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from snn_kuramoto_bidirectional.membrane_layer import MembraneLayer, act_fun_adp


COMPONENTS = 4
VTH = 0.06
RETENTION = 0.9
ADAPTATION_DECAY = 0.9
ADAPTATION_INPUT = 0.1


class AdaptiveMembraneLayer(MembraneLayer):
    """MembraneLayer replacement preserving tau_m name and adding adaptive state."""

    def __init__(self, source: MembraneLayer, kappa: torch.Tensor):
        super().__init__(source.output_dim, tau_minitializer="constant", low_m=0.0,
                         vth=source.vth, dt=source.dt, device=source.device)
        self.tau_m.data.copy_(source.tau_m.detach())
        if abs(float(source.vth) - VTH) > 1e-12:
            raise ValueError(f"SW0123 requires fixed reset/vth {VTH}; source has {source.vth}")
        kappa = torch.as_tensor(kappa, dtype=self.tau_m.dtype, device=self.tau_m.device).reshape(-1)
        if kappa.shape != (COMPONENTS,) or not torch.isfinite(kappa).all() or (kappa <= 0).any():
            raise ValueError("kappa must be four finite positive TRAIN-calibrated values")
        self.register_buffer("kappa", kappa.detach().clone())
        raw_base = math.log(math.expm1(VTH))
        self.raw_b = torch.nn.Parameter(torch.full((COMPONENTS,), raw_base,
                                                   dtype=self.tau_m.dtype,
                                                   device=self.tau_m.device))
        self.tau_m.data.fill_(math.log(RETENTION / (1.0 - RETENTION)))
        self.adaptation = None
        self.previous_event = None
        self.capture_event_history = False
        self.event_history = None

    def set_neuron_state(self, batch_size):
        if batch_size % COMPONENTS:
            raise ValueError("folded membrane batch must preserve four component rows per image")
        device = self.tau_m.device
        self.mem = torch.zeros(batch_size, self.output_dim, device=device)
        self.spike = torch.zeros(batch_size, self.output_dim, device=device)
        self.v_th = torch.full((batch_size, self.output_dim), VTH, device=device)
        self.adaptation = torch.zeros(batch_size, self.output_dim, device=device)
        self.previous_event = torch.zeros(batch_size, self.output_dim, device=device)
        self.event_history = [] if self.capture_event_history else None

    def forward(self, h_wave_t, g_wave_t):
        if self.mem is None or self.spike is None or self.adaptation is None:
            raise RuntimeError("call set_neuron_state before adaptive membrane forward")
        gate = g_wave_t.expand_as(self.mem)
        if h_wave_t.shape != self.mem.shape or gate.shape != self.mem.shape:
            raise ValueError("adaptive membrane input/gate shape mismatch")
        active = gate > 0
        candidate_a = ADAPTATION_DECAY * self.adaptation + ADAPTATION_INPUT * self.previous_event
        next_a = torch.where(active, candidate_a, self.adaptation)
        alpha = torch.sigmoid(self.tau_m).unsqueeze(0)
        candidate_mem = alpha * self.mem + (1.0 - alpha) * h_wave_t - VTH * self.spike
        next_mem = torch.where(active, candidate_mem, self.mem)
        components = torch.arange(self.mem.shape[0], device=self.mem.device) % COMPONENTS
        base_threshold = F.softplus(self.raw_b[components]).unsqueeze(1)
        adaptive_threshold = base_threshold + self.kappa[components].unsqueeze(1) * next_a
        event = act_fun_adp(next_mem - adaptive_threshold) * active.to(next_mem.dtype)
        actual_spike = event * gate
        self.mem = next_mem
        self.adaptation = next_a
        self.previous_event = event
        self.spike = actual_spike
        if self.event_history is not None:
            self.event_history.append(event)
        return self.mem, self.spike


def install_adaptive_dynamics(core, kappa):
    """Install candidate dynamics after strict-loading the original source core."""
    if getattr(core, "spike_per_component", False) is not True or getattr(core, "osc_dim", None) != COMPONENTS:
        raise ValueError("SW0123 requires the registered four-component folded S2Net core")
    dendrite_tau = core.dendric_layer.tau_n
    if dendrite_tau.ndim != 2:
        raise ValueError("unexpected dendritic retention parameter shape")
    with torch.no_grad():
        dendrite_tau.fill_(math.log(RETENTION / (1.0 - RETENTION)))
    core.membrane_layer = AdaptiveMembraneLayer(core.membrane_layer, kappa).to(dendrite_tau.device)
    return core
