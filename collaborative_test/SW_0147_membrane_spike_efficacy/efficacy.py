"""Local SW0147 membrane-conditioned efficacy kernel and rollout.

This is a leaf implementation: it does not patch the frozen SW0130/SW0134
modules. The native event and gate are computed first; the bounded residual
is then added and the resulting spike is used for both reset and readout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0130_phase_state_integration.model import (  # noqa: E402
    COMPONENTS, PhaseStateIntegration,
)
from collaborative_test.SW_0134_native_spike_binding.rollout import _validate  # noqa: E402
from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp  # noqa: E402
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating  # noqa: E402


class MembraneSpikeEfficacy(nn.Module):
    """Four shared bounded efficacy coefficients and frozen TRAIN calibration."""

    def __init__(self, mu: torch.Tensor, scale: torch.Tensor, *, device=None):
        super().__init__()
        mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        if mu.shape != (COMPONENTS,) or scale.shape != (COMPONENTS,):
            raise ValueError("mu and scale must each have four component values")
        if not torch.isfinite(mu).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("calibration must be finite with strictly positive scales")
        self.register_buffer("mu", mu.detach().clone())
        self.register_buffer("scale", scale.detach().clone())
        self.rho = nn.Parameter(torch.zeros(COMPONENTS, device=mu.device))

    def efficacy(self, margin: torch.Tensor, component_ids: torch.Tensor,
                 *, credit: str = "live") -> tuple[torch.Tensor, torch.Tensor]:
        if credit not in ("live", "detached"):
            raise ValueError("credit must be live or detached")
        if margin.ndim != 2 or component_ids.ndim != 1 or margin.shape[0] != component_ids.numel():
            raise ValueError("margin must be [component_rows, units] with one component ID per row")
        ids = component_ids.to(device=margin.device, dtype=torch.long)
        phi = torch.tanh((margin - self.mu[ids, None]) / self.scale[ids, None])
        phi_credit = phi if credit == "live" else phi.detach()
        coeff = 0.5 * torch.tanh(self.rho[ids])
        return coeff[:, None] * phi_credit, phi


def efficacy_rollout(wrapped: PhaseStateIntegration, gamma: torch.Tensor,
                     efficacy: MembraneSpikeEfficacy, *, total_steps: int | None = None,
                     live_tail_steps: int = 64, credit: str = "live") -> dict:
    """SW0134 rollout with only the emitted-spike amplitude path augmented."""
    core = wrapped.core
    gamma = gamma.to(core.device)
    steps = int(total_steps if total_steps is not None else core.num_time_steps)
    tail = int(live_tail_steps)
    _validate(wrapped, gamma, steps, tail)
    if efficacy.rho.device != gamma.device:
        raise ValueError("efficacy and source core must be on the same device")
    if credit not in ("live", "detached"):
        raise ValueError("credit must be live or detached")

    boundary = steps - tail
    batch = gamma.shape[0]
    fold = COMPONENTS
    feedback = core.graph_generator is not None and core.graph_generator.uses_feedback
    setup_context = torch.enable_grad() if tail else torch.no_grad()
    with setup_context:
        graph = (core.graph_generator(gamma) if core.graph_generator is not None
                 else core.sc.to(gamma.device).unsqueeze(0).expand(batch, -1, -1))
        drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
        theta = core._init_theta(drive, batch)
        coupling = None
        if core.kuramoto_backend == "factorized" and not feedback:
            coupling = core.kuramoto.prepare_coupling(
                graph, batch_size=batch, num_units=core.in_dim, device=gamma.device)

    theta_history = []
    membrane_frames, spike_frames, native_frames, event_frames, gate_frames = [], [], [], [], []
    previous_spike_frames = []
    core.dendric_layer.set_neuron_state(batch * fold)
    core.membrane_layer.set_neuron_state(batch * fold)
    dendrite, membrane = core.dendric_layer, core.membrane_layer

    for t in range(steps):
        if t == boundary and boundary > 0:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            dendrite.h = dendrite.h.detach()
            membrane.mem = membrane.mem.detach()
            membrane.spike = membrane.spike.detach()

        with torch.set_grad_enabled(t >= boundary):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(
                theta_history, t, core.phase_delay_steps, gate_mode=core.gate_mode)
            folded_carrier = carrier.permute(0, 2, 1).reshape(batch * fold, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(fold, dim=0)
            integration_gate = (folded_gate if wrapped.arm == "phase"
                                else torch.full_like(folded_gate, 0.5))

            previous_h = dendrite.h
            previous_mem = membrane.mem
            previous_spike = membrane.spike
            dense = wrapped._dense_dendrite(dendrite, folded_carrier, previous_spike)
            beta_d = torch.sigmoid(dendrite.tau_n).unsqueeze(0)
            native_h = beta_d * previous_h + (1.0 - beta_d) * dense
            component_ids = torch.arange(fold, device=gamma.device).repeat(batch)
            a_d = wrapped.a_d[component_ids].view(batch * fold, 1, 1)
            u = integration_gate.unsqueeze(-1)
            integrated_h = native_h + a_d * (u - 1.0) * (native_h - previous_h)
            dendrite.h = integrated_h
            h_wave = integrated_h.sum(dim=2)

            alpha = torch.sigmoid(membrane.tau_m).unsqueeze(0)
            threshold = 0.06 * torch.exp(wrapped.b[component_ids].view(batch * fold, 1))
            native_mem = (previous_mem * alpha + (1.0 - alpha) * R_m * h_wave
                          - threshold * previous_spike)
            a_m = wrapped.a_m[component_ids].view(batch * fold, 1)
            integrated_mem = native_mem + a_m * (integration_gate - 1.0) * (
                native_mem - previous_mem)
            integrated_mem = torch.where(folded_gate == 0, previous_mem, integrated_mem)
            events = act_fun_adp(integrated_mem - threshold)
            native_emitted = events * folded_gate
            efficacy_factor, phi = efficacy.efficacy(
                integrated_mem - threshold, component_ids, credit=credit)
            emitted = native_emitted + efficacy_factor * native_emitted

            # This assignment is the recurrent reset path as well as the output.
            membrane.mem, membrane.spike = integrated_mem, emitted
            membrane_frames.append(integrated_mem)
            spike_frames.append(emitted)
            native_frames.append(native_emitted)
            event_frames.append(events)
            gate_frames.append(folded_gate)
            previous_spike_frames.append(previous_spike)

    component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_native_spikes = torch.stack(native_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_events = torch.stack(event_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_previous_spikes = torch.stack(previous_spike_frames, dim=0).permute(
        1, 2, 0).reshape(batch, fold, core.in_dim, steps)
    theta_trace = torch.stack(theta_history, dim=1)
    component_membrane = component_membrane
    mean_membrane = component_membrane.mean(dim=1)
    mean_spikes = component_spikes.mean(dim=1)
    core.last_component_out = component_membrane
    core.last_component_spikes = component_spikes
    core.last_gate_history = component_gates
    return {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "component_native_spikes": component_native_spikes,
        "component_events": component_events,
        "component_previous_spikes": component_previous_spikes,
        "component_gates": component_gates,
        "membrane": mean_membrane,
        "spikes": mean_spikes,
        "theta": theta_trace,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": 0 < tail < steps,
    }
