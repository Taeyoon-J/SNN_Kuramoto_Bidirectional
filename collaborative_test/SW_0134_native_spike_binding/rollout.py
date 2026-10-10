"""Native SW0130 integration rollout with an exact no-grad prefix/live tail."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0130_phase_state_integration.model import (
    COMPONENTS, PhaseStateIntegration,
)
from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


def _validate(wrapped, gamma, total_steps, live_tail_steps):
    if not isinstance(wrapped, PhaseStateIntegration):
        raise TypeError("late SW0134 rollout requires the registered SW0130 adapter")
    core = wrapped.core
    if (getattr(core, "gamma_drive_mode", None) != "static"
            or getattr(core, "spike_per_component", False) is not True
            or int(getattr(core, "osc_dim", -1)) != COMPONENTS
            or core.kuramoto.spike_pulse_gain is not None
            or (core.graph_generator is not None and core.graph_generator.uses_feedback)):
        raise ValueError("late SW0134 rollout supports only the registered static four-component source")
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (core.T, core.in_dim):
        raise ValueError(f"gamma must have static source shape [B,{core.T},{core.in_dim}]")
    if total_steps < 1 or not 0 <= live_tail_steps <= total_steps:
        raise ValueError("invalid total or live-tail step count")
    if gamma.device != next(core.parameters()).device:
        raise ValueError("gamma and wrapped source core must be on the same device")


def late_rollout(wrapped, gamma, *, total_steps=None, live_tail_steps=64):
    """Roll out the complete sequence, detaching only recurrent state at boundary.

    The static graph, gamma drive, and factorized coupling are prepared once
    with gradients enabled, consumed under no-grad during the prefix, then
    reused for the live tail. Prefix states and the full delayed theta history
    are detached at the boundary. Returned actual component spikes and native
    gates retain all time values; only the requested tail has gradient credit.
    """
    core = wrapped.core
    gamma = gamma.to(core.device)
    steps = int(total_steps if total_steps is not None else core.num_time_steps)
    tail = int(live_tail_steps)
    _validate(wrapped, gamma, steps, tail)
    boundary = steps - tail
    batch = gamma.shape[0]
    fold = COMPONENTS
    feedback = core.graph_generator is not None and core.graph_generator.uses_feedback

    # Preserve graph/drive/coupling credit across the detached state boundary.
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
    membrane_frames, spike_frames, gate_frames = [], [], []
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
            emitted = events * folded_gate
            membrane.mem, membrane.spike = integrated_mem, emitted
            membrane_frames.append(integrated_mem)
            spike_frames.append(emitted)
            gate_frames.append(folded_gate)

    # Match production's stack/permute/reshape strides and reduction order.
    component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    theta_trace = torch.stack(theta_history, dim=1)
    mean_membrane = component_membrane.mean(dim=1)
    mean_spikes = component_spikes.mean(dim=1)
    core.last_component_out = component_membrane
    core.last_component_spikes = component_spikes
    core.last_gate_history = component_gates
    return {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "component_gates": component_gates,
        "membrane": mean_membrane,
        "spikes": mean_spikes,
        "theta": theta_trace,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": 0 < tail < steps,
    }
