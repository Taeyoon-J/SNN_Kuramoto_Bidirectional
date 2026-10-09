"""Full 1024-step SW0126 rollout with a no-grad prefix and live 64-step tail."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SNN = ROOT / "snn_kuramoto_bidirectional"
if str(SNN) not in sys.path:
    sys.path.insert(0, str(SNN))
from sinusoidal_gating import sinusoidal_gating


def rollout(core, gamma, total_steps=1024, settle=512, live_tail_steps=64):
    """Run full forward values and truncated late-tail gradients.

    Static graph/drive/coupling are prepared once with gradients enabled and
    reused after the no-grad prefix. Only recurrent hidden values detach at the
    boundary; the history-event adapter also detaches its captured prefix.
    """
    if (core.gamma_drive_mode != "static" or core.osc_dim != 4
            or core.spike_per_component is not True
            or core.graph_generator is None or core.graph_generator.uses_feedback
            or core.kuramoto.spike_pulse_gain is not None):
        raise ValueError("SW0126 pilot requires the registered static D4, no-feedback, no-pulse core")
    if total_steps != 1024 or settle != 512 or live_tail_steps not in (0, 64):
        raise ValueError("SW0126 pilot rollout is fixed at T1024, settle512, tail64 or no-grad")
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (core.T, core.in_dim):
        raise ValueError("gamma must be static [B,8,256]")
    gamma = gamma.to(core.device)
    batch = gamma.shape[0]
    boundary = total_steps - live_tail_steps
    setup = torch.enable_grad() if live_tail_steps else torch.no_grad()
    with setup:
        graph = core.graph_generator(gamma)
        drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
        theta = core._init_theta(drive, batch)
        coupling = None
        if core.kuramoto_backend == "factorized":
            coupling = core.kuramoto.prepare_coupling(
                graph, batch_size=batch, num_units=core.in_dim, device=gamma.device)
    folded = core.osc_dim
    core.dendric_layer.set_neuron_state(batch * folded)
    core.membrane_layer.set_neuron_state(batch * folded)
    theta_history, carrier_history, gate_history = [], [], []
    membrane_history, spike_history = [], []
    for step in range(total_steps):
        if live_tail_steps and step == boundary:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            core.dendric_layer.h = core.dendric_layer.h.detach()
            membrane = core.membrane_layer
            if hasattr(membrane, "detach_rollout_boundary"):
                membrane.detach_rollout_boundary()
            else:
                membrane.mem = membrane.mem.detach()
                membrane.spike = membrane.spike.detach()
        with torch.set_grad_enabled(bool(live_tail_steps and step >= boundary)):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(
                theta_history, step, core.phase_delay_steps, gate_mode=core.gate_mode)
            folded_wave = carrier.permute(0, 2, 1).reshape(batch * folded, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(folded, dim=0)
            h_wave = core.dendric_layer(folded_wave, core.membrane_layer.spike)
            membrane, spike = core.membrane_layer(h_wave, folded_gate)
            membrane_history.append(membrane.reshape(batch, folded, core.in_dim))
            spike_history.append(spike.reshape(batch, folded, core.in_dim))
        carrier_history.append(carrier)
        gate_history.append(gate)
    with torch.set_grad_enabled(live_tail_steps > 0):
        component_membrane = torch.stack(membrane_history, dim=0).permute(1, 2, 3, 0).reshape(
            batch, folded, core.in_dim, total_steps)
        component_spikes = torch.stack(spike_history, dim=0).permute(1, 2, 3, 0).reshape(
            batch, folded, core.in_dim, total_steps)
    return {
        # Match the production core layout [B,D,N,T] exactly.
        "theta": torch.stack(theta_history, dim=-1),
        "carrier": torch.stack(carrier_history, dim=-1),
        "gate": torch.stack(gate_history, dim=-1),
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "spikes": component_spikes.mean(dim=1),
        "truncated_bptt": bool(live_tail_steps),
        "prefix_steps": boundary,
        "live_tail_steps": live_tail_steps,
    }
