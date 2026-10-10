"""Instance-local SW0140 interventions for the frozen native32 source."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0130_phase_state_integration.model import (
    BASE_THRESHOLD, COMPONENTS, PhaseStateIntegration,
)
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating

INTERVENTIONS = (
    "kuramoto_interaction_off",
    "delayed_gate_constant_half",
    "dendrite_history_off",
    "membrane_voltage_history_off",
    "spike_reset_off",
    "binary_events_forced_on",
)
CONDITIONS = ("none", *INTERVENTIONS)
TRACE_FIELDS = ("theta", "membrane", "spikes", "component_membrane",
                "component_spikes", "component_gates")


def _validate(wrapped, gamma, steps, intervention):
    if not isinstance(wrapped, PhaseStateIntegration):
        raise TypeError("SW0140 requires a strict-loaded SW0130 source wrapper")
    core = wrapped.core
    if wrapped.arm != "phase":
        raise ValueError("SW0140 source must use the registered phase integration arm")
    if (getattr(core, "gamma_drive_mode", None) != "static"
            or getattr(core, "spike_per_component", False) is not True
            or int(getattr(core, "osc_dim", -1)) != COMPONENTS
            or core.kuramoto.spike_pulse_gain is not None
            or (core.graph_generator is not None and core.graph_generator.uses_feedback)):
        raise ValueError("SW0140 supports only frozen static four-component no-pulse source dynamics")
    if getattr(core, "gate_mode", None) != "raw":
        raise ValueError("SW0140 intervention semantics are registered only for raw sinusoidal gating")
    if intervention not in CONDITIONS:
        raise ValueError(f"unregistered SW0140 condition: {intervention}")
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (core.T, core.in_dim):
        raise ValueError(f"gamma must have static source shape [B,{core.T},{core.in_dim}]")
    if steps < 1 or gamma.device != next(core.parameters()).device:
        raise ValueError("invalid SW0140 rollout horizon or gamma device")


@torch.no_grad()
def causal_rollout(wrapped, gamma, *, intervention="none", total_steps=1024,
                   capture_internal=False):
    """Run a complete reset-from-zero source trajectory under one registered intervention.

    The NONE path is an independent implementation of the frozen late rollout and
    must remain bitwise equal on all six trace fields. The input wrapper and its
    checkpoint are never edited; K is changed only transiently for the isolated
    interaction-off condition and restored in a ``finally`` block.
    """
    core = wrapped.core
    gamma = gamma.to(core.device)
    steps = int(total_steps)
    _validate(wrapped, gamma, steps, intervention)
    batch, fold = gamma.shape[0], COMPONENTS
    feedback = core.graph_generator is not None and core.graph_generator.uses_feedback

    graph = (core.graph_generator(gamma) if core.graph_generator is not None
             else core.sc.to(gamma.device).unsqueeze(0).expand(batch, -1, -1))
    drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
    theta = core._init_theta(drive, batch)
    coupling = None
    if core.kuramoto_backend == "factorized" and not feedback:
        coupling = core.kuramoto.prepare_coupling(
            graph, batch_size=batch, num_units=core.in_dim, device=gamma.device)

    # Reset state for every intervention, never warm-start from another condition.
    core.dendric_layer.set_neuron_state(batch * fold)
    core.membrane_layer.set_neuron_state(batch * fold)
    dendrite, membrane = core.dendric_layer, core.membrane_layer
    theta_history = []
    membrane_frames, spike_frames, gate_frames = [], [], []
    carrier_frames, event_frames, natural_event_frames = [], [], []
    dense_frames, dendrite_frames = [], []
    previous_events = None
    zero_gate_holds = torch.zeros((), device=gamma.device, dtype=torch.long)
    event_transitions = torch.zeros((), device=gamma.device, dtype=torch.long)
    zero_gate_count = torch.zeros((), device=gamma.device, dtype=torch.long)
    original_k = core.kuramoto.K
    if intervention == "kuramoto_interaction_off":
        core.kuramoto.K = 0.0
    try:
        for t in range(steps):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(
                theta_history, t, core.phase_delay_steps, gate_mode=core.gate_mode)
            if intervention == "delayed_gate_constant_half":
                gate = torch.full_like(gate, 0.5)
                carrier = torch.sin(theta) * gate.unsqueeze(-1)
            folded_carrier = carrier.permute(0, 2, 1).reshape(batch * fold, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(fold, dim=0)
            integration_gate = folded_gate

            previous_h = dendrite.h
            previous_mem = membrane.mem
            previous_spike = membrane.spike
            dense = wrapped._dense_dendrite(dendrite, folded_carrier, previous_spike)
            beta_d = torch.sigmoid(dendrite.tau_n).unsqueeze(0)
            component_ids = torch.arange(fold, device=gamma.device).repeat(batch)
            a_d = wrapped.a_d[component_ids].view(batch * fold, 1, 1)
            if intervention == "dendrite_history_off":
                # Remove only the explicit previous_h term, retaining the
                # native current-input gain and its integration modulation.
                j_d = (1.0 - beta_d) * dense
                current_h = (1.0 + a_d * (integration_gate.unsqueeze(-1) - 1.0)) * j_d
            else:
                native_h = beta_d * previous_h + (1.0 - beta_d) * dense
                current_h = native_h + a_d * (integration_gate.unsqueeze(-1) - 1.0) * (
                    native_h - previous_h)
            dendrite.h = current_h
            h_wave = current_h.sum(dim=2)

            alpha = torch.sigmoid(membrane.tau_m).unsqueeze(0)
            b = wrapped.b[component_ids].view(batch * fold, 1)
            threshold = BASE_THRESHOLD * torch.exp(b)
            a_m = wrapped.a_m[component_ids].view(batch * fold, 1)
            if intervention == "membrane_voltage_history_off":
                j_m = (1.0 - alpha) * R_m * h_wave - threshold * previous_spike
                candidate_mem = (1.0 + a_m * (integration_gate - 1.0)) * j_m
            else:
                reset = (0.0 if intervention == "spike_reset_off"
                         else threshold * previous_spike)
                native_mem = previous_mem * alpha + (1.0 - alpha) * R_m * h_wave - reset
                candidate_mem = native_mem + a_m * (integration_gate - 1.0) * (
                    native_mem - previous_mem)
            current_mem = torch.where(folded_gate == 0, previous_mem, candidate_mem)
            natural_events = act_fun_adp(current_mem - threshold)
            if intervention == "binary_events_forced_on":
                events = torch.ones_like(current_mem)
            else:
                events = natural_events
            emitted = events * folded_gate
            membrane.mem, membrane.spike = current_mem, emitted

            membrane_frames.append(current_mem)
            spike_frames.append(emitted)
            gate_frames.append(folded_gate)
            carrier_frames.append(folded_carrier.squeeze(-1))
            event_frames.append(events)
            natural_event_frames.append(natural_events)
            if capture_internal:
                dense_frames.append(dense)
                dendrite_frames.append(current_h)
            zero_gate_count += (folded_gate == 0).sum()
            zero_gate_holds += ((folded_gate == 0) & (current_mem == previous_mem)).sum()
            if previous_events is not None:
                event_transitions += (events != previous_events).sum()
            previous_events = events
    finally:
        core.kuramoto.K = original_k

    # Exactly preserve late_rollout's native stack/permute/reduction order.
    component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_carrier = torch.stack(carrier_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_events = torch.stack(event_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_natural_events = torch.stack(natural_event_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    trace = {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "component_gates": component_gates,
        "membrane": component_membrane.mean(dim=1),
        "spikes": component_spikes.mean(dim=1),
        "theta": torch.stack(theta_history, dim=1),
        "component_carrier": component_carrier,
        "component_events": component_events,
        "component_natural_threshold_events": component_natural_events,
        "diagnostics": {"zero_gate_positions": int(zero_gate_count.item()),
                        "zero_gate_hold_positions": int(zero_gate_holds.item()),
                        "event_transitions": int(event_transitions.item()),
                        "event_occupancy": float(component_events.mean().item()),
                        "natural_threshold_event_occupancy": float(component_natural_events.mean().item()),
                        "emitted_spike_occupancy": float(component_spikes.ne(0).float().mean().item())},
    }
    if capture_internal:
        trace["component_dense"] = torch.stack(dense_frames, dim=0).permute(1, 2, 3, 0).reshape(
            batch, fold, core.in_dim, dense_frames[0].shape[-1], steps)
        trace["component_dendrite"] = torch.stack(dendrite_frames, dim=0).permute(1, 2, 3, 0).reshape(
            batch, fold, core.in_dim, dendrite_frames[0].shape[-1], steps)
    core.last_component_out = component_membrane
    core.last_component_spikes = component_spikes
    core.last_gate_history = component_gates
    if not torch.equal(component_spikes, component_events * component_gates):
        raise AssertionError("SW0140 emitted spikes must be the exact selected events multiplied by gates")
    return trace


def assert_none_parity(wrapped, gamma, *, total_steps):
    expected = late_rollout(wrapped, gamma, total_steps=total_steps, live_tail_steps=0)
    got = causal_rollout(wrapped, gamma, intervention="none", total_steps=total_steps)
    for name in TRACE_FIELDS:
        if not torch.equal(expected[name], got[name]):
            raise AssertionError(f"SW0140 NONE differs from frozen late_rollout in {name}")
    return got
