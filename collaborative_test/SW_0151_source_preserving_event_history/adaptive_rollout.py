"""SW0151 isolated wrapper preserving the frozen SW0134 rollout equations."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0130_phase_state_integration.model import COMPONENTS, PhaseStateIntegration
from collaborative_test.SW_0151_source_preserving_event_history.event_history import AdaptiveEventHistory
from collaborative_test.SW_0134_native_spike_binding.rollout import _validate as _validate_native
from snn_kuramoto_bidirectional.membrane_layer import R_m
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


SW0134_ROLLOUT_SHA256 = "66c08404af91b50fb4cbfa63fe44650dd6a698ea669cd3d6a672dfc5c78503f3"
SOURCE_DEPENDENCY_SHA256 = {
    "collaborative_test/SW_0134_native_spike_binding/rollout.py": SW0134_ROLLOUT_SHA256,
    "collaborative_test/SW_0130_phase_state_integration/model.py":
        "2ba6bbe44dbbf2a5a444e724db71caafc1b9915570bfd7ab31343b86915cbd0b",
    "snn_kuramoto_bidirectional/membrane_layer.py":
        "680ad89277b60206f192123165d13eca916e5183dc71eaa443a8eeb5f4b8fac8",
    "snn_kuramoto_bidirectional/sinusoidal_gating.py":
        "e323a3b6472f51def23d63628fe25cc83088822ae73b5e0fcd1789884739f6c4",
}


def source_rollout_sha256() -> str:
    return source_dependency_hashes()["collaborative_test/SW_0134_native_spike_binding/rollout.py"]


def source_dependency_hashes() -> dict[str, str]:
    return {
        relpath: hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()
        for relpath in SOURCE_DEPENDENCY_SHA256
    }


def adaptive_rollout(wrapped, gamma, event_history: AdaptiveEventHistory, *,
                     total_steps=None, live_tail_steps=64, scale: float | torch.Tensor):
    """Run the SW0134 equations with only the event threshold path adapted.

    The source membrane integration and reset subtraction are kept intact.
    Adaptive emitted spikes are assigned to ``membrane.spike`` and therefore
    remain the next-step reset and dendrite inputs, exactly as in SW0134.
    Auxiliary event/history state is detached only at the same TBPTT boundary
    as native recurrent state.
    """
    current_dependencies = source_dependency_hashes()
    if current_dependencies != SOURCE_DEPENDENCY_SHA256:
        raise RuntimeError("frozen native rollout dependency fingerprint changed")
    core = wrapped.core
    steps = int(total_steps if total_steps is not None else core.num_time_steps)
    tail = int(live_tail_steps)
    _validate_native(wrapped, gamma, steps, tail)
    if not isinstance(event_history, AdaptiveEventHistory):
        raise TypeError("event_history must be an AdaptiveEventHistory module")
    if next(event_history.parameters()).device != gamma.device:
        raise ValueError("event-history module and gamma must be on the same device")
    if not gamma.is_floating_point() or gamma.dtype != torch.float32:
        raise ValueError("native SW0134 rollout gamma must be float32")
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
    membrane_frames, spike_frames, gate_frames = [], [], []
    event_frames, history_frames, threshold_frames = [], [], []
    core.dendric_layer.set_neuron_state(batch * fold)
    core.membrane_layer.set_neuron_state(batch * fold)
    dendrite, membrane = core.dendric_layer, core.membrane_layer
    previous_event = gamma.new_zeros(batch, fold, core.in_dim)
    history = gamma.new_zeros(batch, fold, core.in_dim)

    for t in range(steps):
        if t == boundary and boundary > 0:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            dendrite.h = dendrite.h.detach()
            membrane.mem = membrane.mem.detach()
            membrane.spike = membrane.spike.detach()
            previous_event = previous_event.detach()
            history = history.detach()

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
            native_threshold = 0.06 * torch.exp(wrapped.b[component_ids].view(batch * fold, 1))
            native_mem = (previous_mem * alpha + (1.0 - alpha) * R_m * h_wave
                          - native_threshold * previous_spike)
            a_m = wrapped.a_m[component_ids].view(batch * fold, 1)
            integrated_mem = native_mem + a_m * (integration_gate - 1.0) * (
                native_mem - previous_mem)
            integrated_mem = torch.where(folded_gate == 0, previous_mem, integrated_mem)

            native_threshold_3d = native_threshold.reshape(batch, fold, 1)
            event_step = event_history(
                integrated_mem.reshape(batch, fold, core.in_dim),
                folded_gate.reshape(batch, fold, core.in_dim),
                native_threshold_3d[0, :, 0], scale,
                previous_event, history, detach_state=False,
            )
            events = event_step["event"]
            emitted = event_step["spike"]
            previous_event, history = event_step["previous_event"], event_step["history"]
            membrane.mem = integrated_mem
            membrane.spike = emitted.reshape(batch * fold, core.in_dim)
            membrane_frames.append(integrated_mem)
            spike_frames.append(membrane.spike)
            gate_frames.append(folded_gate)
            event_frames.append(events.reshape(batch * fold, core.in_dim))
            history_frames.append(history.reshape(batch * fold, core.in_dim))
            threshold_frames.append(event_step["threshold"].reshape(batch * fold, core.in_dim))

    component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_events = torch.stack(event_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_history = torch.stack(history_frames, dim=-1).reshape(
        batch, fold, core.in_dim, steps)
    component_thresholds = torch.stack(threshold_frames, dim=-1).reshape(
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
        "component_events": component_events,
        "component_event_history": component_history,
        "component_event_thresholds": component_thresholds,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": 0 < tail < steps,
        "sw0134_rollout_sha256": SW0134_ROLLOUT_SHA256,
        "source_dependency_sha256": current_dependencies,
    }
