"""Opt-in membrane-conditioned amplitude credit for actual emitted spikes.

This module is a leaf wrapper: SW0130/SW0134 and the native membrane module
remain untouched. The returned emitted spike is assigned to ``membrane.spike``
inside the rollout, so both next-step dendritic input and membrane reset consume
the same amplitude-modulated value.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from collaborative_test.SW_0134_native_spike_binding import rollout as native_rollout
from collaborative_test.SW_0130_phase_state_integration.model import COMPONENTS
from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SW157_REPORT = ROOT / "collaborative_test/SW_0157_train_gate_credit/results_archive/completed_20261014/0_train_gate_credit_seed1.json"
SW157_REPORT_SHA256 = "83e968cade400d54dae56395c42f1bb54443231903cb5f2642fc4f34e5e56b0e"
SOURCE_T_MARGIN_RMS = (2.7085165787417393, 2.598093824488581,
                      2.623449164156643, 2.2200754439435535)
AMPLITUDE = 0.5
ETA_NAMES = tuple(f"spike_credit_eta_{i}" for i in range(COMPONENTS))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_margin_rms(report_path=SW157_REPORT):
    """Read only the pinned source97_T calibration from the completed SW0157 report."""
    path = Path(report_path)
    if sha256_file(path) != SW157_REPORT_SHA256:
        raise ValueError("registered SW0157 calibration report SHA changed")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("experiment") != "SW0157_train_gate_credit_diagnostic"
            or record.get("status") != "complete" or record.get("seed") != 1
            or record.get("optimizer_updates") != 0
            or record.get("validation_or_test_ground_truth_used") is not False):
        raise ValueError("SW0157 source report is not the registered completed diagnostic")
    source_t = record.get("models", {}).get("source97_T", {})
    if (source_t.get("graph_operator") != "SW0143_T inference"
            or source_t.get("model_state_unchanged") is not True
            or source_t.get("optimizer_updates") != 0):
        raise ValueError("SW0157 scale calibration is not from the frozen source-T model")
    batches = source_t.get("batches", ())
    if (len(batches) != 2 or [row.get("source_positions") for row in batches] !=
            [[0, 1, 2, 3], [2048, 2049, 2050, 2051]]):
        raise ValueError("SW0157 source-T scale rows differ from the registered TRAIN batches")
    by_batch = []
    for batch in batches:
        components = batch.get("event_dynamics", {}).get("per_component", ())
        if len(components) != COMPONENTS:
            raise ValueError("SW0157 source-T report has incomplete component margins")
        values = [float(item["margin_rms"]) for item in components]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("SW0157 component margin RMS must be finite and positive")
        by_batch.append(values)
    pooled = tuple(math.sqrt(sum(values[i] ** 2 for values in by_batch) / len(by_batch))
                   for i in range(COMPONENTS))
    if any(abs(a - b) > 1e-12 for a, b in zip(pooled, SOURCE_T_MARGIN_RMS)):
        raise ValueError("SW0157 pooled source-T scales do not match the registered constants")
    return pooled


def install_spike_credit(wrapped, margin_rms, *, eta_initial=None):
    """Register the four learnable eta values and fixed source-T margin scales."""
    if hasattr(wrapped, "sw0158_eta") or hasattr(wrapped, "sw0158_margin_rms"):
        raise ValueError("SW0158 membrane credit is already installed")
    scale = torch.as_tensor(margin_rms, dtype=torch.float32,
                            device=next(wrapped.parameters()).device)
    if (tuple(scale.shape) != (COMPONENTS,) or not torch.isfinite(scale).all()
            or bool((scale <= 0).any())):
        raise ValueError("SW0158 requires four finite positive registered source-T scales")
    eta = torch.zeros(COMPONENTS, dtype=scale.dtype, device=scale.device)
    if eta_initial is not None:
        initial = torch.as_tensor(eta_initial, dtype=scale.dtype, device=scale.device)
        if tuple(initial.shape) != (COMPONENTS,) or not torch.isfinite(initial).all():
            raise ValueError("eta_initial must contain four finite component values")
        eta.copy_(initial)
    wrapped.register_parameter("sw0158_eta", torch.nn.Parameter(eta))
    wrapped.register_buffer("sw0158_margin_rms", scale.detach().clone())
    return wrapped.sw0158_eta


def emit_spikes(events, gates, membrane, threshold, eta, margin_rms, component_ids,
                *, detached_control=False, validate_values=True):
    """Apply the registered bounded amplitude path to native E and gate tensors."""
    if not all(torch.is_tensor(value) for value in
               (events, gates, membrane, threshold, eta, margin_rms, component_ids)):
        raise TypeError("SW0158 emitter inputs must be tensors")
    if (events.shape != gates.shape or events.shape != membrane.shape
            or events.shape != threshold.shape or events.ndim != 3
            or tuple(eta.shape) != (COMPONENTS,) or tuple(margin_rms.shape) != (COMPONENTS,)
            or component_ids.shape != events.shape[:1] or component_ids.dtype != torch.long
            or component_ids.device != events.device):
        raise ValueError("SW0158 emitter expects [B*4,N,1] and component-indexed parameters")
    if validate_values and (not all(torch.isfinite(value).all() for value in
                (events, gates, membrane, threshold, eta, margin_rms))
            or bool((margin_rms <= 0).any())
            or bool(((component_ids < 0) | (component_ids >= COMPONENTS)).any())):
        raise ValueError("SW0158 emitter inputs are nonfinite or outside their registered range")
    selected_eta = eta[component_ids].view(-1, 1, 1)
    selected_r = margin_rms[component_ids].view(-1, 1, 1)
    margin_input = membrane.detach() if detached_control else membrane
    margin = margin_input - threshold
    amplitude = 1.0 + AMPLITUDE * torch.tanh(selected_eta) * torch.tanh(margin / selected_r)
    return (events * gates) * amplitude


def late_rollout_credit(wrapped, gamma, *, total_steps=1024, live_tail_steps=64,
                        detached_control=False, capture_state=False):
    """SW0134-equation rollout with only the emitted spike amplitude overridden."""
    if not hasattr(wrapped, "sw0158_eta") or not hasattr(wrapped, "sw0158_margin_rms"):
        raise ValueError("install SW0158 credit parameters before rollout")
    native_rollout._validate(wrapped, gamma, int(total_steps), int(live_tail_steps))
    core = wrapped.core
    gamma = gamma.to(core.device)
    steps, tail = int(total_steps), int(live_tail_steps)
    boundary = steps - tail
    batch, fold = gamma.shape[0], COMPONENTS
    feedback = core.graph_generator is not None and core.graph_generator.uses_feedback
    # Parameters/scales are static across the rollout; validate them once instead
    # of forcing a device synchronization at every time step.
    if (not torch.isfinite(wrapped.sw0158_eta).all()
            or not torch.isfinite(wrapped.sw0158_margin_rms).all()
            or bool((wrapped.sw0158_margin_rms <= 0).any())):
        raise ValueError("SW0158 parameters/scales are nonfinite or invalid")
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
    membrane_frames, spike_frames, gate_frames, event_frames = [], [], [], []
    auxiliary = {key: [] for key in ("native_membrane", "previous_membrane", "previous_spike",
        "dendrite_previous_spike", "h_wave", "alpha", "threshold")} if capture_state else None
    core.dendric_layer.set_neuron_state(batch * fold)
    core.membrane_layer.set_neuron_state(batch * fold)
    dendrite, membrane = core.dendric_layer, core.membrane_layer

    for step in range(steps):
        if step == boundary and boundary > 0:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            dendrite.h = dendrite.h.detach()
            membrane.mem = membrane.mem.detach()
            membrane.spike = membrane.spike.detach()
        with torch.set_grad_enabled(step >= boundary):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(
                theta_history, step, core.phase_delay_steps, gate_mode=core.gate_mode)
            folded_carrier = carrier.permute(0, 2, 1).reshape(batch * fold, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(fold, dim=0)
            integration_gate = (folded_gate if wrapped.arm == "phase"
                                else torch.full_like(folded_gate, 0.5))

            previous_h, previous_mem, previous_spike = dendrite.h, membrane.mem, membrane.spike
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
            emitted = emit_spikes(events.unsqueeze(-1), folded_gate.unsqueeze(-1),
                integrated_mem.unsqueeze(-1),
                threshold.expand_as(integrated_mem).unsqueeze(-1),
                wrapped.sw0158_eta, wrapped.sw0158_margin_rms,
                component_ids, detached_control=detached_control, validate_values=False
                ).squeeze(-1)
            membrane.mem, membrane.spike = integrated_mem, emitted
            membrane_frames.append(integrated_mem)
            spike_frames.append(emitted)
            gate_frames.append(folded_gate)
            event_frames.append(events)
            if capture_state:
                auxiliary["native_membrane"].append(native_mem)
                auxiliary["previous_membrane"].append(previous_mem)
                auxiliary["previous_spike"].append(previous_spike)
                auxiliary["dendrite_previous_spike"].append(previous_spike)
                auxiliary["h_wave"].append(h_wave)
                auxiliary["alpha"].append(alpha.expand_as(native_mem))
                auxiliary["threshold"].append(threshold.expand_as(native_mem))

    component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 0).reshape(
        batch, fold, core.in_dim, steps)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(batch, fold, core.in_dim, steps)
    theta_trace = torch.stack(theta_history, dim=1)
    result = {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "component_gates": component_gates,
        "component_events": torch.stack(event_frames, dim=0).permute(1, 2, 0).reshape(
            batch, fold, core.in_dim, steps),
        "membrane": component_membrane.mean(dim=1),
        "spikes": component_spikes.mean(dim=1),
        "theta": theta_trace,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": 0 < tail < steps,
    }
    core.last_component_out = component_membrane
    core.last_component_spikes = component_spikes
    core.last_gate_history = component_gates
    if capture_state:
        result["credit_state"] = {
            key: torch.stack(values, dim=-1) for key, values in auxiliary.items()}
    return result
