"""Opt-in SW0169 normalized slow-membrane hard-reset rollout.

This module deliberately leaves the registered native model untouched.  It
reuses SW0130's four-component dendrite and SW0134's native forward equations,
changing only the membrane recurrence described in DESIGN_KO.md.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating

COMPONENTS = 4
BASE_THRESHOLD = 0.06
TARGET_EFOLD = 16.0
TARGET_THRESHOLD = 0.5


class MembraneStep(NamedTuple):
    voltage: Tensor
    raw_event: Tensor
    history_event: Tensor
    emitted_spike: Tensor
    current: Tensor


def normalized_membrane_step(*, voltage: Tensor, event_history: Tensor,
                             previous_membrane: Tensor, previous_event: Tensor,
                             current: Tensor, gate: Tensor, retention: Tensor,
                             threshold: Tensor, conductance_scale: Tensor,
                             detach_history: bool = False,
                             validate_inputs: bool = True,
                             event_fn=act_fun_adp) -> MembraneStep:
    """One exact SW0169 update; inputs use folded ``[B*4,N]`` geometry.

    ``voltage`` and ``event_history`` are the persistent V/E states.  The
    separate previous operands make the detach-control scope explicit and
    testable.  No detach is applied to current, threshold, gate, or parameters.
    """
    tensors = (voltage, event_history, previous_membrane, previous_event,
               current, gate, retention, threshold, conductance_scale)
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("membrane step inputs must be tensors")
    if voltage.shape != event_history.shape or voltage.shape != current.shape:
        raise ValueError("V, E-history, and current must have matching shapes")
    if previous_membrane.shape != voltage.shape or previous_event.shape != voltage.shape:
        raise ValueError("previous V/E carry operands must match the folded state")
    try:
        g = torch.broadcast_to(gate, voltage.shape)
        lam = torch.broadcast_to(retention, voltage.shape)
        theta = torch.broadcast_to(threshold, voltage.shape)
        scale = torch.broadcast_to(conductance_scale, voltage.shape)
    except RuntimeError as exc:
        raise ValueError("gate, retention, threshold, and conductance must broadcast to V") from exc
    if validate_inputs:
        if any(not torch.isfinite(x).all() for x in (voltage, event_history, current, g, lam, theta, scale)):
            raise FloatingPointError("nonfinite membrane-step input")
        if (g < 0).any() or (g > 1).any() or (lam < 0).any() or (lam > 1).any():
            raise ValueError("gate and retention must lie in [0,1]")
        if (theta <= 0).any():
            raise ValueError("threshold must be strictly positive")

    vcarry = previous_membrane.detach() if detach_history else previous_membrane
    ecarry = previous_event.detach() if detach_history else previous_event
    proposal = lam * (1.0 - ecarry) * vcarry + (1.0 - lam) * scale * current
    next_v = torch.where(g == 0, vcarry, proposal)
    raw = event_fn(next_v - theta)
    emitted = raw * g
    next_e = torch.where(g == 0, ecarry, raw)
    return MembraneStep(next_v, raw, next_e, emitted, current)


def source_calibration(h_wave: Tensor, gate: Tensor, *, settle: int = 512):
    """Return CPU-float64 active-gate RMS current calibration for four rows."""
    if h_wave.ndim != 4 or h_wave.shape[1] != COMPONENTS:
        raise ValueError("h_wave must have shape [B,4,N,T]")
    if gate.shape != h_wave.shape:
        raise ValueError("gate and h_wave shapes must match")
    if not 0 <= settle < h_wave.shape[-1]:
        raise ValueError("settle boundary is outside the trace")
    active = (gate[..., settle:].detach().to(device="cpu") > 0)
    current = (float(R_m) * h_wave[..., settle:].detach().to(
        device="cpu", dtype=torch.float64))
    rows, counts, rms = [], [], []
    for component in range(COMPONENTS):
        mask = active[:, component]
        count = int(mask.sum().item())
        if count == 0:
            raise ValueError(f"component {component} has no active calibration samples")
        values = current[:, component][mask]
        mean_square = values.square().mean()
        value = float(torch.sqrt(mean_square).item())
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"component {component} RMS calibration is not finite and positive")
        rows.append(value)
        counts.append(count)
        rms.append(value)
    calibration = torch.tensor(rows, dtype=torch.float32)
    return {"r": calibration, "active_counts": counts, "r_float64": rms,
            "settle": int(settle), "source": "R_m*h_wave, active gate only"}


def initialize_slow_membrane_(wrapped) -> dict:
    """Apply the preregistered tau_m recentering and normalized threshold."""
    with torch.no_grad():
        tau = wrapped.core.membrane_layer.tau_m
        if not torch.isfinite(tau).all():
            raise FloatingPointError("source tau_m contains nonfinite values")
        source = tau.detach().clone()
        target = math.log(math.exp(-1.0 / TARGET_EFOLD) /
                          (1.0 - math.exp(-1.0 / TARGET_EFOLD)))
        tau.copy_(source - source.median() + target)
        b = wrapped.b
        b.fill_(math.log(TARGET_THRESHOLD / BASE_THRESHOLD))
        actual_lambda = torch.sigmoid(tau.median())
        actual_theta = BASE_THRESHOLD * torch.exp(b)
    if not torch.allclose(actual_lambda, torch.tensor(math.exp(-1.0 / TARGET_EFOLD),
                                                      device=tau.device), rtol=2e-6, atol=2e-7):
        raise AssertionError("median tau_m does not produce the registered e-fold")
    if not torch.allclose(actual_theta, torch.full_like(actual_theta, TARGET_THRESHOLD),
                          rtol=2e-6, atol=2e-7):
        raise AssertionError("normalized membrane threshold is not 0.5")
    return {"tau_m_sha256": _tensor_sha(tau), "b_sha256": _tensor_sha(b),
            "median_retention": float(actual_lambda),
            "threshold": actual_theta.detach().cpu().tolist()}


def _tensor_sha(value: Tensor) -> str:
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rollout(wrapped, gamma: Tensor, calibration_r: Tensor, *, total_steps=None,
            live_tail_steps: int = 64, detach_history: bool = False,
            source_mode: bool = False):
    """Run native static graph dynamics with an opt-in SW0169 membrane.

    ``source_mode`` is the exact SW0134 membrane equation, used only to capture
    calibration ``h_wave`` and parity evidence.  Candidate/control use the
    normalized recurrence.  All ordinary source six-trace fields are returned.
    """
    # Local imports avoid changing the production modules or their import state.
    from collaborative_test.SW_0130_phase_state_integration.model import (
        COMPONENTS as PHASE_COMPONENTS, PhaseStateIntegration,
    )
    from collaborative_test.SW_0134_native_spike_binding.rollout import _validate

    core = wrapped.core
    gamma = gamma.to(core.device)
    steps = int(total_steps if total_steps is not None else core.num_time_steps)
    tail = int(live_tail_steps)
    _validate(wrapped, gamma, steps, tail)
    if calibration_r.shape != (COMPONENTS,) or not torch.isfinite(calibration_r).all():
        raise ValueError("calibration r must be four finite values")
    if (calibration_r <= 0).any():
        raise ValueError("calibration r must be strictly positive")
    if not torch.isfinite(gamma).all():
        raise FloatingPointError("gamma contains nonfinite values")
    for name, parameter in (("tau_m", core.membrane_layer.tau_m),
                            ("b", wrapped.b), ("a_m", wrapped.a_m),
                            ("a_d", wrapped.a_d), ("tau_n", core.dendric_layer.tau_n)):
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"{name} contains nonfinite values")
    if not isinstance(wrapped, PhaseStateIntegration) or wrapped.arm != "phase":
        raise TypeError("SW0169 requires the registered phase integration wrapper")

    boundary = steps - tail
    batch, fold = gamma.shape[0], COMPONENTS
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
    raw_frames, history_frames, current_frames, h_frames = [], [], [], []
    core.dendric_layer.set_neuron_state(batch * fold)
    core.membrane_layer.set_neuron_state(batch * fold)
    dendrite, membrane = core.dendric_layer, core.membrane_layer
    event_history = torch.zeros_like(membrane.mem)
    r = calibration_r.to(device=gamma.device, dtype=gamma.dtype)
    component_ids = torch.arange(fold, device=gamma.device).repeat(batch)

    for t in range(steps):
        if t == boundary and boundary > 0:
            theta_history = [value.detach() for value in theta_history]
            theta = theta.detach()
            dendrite.h = dendrite.h.detach()
            membrane.mem = membrane.mem.detach()
            membrane.spike = membrane.spike.detach()
            # E history is stored separately below and detached at the same boundary.
            event_history = event_history.detach()

        with torch.set_grad_enabled(t >= boundary):
            theta = core.kuramoto(theta, drive, A=graph, spike=None, coupling=coupling)
            theta_history.append(theta)
            carrier, gate = sinusoidal_gating(
                theta_history, t, core.phase_delay_steps, gate_mode=core.gate_mode)
            folded_carrier = carrier.permute(0, 2, 1).reshape(batch * fold, core.in_dim, 1)
            folded_gate = gate.repeat_interleave(fold, dim=0)
            integration_gate = folded_gate

            previous_h = dendrite.h
            previous_mem = membrane.mem
            previous_spike = membrane.spike
            dense = wrapped._dense_dendrite(dendrite, folded_carrier, previous_spike)
            beta_d = torch.sigmoid(dendrite.tau_n).unsqueeze(0)
            native_h = beta_d * previous_h + (1.0 - beta_d) * dense
            a_d = wrapped.a_d[component_ids].view(batch * fold, 1, 1)
            integrated_h = native_h + a_d * (integration_gate.unsqueeze(-1) - 1.0) * (
                native_h - previous_h)
            dendrite.h = integrated_h
            h_wave = integrated_h.sum(dim=2).squeeze(-1)
            h_frames.append(h_wave.reshape(batch, fold, core.in_dim))

            if source_mode:
                alpha = torch.sigmoid(membrane.tau_m).unsqueeze(0)
                threshold = BASE_THRESHOLD * torch.exp(
                    wrapped.b[component_ids].view(batch * fold, 1))
                native_mem = (previous_mem * alpha + (1.0 - alpha) * R_m * h_wave
                              - threshold * previous_spike)
                a_m = wrapped.a_m[component_ids].view(batch * fold, 1)
                integrated_mem = native_mem + a_m * (integration_gate - 1.0) * (
                    native_mem - previous_mem)
                integrated_mem = torch.where(folded_gate == 0, previous_mem, integrated_mem)
                raw_event = act_fun_adp(integrated_mem - threshold)
                emitted = raw_event * folded_gate
                event_history = raw_event
            else:
                row_r = r[component_ids].unsqueeze(-1)
                current = (float(R_m) * h_wave) / row_r
                alpha = torch.sigmoid(membrane.tau_m).unsqueeze(0)
                threshold = BASE_THRESHOLD * torch.exp(
                    wrapped.b[component_ids].view(batch * fold, 1))
                a_m = wrapped.a_m[component_ids].view(batch * fold, 1)
                conductance = 1.0 + a_m * (integration_gate - 1.0)
                step = normalized_membrane_step(
                    voltage=membrane.mem, event_history=event_history,
                    previous_membrane=previous_mem, previous_event=event_history,
                    current=current, gate=folded_gate, retention=alpha,
                    threshold=threshold, conductance_scale=conductance,
                    detach_history=detach_history, validate_inputs=False)
                integrated_mem = step.voltage
                raw_event, emitted, event_history = step.raw_event, step.emitted_spike, step.history_event
                current_frames.append(current.reshape(batch, fold, core.in_dim))

            membrane.mem, membrane.spike = integrated_mem, emitted
            membrane_frames.append(integrated_mem)
            spike_frames.append(emitted)
            gate_frames.append(folded_gate)
            raw_frames.append(raw_event)
            history_frames.append(event_history)

    def _stack_component(frames):
        return torch.stack(frames, dim=0).permute(1, 2, 0).reshape(batch, fold, core.in_dim, steps)

    component_membrane = _stack_component(membrane_frames)
    component_spikes = _stack_component(spike_frames)
    component_gates = torch.stack(gate_frames, dim=-1).reshape(batch, fold, core.in_dim, steps)
    component_raw = _stack_component(raw_frames)
    component_history = _stack_component(history_frames)
    component_current = torch.stack(current_frames, dim=-1) if current_frames else None
    component_h = torch.stack(h_frames, dim=-1)
    theta_trace = torch.stack(theta_history, dim=1)
    core.last_component_out = component_membrane
    core.last_component_spikes = component_spikes
    core.last_gate_history = component_gates
    return {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "component_gates": component_gates,
        "membrane": component_membrane.mean(dim=1),
        "spikes": component_spikes.mean(dim=1),
        "theta": theta_trace,
        "raw_events": component_raw,
        "event_history": component_history,
        "normalized_current": component_current,
        "h_wave": component_h,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": 0 < tail < steps,
    }
