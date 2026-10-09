"""Exact S2Net static-gamma rollout with optional truncated late-time credit.

The prefix is executed without gradients. Its full phase history and recurrent
states are retained as values and detached at the boundary. Static graph and
drive tensors are prepared once with gradients enabled, used under no-grad for
the prefix, and reused by the tail so trainable graph/drive parameters receive
tail gradients. This is truncated BPTT, not full-rollout 1024-step BPTT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SNN = ROOT / "snn_kuramoto_bidirectional"
if str(SNN) not in sys.path:
    sys.path.insert(0, str(SNN))

from sinusoidal_gating import sinusoidal_gating


def _validate(core, gamma, total_steps, tail_steps):
    if getattr(core, "gamma_drive_mode", None) != "static":
        raise ValueError("late rollout supports only static-gamma source cores")
    if getattr(core, "spike_per_component", False) is not True:
        raise ValueError("late rollout requires the registered four-component core")
    if int(getattr(core, "osc_dim", -1)) != 4:
        raise ValueError("late rollout requires exactly four oscillator components")
    if getattr(core, "graph_generator", None) is not None and core.graph_generator.uses_feedback:
        raise ValueError("feedback graph dynamics are outside the late-rollout contract")
    if core.kuramoto.spike_pulse_gain is not None:
        raise ValueError("spike-pulse coupling is outside the late-rollout contract")
    if gamma.ndim != 3 or tuple(gamma.shape[1:]) != (core.T, core.in_dim):
        raise ValueError(f"gamma must be static [B,{core.T},{core.in_dim}]")
    if total_steps < 1 or not 0 <= tail_steps <= total_steps:
        raise ValueError("invalid rollout or live-tail length")
    if gamma.device != next(core.parameters()).device:
        raise ValueError("gamma and source core must be on the same device")


def late_rollout(core, gamma, total_steps=None, live_tail_steps=64):
    """Return actual traces from a static-gamma rollout.

    ``live_tail_steps=0`` is a fully no-grad reference rollout. Otherwise the
    preceding frames are no-grad and only the final requested steps retain the
    recurrent graph. At the boundary, theta history values remain exact while
    hidden states detach; the graph, gamma-to-drive and factorized coupling
    prepared before burn-in are reused with gradients enabled for the tail.
    """
    gamma = gamma.to(core.device)
    steps = int(total_steps if total_steps is not None else core.num_time_steps)
    tail = int(live_tail_steps)
    _validate(core, gamma, steps, tail)
    boundary = steps - tail
    batch = gamma.shape[0]
    components = core.osc_dim
    feedback = (core.graph_generator is not None
                and core.graph_generator.uses_feedback)

    setup_context = torch.enable_grad() if tail > 0 else torch.no_grad()
    with setup_context:
        if core.graph_generator is not None:
            sc = core.graph_generator(gamma)
        else:
            sc = core.sc.to(gamma.device).unsqueeze(0).expand(batch, -1, -1)
        drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
        theta = core._init_theta(drive, batch)
        coupling = None
        if core.kuramoto_backend == "factorized" and not feedback:
            coupling = core.kuramoto.prepare_coupling(
                sc, batch_size=batch, num_units=core.in_dim,
                device=gamma.device,
            )

    theta_hist = []
    membrane_frames = []
    spike_frames = []
    folded = components if core.spike_per_component else 1
    core.dendric_layer.set_neuron_state(batch * folded)
    core.membrane_layer.set_neuron_state(batch * folded)
    pulse_enabled = core.kuramoto.spike_pulse_gain is not None
    for t in range(steps):
        if t == boundary and boundary > 0:
            # Keep values/forward path while cutting prefix history gradients.
            theta_hist = [item.detach() for item in theta_hist]
            theta = theta.detach()
            core.dendric_layer.h = core.dendric_layer.h.detach()
            core.membrane_layer.mem = core.membrane_layer.mem.detach()
            core.membrane_layer.spike = core.membrane_layer.spike.detach()
            # ``sc``, ``drive`` and factorized coupling were prepared once
            # before burn-in. Prefix steps consumed them under no-grad; these
            # same live tensors now carry tail credit into graph/drive weights.

        grad_enabled = t >= boundary
        with torch.set_grad_enabled(grad_enabled):
            theta = core.kuramoto(
                theta, drive, A=sc,
                spike=core.membrane_layer.spike if pulse_enabled else None,
                coupling=coupling,
            )
            theta_hist.append(theta)
            gamma_wave, gate = sinusoidal_gating(
                theta_hist, t, core.phase_delay_steps, gate_mode=core.gate_mode
            )
            if core.spike_per_component:
                folded_wave = gamma_wave.permute(0, 2, 1).reshape(
                    batch * folded, core.in_dim, 1
                )
                folded_gate = gate.repeat_interleave(folded, dim=0)
                h_wave = core.dendric_layer(folded_wave, core.membrane_layer.spike)
                membrane, spike = core.membrane_layer(h_wave, folded_gate)
            else:  # guarded above; retained for an explicit fail-closed contract
                h_wave = core.dendric_layer(gamma_wave, core.membrane_layer.spike)
                membrane, spike = core.membrane_layer(h_wave, gate)
            membrane_frames.append(membrane.reshape(batch, folded, core.in_dim))
            spike_frames.append(spike.reshape(batch, folded, core.in_dim))

    with torch.set_grad_enabled(tail > 0):
        # Match S2NetCore's production reduction layout exactly: stack as
        # [T,B*D,N], then fold and reshape to [B,D,N,T]. In particular, do not
        # stack directly into [B,D,N,T]; its non-contiguous reduction order
        # changes mean-over-components rounding on CUDA.
        component_membrane = torch.stack(membrane_frames, dim=0).permute(1, 2, 3, 0).reshape(
            batch, folded, core.in_dim, steps
        )
        component_spikes = torch.stack(spike_frames, dim=0).permute(1, 2, 3, 0).reshape(
            batch, folded, core.in_dim, steps
        )
        theta_trace = torch.stack(theta_hist, dim=1)
    return {
        "component_membrane": component_membrane,
        "component_spikes": component_spikes,
        "membrane": component_membrane.mean(dim=1),
        "spikes": component_spikes.mean(dim=1),
        "theta": theta_trace,
        "live_tail_steps": tail,
        "prefix_steps": boundary,
        "truncated_bptt": tail > 0,
    }
