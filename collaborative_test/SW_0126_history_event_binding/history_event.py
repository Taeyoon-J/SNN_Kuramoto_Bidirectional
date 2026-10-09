"""Opt-in history-centered event membrane adapter for SW0126.

Strict-load the registered source core first, then call
``attach_history_event_membrane``. The untouched production membrane remains
the default everywhere else.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from snn_kuramoto_bidirectional.membrane_layer import (
    MembraneLayer,
    R_m,
    act_fun_adp,
)


COMPONENTS = 4
VTH = 0.06
BETA_INIT = 0.95


class HistoryBoundMembrane(MembraneLayer):
    """Membrane update with a per-component history-centered event state.

    ``event_history`` is per folded batch row and unit. Folded rows must be
    ordered as ``(batch, component)`` with four components, matching S2NetCore.
    Exact-zero gates preserve both the membrane and event history. The forward
    event is hard-thresholded; ``act_fun_adp`` supplies the registered SNN
    surrogate derivative.
    """

    def __init__(self, native: MembraneLayer, beta_init: float = BETA_INIT,
                 capture_gate_trace: bool = False):
        if not isinstance(native, MembraneLayer):
            raise TypeError("native must be the registered MembraneLayer")
        if abs(float(native.vth) - VTH) > 1e-12:
            raise ValueError("SW0126 requires the registered membrane threshold 0.06")
        if not 0.0 < beta_init < 1.0:
            raise ValueError("beta_init must lie strictly between zero and one")

        nn.Module.__init__(self)
        self.output_dim = native.output_dim
        self.device = native.device
        self.vth = native.vth
        self.dt = native.dt
        # Preserve the source parameter object/name; source core is loaded
        # before replacement and the adapter adds only beta_logits.
        self.tau_m = native.tau_m
        self.mem = native.mem
        self.spike = native.spike
        self.v_th = getattr(native, "v_th", None)
        beta_logit = math.log(beta_init / (1.0 - beta_init))
        self.beta_logits = nn.Parameter(torch.full((COMPONENTS,), beta_logit,
                                                   dtype=self.tau_m.dtype,
                                                   device=self.tau_m.device))
        self.event_history = None
        self.last_event = None
        self.last_gate = None
        self.capture_gate_trace = bool(capture_gate_trace)
        self.gate_trace_history = None
        self.event_trace_history = None

    def set_neuron_state(self, batch_size):
        super().set_neuron_state(batch_size)
        self.event_history = torch.zeros_like(self.mem)
        self.last_event = torch.zeros_like(self.mem)
        self.last_gate = torch.zeros_like(self.mem)
        self.gate_trace_history = [] if self.capture_gate_trace else None
        self.event_trace_history = [] if self.capture_gate_trace else None

    def forward(self, h_wave_t, g_wave_t):
        if self.mem is None or self.spike is None or self.v_th is None:
            raise RuntimeError("set_neuron_state must be called before forward")
        rows = self.mem.shape[0]
        if rows % COMPONENTS:
            raise ValueError("folded membrane batch must contain four rows per image")
        if tuple(h_wave_t.shape) != tuple(self.mem.shape):
            raise ValueError("dendritic input shape must match membrane state")
        gate = g_wave_t.expand(rows, -1)
        if tuple(gate.shape) != tuple(self.mem.shape):
            raise ValueError("gate shape must match folded membrane state")

        previous_mem = self.mem
        alpha = torch.sigmoid(self.tau_m)
        candidate_mem = (previous_mem * alpha + (1.0 - alpha) * R_m * h_wave_t
                         - self.v_th * self.spike)
        open_gate = gate != 0
        self.mem = torch.where(open_gate, candidate_mem, previous_mem)

        u = self.mem - VTH
        event = act_fun_adp(u - self.event_history)
        component = torch.arange(rows, device=self.mem.device) % COMPONENTS
        beta = torch.sigmoid(self.beta_logits[component]).unsqueeze(-1)
        candidate_history = beta * self.event_history + (1.0 - beta) * u
        self.event_history = torch.where(open_gate, candidate_history, self.event_history)
        self.last_event = event
        self.last_gate = gate
        if self.gate_trace_history is not None:
            self.gate_trace_history.append(gate)
            self.event_trace_history.append(event)
        self.spike = gate * event
        return self.mem, self.spike

    def _folded_trace(self, trace, label: str, batch_size: int):
        if trace is None:
            raise RuntimeError(f"{label} trace capture is disabled")
        if not trace:
            raise RuntimeError(f"no {label} trace has been recorded")
        folded = torch.stack(trace, dim=-1)
        if folded.shape[0] != batch_size * COMPONENTS:
            raise ValueError(f"captured {label} does not match the requested folded batch")
        return folded.reshape(batch_size, COMPONENTS, self.output_dim, -1)

    def component_gate_trace(self, batch_size: int):
        return self._folded_trace(self.gate_trace_history, "gate", batch_size)

    def component_event_trace(self, batch_size: int):
        return self._folded_trace(self.event_trace_history, "event", batch_size)

    def detach_rollout_boundary(self):
        """Detach recurrent states at a truncated-BPTT boundary, preserving values."""
        for name in ("mem", "spike", "event_history", "last_event", "last_gate"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.detach())
        if self.gate_trace_history is not None:
            self.gate_trace_history = [value.detach() for value in self.gate_trace_history]
            self.event_trace_history = [value.detach() for value in self.event_trace_history]


def attach_history_event_membrane(core, beta_init: float = BETA_INIT,
                                  capture_gate_trace: bool = False):
    """Replace only a source-loaded core's membrane layer with the opt-in adapter."""
    if int(getattr(core, "osc_dim", -1)) != COMPONENTS:
        raise ValueError("SW0126 requires a four-component source core")
    current = core.membrane_layer
    if isinstance(current, HistoryBoundMembrane):
        raise ValueError("history event membrane is already attached")
    if not isinstance(current, MembraneLayer):
        raise TypeError("core membrane layer is not the registered production class")
    adapted = HistoryBoundMembrane(current, beta_init=beta_init,
                                   capture_gate_trace=capture_gate_trace)
    core.membrane_layer = adapted
    return adapted


def strict_load_source_then_attach(core, source_state_dict, beta_init: float = BETA_INIT,
                                   capture_gate_trace: bool = False):
    """Strict-load an unmodified source core before installing the adapter."""
    core.load_state_dict(source_state_dict, strict=True)
    return attach_history_event_membrane(core, beta_init=beta_init,
                                         capture_gate_trace=capture_gate_trace)


def head_trace_for_arm(event_trace, gate_trace, arm):
    """Select the only arm-specific signal consumed by the imagewise binder.

    Recurrence remains event-driven in both arms. ``gate_only`` is a diagnostic
    head input that omits event timing and consumes the continuous gate trace.
    """
    if arm == "history_event":
        return event_trace
    if arm == "gate_only":
        return gate_trace
    raise ValueError("arm must be 'history_event' or 'gate_only'")
