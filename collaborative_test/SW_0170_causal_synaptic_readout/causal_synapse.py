"""Causal synaptic-state transform for SW0170.

This module only transforms the already emitted component spikes ``S``. It
does not alter the source rollout or its membrane/reset/dendrite recurrence.
"""

from __future__ import annotations

import math

import torch


DECAY = math.exp(-1.0 / 16.0)
LIVE_START = 960
SETTLE_STEPS = 512


def causal_synapse(component_spikes: torch.Tensor, *, identity: bool = False,
                   live_start: int = LIVE_START, capture_step: int | None = None):
    """Return ``Z`` with fixed causal decay, preserving the final live tail.

    Args:
        component_spikes: Actual emitted ``S=Eraw*g``, shaped ``[B,D,N,T]``.
        identity: Return the exact input object when true (the point-event
            control ``Z=S``).
        live_start: Truncation boundary. The state is updated through every
            frame, detached at this boundary, then the suffix recurrence stays
            differentiable. Production uses the registered value 960; the
            argument permits short deterministic CPU recurrence tests.
        capture_step: Optional timestep in the live suffix at which to return
            the actual previous/input/next recurrence tensors for a local
            partial-gradient audit. With this option the return is ``(Z, refs)``.

    The recurrence is ``Z_t = DECAY*Z_(t-1) + (1-DECAY)*S_t``, initialized
    from zero. In particular, zero input decays the state; it does not hold it.
    """
    if not isinstance(component_spikes, torch.Tensor):
        raise TypeError("component_spikes must be a torch.Tensor")
    if component_spikes.ndim != 4:
        raise ValueError("component_spikes must have shape [B,D,N,T]")
    if not component_spikes.is_floating_point():
        raise TypeError("component_spikes must use a floating dtype")
    if component_spikes.shape[0] < 1 or component_spikes.shape[1] != 4 \
            or component_spikes.shape[2] < 1 or component_spikes.shape[3] < 1:
        raise ValueError("expected nonempty [B,4,N,T] component-spike geometry")
    if identity and capture_step is not None:
        raise ValueError("identity control has no causal-state step to capture")
    if identity:
        return component_spikes
    if not 0 <= int(live_start) <= component_spikes.shape[-1]:
        raise ValueError("live_start must lie within the trace")
    if capture_step is not None and not int(live_start) <= int(capture_step) < component_spikes.shape[-1]:
        raise ValueError("capture_step must be inside the differentiable suffix")

    steps = component_spikes.shape[-1]
    state = component_spikes.new_zeros(component_spikes.shape[:-1])
    frames: list[torch.Tensor] = []
    captured = None

    # Keep the entire prefix recurrence in no-grad mode. Its final value is
    # carried exactly into the suffix, with only the carried state detached.
    with torch.no_grad():
        for t in range(min(int(live_start), steps)):
            state = DECAY * state + (1.0 - DECAY) * component_spikes[..., t]
            frames.append(state)

    if live_start < steps:
        state = state.detach()
        for t in range(int(live_start), steps):
            previous = state
            current = component_spikes[..., t]
            state = DECAY * previous + (1.0 - DECAY) * current
            if capture_step is not None and t == int(capture_step):
                if state.requires_grad:
                    state.retain_grad()
                captured = {"step": t, "previous_state": previous,
                            "input_spike": current, "next_state": state}
            frames.append(state)
    result = torch.stack(frames, dim=-1)
    return (result, captured) if capture_step is not None else result


def production_affinity(component_trace: torch.Tensor, *, settle: int = SETTLE_STEPS):
    """Apply the existing four-component production affinity unchanged."""
    if component_trace.ndim != 4 or component_trace.shape[1] != 4:
        raise ValueError("component_trace must have shape [B,4,N,T]")
    if not 0 <= int(settle) < component_trace.shape[-1]:
        raise ValueError("settle must leave at least one time step")
    from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
    return spike_synchrony_affinity(
        component_trace.mean(dim=1), components=component_trace,
        settle=int(settle), affinity_mode="spike",
    )
