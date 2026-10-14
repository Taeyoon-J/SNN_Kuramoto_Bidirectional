"""Opt-in signed phase-interaction kernels for SW0156; does not patch native code."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

NATIVE_GRAPH_FLOOR = 1e-6


def prepare_signed_coupling(layer, adjacency: torch.Tensor, symmetric_u: torch.Tensor,
                            beta: float, *, base_scale: float = 1.0):
    """Return native A0 pulse graph with signed A_beta lag interactions.

    ``symmetric_u`` is U after the registered directed scatter and one
    symmetrization.
    Native lag/cost/alpha are always prepared from the original positive A0.
    Only the factorized interaction multipliers are signed; A0 remains the
    returned pulse-coupling graph. At beta=0 the original native tuple is
    returned unchanged, preserving its exact numerical path.
    """
    if (not isinstance(beta, (int, float)) or not torch.isfinite(torch.tensor(float(beta)))
            or not isinstance(base_scale, (int, float))
            or not torch.isfinite(torch.tensor(float(base_scale))) or base_scale < 0):
        raise ValueError("beta and nonnegative base_scale must be finite")
    native = layer.prepare_coupling(adjacency, batch_size=adjacency.shape[0],
                                    num_units=adjacency.shape[-1], device=adjacency.device)
    return transform_native_tuple(native, adjacency, symmetric_u, beta, base_scale=base_scale)


def transform_native_tuple(native, adjacency: torch.Tensor, symmetric_u: torch.Tensor,
                           beta: float, *, base_scale: float = 1.0):
    """Apply signed interaction weights to an already-produced native tuple."""
    if beta == 0 and base_scale == 1:
        return native
    if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must be [B,N,N]")
    if symmetric_u.shape != adjacency.shape or symmetric_u.device != adjacency.device:
        raise ValueError("symmetric_u must match adjacency shape and device")
    if not torch.isfinite(symmetric_u).all() or not torch.isfinite(adjacency).all():
        raise ValueError("adjacency and symmetric_u must be finite")
    if not torch.equal(symmetric_u, symmetric_u.transpose(-1, -2)):
        raise ValueError("symmetric_u must be exactly symmetric")
    a0, lag_cos0, lag_sin0 = native
    # This divides by the native A0 (which includes its 1e-6 floor) to recover
    # exactly the native cos(alpha), sin(alpha) factors.
    cos_alpha = lag_cos0 / a0
    sin_alpha = lag_sin0 / a0
    a_beta = (float(base_scale) * a0
              + (1.0 - float(base_scale)) * NATIVE_GRAPH_FLOOR
              - float(beta) * symmetric_u)
    return a0, a_beta * cos_alpha, a_beta * sin_alpha


def topk_signed_perturbation(logits: torch.Tensor, top_k: int, gain: torch.Tensor | float) -> torch.Tensor:
    """Build symmetric U from source top-k logits and learned gain.

    The graph generator's support is ``logits.topk(min(top_k,N))`` (including
    whatever diagonal entries that native rule selects). U scatters gain/K on
    that directed support, then symmetrizes. It intentionally does not use the
    native softmax weights; this is the registered fixed-magnitude perturbation.
    """
    if logits.ndim != 3 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("logits must be [B,N,N]")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    k = min(top_k, logits.shape[-1])
    gain_t = torch.as_tensor(gain, dtype=logits.dtype, device=logits.device)
    if not torch.isfinite(gain_t).all() or (gain_t <= 0).any():
        raise ValueError("learned gain must be finite and positive")
    indices = logits.topk(k, dim=-1).indices
    values = (gain_t / k).expand_as(indices).to(dtype=logits.dtype)
    directed = torch.zeros_like(logits).scatter_(-1, indices, values)
    return 0.5 * (directed + directed.transpose(-1, -2))


def native_logits_and_signed_u(graph_module, gamma: torch.Tensor, native_adjacency: torch.Tensor,
                               alignment: torch.Tensor | None = None):
    """Reproduce the frozen graph-generator logits and derive signed U.

    The returned ``native_adjacency`` must be the output from this same
    graph-module call/setting. Exact equality is required so a local T geodesic
    override, alignment setting, learned gain, and source ordering cannot drift.
    """
    if gamma.ndim != 3:
        raise ValueError("gamma must be rank 3")
    g = gamma
    if g.size(1) == graph_module.projection.in_features and g.size(2) != graph_module.projection.in_features:
        g = g.transpose(1, 2)
    z = F.normalize(graph_module.projection(g), dim=-1)
    logits = torch.bmm(z, z.transpose(1, 2)) / graph_module.log_temperature.exp().clamp_min(1e-3)
    if graph_module.spatial_rate is not None:
        distance = graph_module.grid_distance.unsqueeze(0)
        if graph_module.geodesic_steps > 0:
            distance = graph_module._geodesic_distance(z, distance)
        logits = logits - F.softplus(graph_module.spatial_rate) * distance
    if graph_module.feedback_strength is not None and alignment is not None:
        logits = logits + graph_module.feedback_strength * alignment
    k = min(graph_module.top_k, logits.size(-1))
    values, indices = logits.topk(k, dim=-1)
    weights = values.softmax(dim=-1) * graph_module.log_coupling_gain.exp()
    adjacency = torch.zeros_like(logits).scatter_(-1, indices, weights)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    if adjacency.shape != native_adjacency.shape or not torch.equal(adjacency, native_adjacency):
        raise ValueError("recomputed frozen graph differs from native graph output")
    u = topk_signed_perturbation(logits, graph_module.top_k,
                                 graph_module.log_coupling_gain.exp())
    return logits, u


def install_instance_signed_interaction(graph_module, kuramoto_layer, *, beta: float,
                                        base_scale: float = 1.0):
    """Install the opt-in factorized interaction on these two instances only.

    The source graph forward still runs first and produces A0. Its logits are
    recomputed and checked against that exact output before the signed U is
    cached for Kuramoto's one-time factorized coupling preparation. The native
    graph generator and shared classes/files are never modified. This bounded
    path currently requires the registered non-feedback factorized backend.
    Call the returned restore function in ``finally``.
    """
    if getattr(kuramoto_layer, "kuramoto_backend", None) != "factorized":
        raise ValueError("SW0156 opt-in currently requires the source factorized backend")
    if getattr(graph_module, "uses_feedback", False):
        raise ValueError("SW0156 registered source graph must not use dynamic feedback")
    original_forward = graph_module.forward
    original_prepare = kuramoto_layer.prepare_coupling
    cache = {"u": None}

    def forward_with_u(this, gamma, alignment=None):
        adjacency = original_forward(gamma, alignment=alignment)
        _logits, u = native_logits_and_signed_u(this, gamma, adjacency, alignment=alignment)
        cache["u"] = u
        return adjacency

    def prepare_with_u(this, adjacency, batch_size=None, num_units=None, device=None):
        u = cache["u"]
        if u is None:
            raise RuntimeError("signed U is unavailable; source graph must run first")
        native = original_prepare(adjacency, batch_size=batch_size, num_units=num_units,
                                  device=device)
        return transform_native_tuple(native, adjacency, u, beta, base_scale=base_scale)

    graph_module.forward = types.MethodType(forward_with_u, graph_module)
    kuramoto_layer.prepare_coupling = types.MethodType(prepare_with_u, kuramoto_layer)

    def restore():
        graph_module.forward = original_forward
        kuramoto_layer.prepare_coupling = original_prepare
        cache["u"] = None

    return restore
