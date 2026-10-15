"""Explicitly separate spike-derived rho from provenance-bound context Q.

The existing ``structured_energy.pair_features`` remains strict: it verifies
that its Q reconstructs from its own spike input. This builder is for the
registered SW0167 cache path, where rho is computed from cached S per image in
B1 while logit(Q), Q row statistics, and A remain bound to the original B4
context cache. It deliberately makes no claim that those two Q computations
are bitwise or tolerance-equal.
"""
from __future__ import annotations

import numpy as np
import torch


def component_rho(component_spikes, *, eps=1e-8):
    """Reproduce the production per-component Pearson correlations exactly.

    ``component_spikes`` is [4,N,T]; the result is float32 [4,N,N] on the
    input device. The component loop and operation order match the registered
    strict wrapper.
    """
    spikes = torch.as_tensor(component_spikes)
    if (spikes.ndim != 3 or spikes.shape[0] != 4 or spikes.shape[1] < 1
            or spikes.shape[2] < 2 or not spikes.is_floating_point()):
        raise ValueError("component spikes must be floating [4,N,T], with T>=2")
    if not torch.isfinite(spikes).all():
        raise ValueError("component spikes must be finite")
    if not np.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    per_component = []
    for index in range(4):
        values = spikes[index].float()
        values = values - values.mean(dim=-1, keepdim=True)
        values = values / values.norm(dim=-1, keepdim=True).clamp_min(float(eps))
        per_component.append((values @ values.transpose(-1, -2)).clamp(-1.0, 1.0))
    rho = torch.stack(per_component, dim=0)
    if not torch.isfinite(rho).all():
        raise FloatingPointError("component rho contains nonfinite values")
    return rho


def features_from_rho_and_context_q(rho, context_q, source_a, gain):
    """Build the registered ten symmetric features from distinct provenances.

    ``rho`` is [4,N,N] from the per-image B1 spike trace. ``context_q`` and
    ``source_a`` are [N,N] float32 matrices from the validated original B4
    context cache. The builder validates each tensor independently; it does
    not assert the cross-mode production-Q reconstruction that belongs to
    ``pair_features``.
    """
    rho = torch.as_tensor(rho)
    q = torch.as_tensor(context_q, device=rho.device)
    adjacency = torch.as_tensor(source_a, device=rho.device)
    if (rho.ndim != 3 or rho.shape[0] != 4 or rho.shape[1] < 1
            or rho.shape[1] != rho.shape[2]):
        raise ValueError("rho must have shape [4,N,N]")
    n = rho.shape[1]
    if q.shape != (n, n) or adjacency.shape != (n, n):
        raise ValueError("rho/context Q/source A geometry mismatch")
    if rho.dtype != torch.float32 or q.dtype != torch.float32 or adjacency.dtype != torch.float32:
        raise ValueError("rho, context Q, and source A must retain float32 cache arithmetic")
    if not torch.isfinite(rho).all() or not torch.isfinite(q).all() or not torch.isfinite(adjacency).all():
        raise ValueError("rho/context Q/source A must be finite")
    if torch.any(rho < -1.0 - 1e-7) or torch.any(rho > 1.0 + 1e-7):
        raise ValueError("rho entries must lie in [-1,1]")
    if torch.any(q < -1e-7) or torch.any(q > 1.0 + 1e-7):
        raise ValueError("context Q entries must lie in [0,1]")
    if not torch.allclose(q, q.T, rtol=0, atol=2e-6):
        raise ValueError("context Q must be symmetric")
    if not torch.allclose(adjacency, adjacency.T, rtol=0, atol=2e-6):
        raise ValueError("source A must be symmetric")
    if not torch.allclose(rho, rho.transpose(1, 2), rtol=0, atol=2e-6):
        raise ValueError("component rho matrices must be symmetric")
    if not np.isfinite(float(gain)) or float(gain) <= 0:
        raise ValueError("gain must be finite and positive")

    qsym = (q + q.T) * 0.5
    qbar = qsym.clamp(1e-6, 1.0 - 1e-6)
    logit = torch.log(qbar / (1.0 - qbar))
    asym = (adjacency + adjacency.T) * 0.5
    if torch.any(asym < 0):
        raise ValueError("prepared source A must be nonnegative")
    loga = torch.log1p(1024.0 * asym / float(gain))

    eye = torch.eye(n, device=rho.device, dtype=torch.bool)
    qstats = q.masked_fill(eye, 0)
    denom = max(n - 1, 1)
    row_mean = qstats.sum(-1) / denom
    row_var = (qstats.square().sum(-1) / denom - row_mean.square()).clamp_min(0)
    row_std = row_var.sqrt()
    mean_avg = (row_mean[:, None] + row_mean[None, :]) * 0.5
    mean_diff = (row_mean[:, None] - row_mean[None, :]).abs()
    std_avg = (row_std[:, None] + row_std[None, :]) * 0.5
    std_diff = (row_std[:, None] - row_std[None, :]).abs()
    features = torch.cat((rho.permute(1, 2, 0), logit[..., None], loga[..., None],
                          mean_avg[..., None], mean_diff[..., None],
                          std_avg[..., None], std_diff[..., None]), dim=-1)
    if features.shape != (n, n, 10) or not torch.isfinite(features).all():
        raise FloatingPointError("split-provenance feature construction failed")
    return 0.5 * (features + features.transpose(0, 1))
