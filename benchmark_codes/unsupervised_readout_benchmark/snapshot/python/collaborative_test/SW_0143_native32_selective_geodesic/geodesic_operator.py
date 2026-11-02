"""Separately versioned native32 selective geodesic contrast operators."""
from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from collaborative_test.SW_0135_native32_spike_binding import resolution as native135

EFFECTIVE_CONTRAST = 8.0


def inverse_softplus(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("effective geodesic contrast must be finite and positive")
    return math.log(math.expm1(value))


def geodesic_distance_power2(z, euclidean, contrast, *, steps=3, radius=1.5,
                              temperature=.5, cap=16., chunk=32, log4=True):
    """Native135 chunked soft geodesic with squared cosine dissimilarity."""
    if z.ndim != 3 or euclidean.ndim not in (2, 3):
        raise ValueError("expected z[B,N,D] and euclidean[N,N] or [1|B,N,N]")
    batch, n, _ = z.shape
    if euclidean.ndim == 2:
        euclidean = euclidean.unsqueeze(0)
    if euclidean.shape[0] not in (1, batch) or tuple(euclidean.shape[1:]) != (n, n):
        raise ValueError("geodesic distance shape does not match embeddings")
    if (not isinstance(chunk, int) or chunk < 1 or steps < 0
            or temperature <= 0 or cap <= 0 or radius <= 0):
        raise ValueError("invalid geodesic relaxation settings")
    euclidean = euclidean.to(device=z.device, dtype=z.dtype)
    contrast_t = torch.as_tensor(contrast, device=z.device, dtype=z.dtype)
    similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1., 1.)
    dissimilarity = (1. - similarity).square()
    step = euclidean * (1. + F.softplus(contrast_t) * dissimilarity)
    distance = torch.where(euclidean <= radius, step, torch.full_like(step, cap))
    correction = temperature * torch.log(torch.tensor(4., device=z.device, dtype=z.dtype)) if log4 else 0.
    for _ in range(steps):
        rows = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            def relax_rows(value, lo=start, hi=stop, temp=temperature, corr=correction):
                left = value[:, lo:hi, :].unsqueeze(-1)
                right = value.unsqueeze(1)
                candidates = left + right
                relaxed = -temp * torch.logsumexp(-candidates / temp, dim=2) + corr
                return torch.minimum(value[:, lo:hi, :], relaxed)
            if torch.is_grad_enabled() and distance.requires_grad:
                rows.append(checkpoint(relax_rows, distance, use_reentrant=False))
            else:
                rows.append(relax_rows(distance))
        distance = torch.cat(rows, dim=1)
    return distance.clamp(max=cap)


def install_operator(graph, arm: str):
    """Override only the graph distance method; preserve all source parameters."""
    if arm not in {"R", "C", "T"}:
        raise ValueError("registered geodesic arm must be R, C, or T")
    if (getattr(graph, "spatial_rate", None) is None or graph.geodesic_contrast is None
            or int(graph.geodesic_steps) < 1):
        raise ValueError("SW0143 requires the source's spatial learned graph and active geodesic")
    fixed_raw = inverse_softplus(EFFECTIVE_CONTRAST)

    def geodesic(self, z, euclidean):
        settings = dict(steps=self.geodesic_steps, radius=self.geodesic_radius,
                        temperature=self.geodesic_temperature, cap=self.geodesic_cap,
                        chunk=native135.ROW_CHUNK, log4=True)
        if arm == "R":
            # Exact SW0135 production value/gradient path and source contrast.
            return native135.geodesic_distance_chunked(
                z, euclidean, self.geodesic_contrast, **settings)
        fixed = torch.as_tensor(fixed_raw, device=z.device, dtype=z.dtype)
        if arm == "C":
            # Same linear dissimilarity as R; only effective contrast is fixed at8.
            return native135.geodesic_distance_chunked(z, euclidean, fixed, **settings)
        return geodesic_distance_power2(z, euclidean, fixed, **settings)

    graph._geodesic_distance = types.MethodType(geodesic, graph)
    graph._sw0143_geodesic_arm = arm
    graph._sw0143_effective_contrast = (
        float(F.softplus(graph.geodesic_contrast.detach()).cpu()) if arm == "R" else EFFECTIVE_CONTRAST)
    return graph

