from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
