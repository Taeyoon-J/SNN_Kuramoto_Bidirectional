"""Degree-corrected modularity objective for SW0149 native32 spike Q.

This module is a small differentiable kernel. The source graph is a fixed
teacher; gradients flow through the student's production actual-spike Q.
"""
from __future__ import annotations

import torch


def source_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Symmetrize source graph weights and set self-edges to exactly zero."""
    if not torch.is_tensor(adjacency) or adjacency.ndim != 3:
        raise ValueError("source adjacency must have shape [B,N,N]")
    if adjacency.shape[0] < 1 or adjacency.shape[1] < 2 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("source adjacency must contain square graphs with at least two nodes")
    if not adjacency.is_floating_point():
        raise TypeError("source adjacency must be floating point")
    if not torch.isfinite(adjacency).all():
        raise ValueError("source adjacency contains nonfinite values")
    if (adjacency < 0).any():
        raise ValueError("raw source adjacency must be nonnegative")
    symmetric = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    diagonal = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device)
    symmetric = symmetric.masked_fill(diagonal.unsqueeze(0), 0.0)
    if (symmetric < 0).any():
        raise ValueError("symmetrized source adjacency must be nonnegative")
    return symmetric.detach()


def validate_modularity_inputs(adjacency: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Validate source A and production Q; return per-image total degree Z."""
    if not torch.is_tensor(adjacency) or not torch.is_tensor(q):
        raise TypeError("A and Q must be torch tensors")
    if adjacency.ndim != 3 or q.ndim != 3 or adjacency.shape != q.shape:
        raise ValueError("A and Q must have the same [B,N,N] shape")
    if adjacency.shape[0] < 1 or adjacency.shape[1] < 2 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("A and Q must contain square graphs with at least two nodes")
    if adjacency.device != q.device or adjacency.dtype != q.dtype:
        raise ValueError("A and Q must use the same device and dtype")
    if not adjacency.is_floating_point() or not q.is_floating_point():
        raise TypeError("A and Q must be floating point")
    if not torch.isfinite(adjacency).all() or not torch.isfinite(q).all():
        raise ValueError("A or Q contains nonfinite values")
    if (adjacency < 0).any():
        raise ValueError("source adjacency must be nonnegative")
    if not torch.allclose(adjacency, adjacency.transpose(-1, -2), rtol=0.0, atol=2e-6):
        raise ValueError("source adjacency must be symmetric")
    diag = torch.diagonal(adjacency, dim1=-2, dim2=-1)
    if (diag.abs() > 2e-6).any():
        raise ValueError("source adjacency diagonal must be zero")
    if not torch.allclose(q, q.transpose(-1, -2), rtol=0.0, atol=2e-6):
        raise ValueError("actual-spike Q must be symmetric")
    if (q < -1e-6).any() or (q > 1.0 + 1e-6).any():
        raise ValueError("actual-spike Q must lie in [0,1]")
    degree = adjacency.sum(dim=-1)
    z = degree.sum(dim=-1)
    if not torch.isfinite(z).all() or (z <= 0).any():
        raise ValueError("each source graph must have positive finite total degree Z")
    return z


def degree_corrected_modularity_loss(adjacency: torch.Tensor,
                                     q: torch.Tensor) -> torch.Tensor:
    """Compute mean_b -(M_b * Q_b).sum() / Z_b, M=A-ddᵀ/Z.

    ``adjacency`` is the detached frozen source graph. ``q`` remains live, so
    autograd reaches the native spike dynamics through the registered QCC.
    """
    z = validate_modularity_inputs(adjacency, q)
    if adjacency.requires_grad:
        raise ValueError("source adjacency teacher must be detached")
    q_diagonal = torch.diagonal(q, dim1=-2, dim2=-1)
    if not torch.allclose(q_diagonal, torch.ones_like(q_diagonal), rtol=0.0, atol=2e-6):
        raise ValueError("production actual-spike Q diagonal must be one")
    degree = adjacency.sum(dim=-1)
    expected = degree.unsqueeze(-1) * degree.unsqueeze(-2) / z[:, None, None]
    modularity = adjacency - expected
    per_image = -(modularity * q).sum(dim=(-2, -1)) / z
    loss = per_image.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("degree-corrected modularity loss is nonfinite")
    return loss
