"""Local SW0120 affinity variant with a backward-only degenerate-norm guard.

The forward normalization, matrix products, clamps, component stacking and
positive product match ``spike_synchrony_affinity(..., affinity_mode='spike')``.
Only gradients through centered trace norms at or below the existing epsilon
are detached.
"""
from __future__ import annotations

import torch


def guarded_spike_affinity(activity, components=None, settle=0, eps=1e-8):
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if not (eps > 0):
        raise ValueError("eps must be positive")
    activity = activity.float()
    if int(settle) > 0:
        activity = activity[:, :, int(settle):]
        if components is not None:
            components = components[..., int(settle):]

    def correlate(x):
        x = x - x.mean(dim=-1, keepdim=True)
        norm = x.norm(dim=-1, keepdim=True)
        normalized = x / norm.clamp_min(eps)
        normalized = torch.where(norm > eps, normalized, normalized.detach())
        return (normalized @ normalized.transpose(-1, -2)).clamp(-1.0, 1.0)

    if components is None:
        return correlate(activity)
    if (components.dim() != 4 or components.size(0) != activity.size(0)
            or components.size(2) != activity.size(1)):
        raise ValueError("components must have shape [B, D, N, T].")
    per = [correlate(components[:, d].float()) for d in range(components.size(1))]
    return torch.stack(per).clamp_min(0.0).prod(dim=0)
