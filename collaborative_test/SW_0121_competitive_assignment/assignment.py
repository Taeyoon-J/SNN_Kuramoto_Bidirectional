"""Competitive eleven-slot assignment and analytic RGB objective for SW0121."""
from __future__ import annotations

import torch
from torch import nn


COMPONENTS = 4
PATCHES = 256
SETTLED_FRAMES = 32
FEATURES = COMPONENTS * SETTLED_FRAMES
SLOTS = 11
HIDDEN = 64
TEMPERATURE = 0.1
EPS = 1e-8
VARIANCE_FLOOR = 1e-6


def patch_features(component_spikes):
    """Flatten D×settled-time into a 128D vector for each image patch."""
    if (component_spikes.ndim != 4
            or tuple(component_spikes.shape[1:]) != (COMPONENTS, PATCHES, SETTLED_FRAMES)):
        raise ValueError("component spikes must be [B,4,256,32] settled actual traces")
    if not torch.isfinite(component_spikes).all():
        raise FloatingPointError("component spike features contain nonfinite values")
    return component_spikes.permute(0, 2, 1, 3).reshape(
        component_spikes.shape[0], PATCHES, FEATURES)


@torch.no_grad()
def fit_channel_rms(training_features):
    """Fit fixed per-channel RMS from TRAIN-only [images,256,128] features."""
    if training_features.ndim != 3 or tuple(training_features.shape[1:]) != (PATCHES, FEATURES):
        raise ValueError("training features must be [M,256,128]")
    if training_features.shape[0] < 1 or not torch.isfinite(training_features).all():
        raise ValueError("training feature population must be nonempty and finite")
    rms = training_features.float().square().mean(dim=(0, 1)).sqrt()
    # Inactive/tiny channels are neutral-scaled, avoiding division by 1e-8.
    rms = torch.where(rms > EPS, rms, torch.ones_like(rms))
    if rms.shape != (FEATURES,) or not torch.isfinite(rms).all():
        raise FloatingPointError("invalid fixed channel RMS")
    return rms


class CompetitiveAssignmentHead(nn.Module):
    """Map normalized settled component traces to 11 uncount-constrained slots."""

    def __init__(self, channel_rms):
        super().__init__()
        rms = torch.as_tensor(channel_rms, dtype=torch.float32).detach().reshape(-1)
        if rms.shape != (FEATURES,) or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("channel_rms must be 128 finite positive frozen values")
        self.register_buffer("channel_rms", rms.reshape(1, 1, FEATURES).clone())
        self.network = nn.Sequential(
            nn.Linear(FEATURES, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, SLOTS))

    def forward(self, component_spikes):
        features = patch_features(component_spikes)
        normalized = features / self.channel_rms.to(device=features.device, dtype=features.dtype)
        logits = self.network(normalized)
        return torch.softmax(logits / TEMPERATURE, dim=-1)


def analytic_rgb_terms(q, assignment, rgb_patch_means):
    """Compute RGB reconstruction R and off-diagonal consistency C.

    Inputs are live Q [B,256,256], probabilities P [B,256,11], and detached
    native RGB patch means X [B,256,3]. Only R trains P; the target PPᵀ in C is
    detached by design, so C supplies direct affinity-Q credit.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (PATCHES, PATCHES):
        raise ValueError("Q must be [B,256,256]")
    if assignment.ndim != 3 or tuple(assignment.shape) != (q.shape[0], PATCHES, SLOTS):
        raise ValueError("assignment must be [B,256,11]")
    if rgb_patch_means.ndim != 3 or tuple(rgb_patch_means.shape) != (q.shape[0], PATCHES, 3):
        raise ValueError("native RGB patch means must be [B,256,3]")
    if not (torch.isfinite(q).all() and torch.isfinite(assignment).all()
            and torch.isfinite(rgb_patch_means).all()):
        raise FloatingPointError("SW0121 objective inputs must be finite")
    if (assignment < 0).any() or not torch.allclose(
            assignment.sum(dim=-1), torch.ones_like(assignment[..., 0]), rtol=1e-5, atol=1e-6):
        raise ValueError("assignment rows must be probability distributions")

    x = rgb_patch_means.detach().to(device=q.device, dtype=q.dtype)
    p = assignment.to(device=q.device, dtype=q.dtype)
    # Per image and slot: weighted native RGB mean, then reassign each patch.
    slot_mass = p.sum(dim=1).unsqueeze(-1).clamp_min(EPS)
    means = torch.bmm(p.transpose(1, 2), x) / slot_mass
    reconstruction = torch.bmm(p, means)
    variance = x.var(dim=1, unbiased=False).mean(dim=-1).detach().clamp_min(VARIANCE_FLOOR)
    r_each = (reconstruction - x).square().mean(dim=(1, 2)) / variance

    offdiag = ~torch.eye(PATCHES, dtype=torch.bool, device=q.device)
    target = torch.bmm(p, p.transpose(1, 2)).detach()
    c_each = (q - target).square()[:, offdiag].mean(dim=-1)
    return r_each.mean(), c_each.mean(), {
        "R_per_image": r_each,
        "C_per_image": c_each,
        "reconstruction": reconstruction,
        "slot_rgb_means": means,
        "rgb_variance": variance,
        "affinity_target_detached": target,
    }
