"""TRAIN-mask target guards and balanced actual-spike relation objective.

This module contains the fixed SW0155 loss only; it does not load data, choose
images, train a model, or read validation masks.
"""
from __future__ import annotations

import torch

from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity

GRID = 32
PIXEL_SIZE = 128
PATCH = PIXEL_SIZE // GRID
COMPONENTS = 4
SETTLE = 512
EPS = 1e-8
PAIR_CLASSES = (
    "same_foreground_instance",
    "different_foreground_instances",
    "foreground_background",
    "background_background",
)


def validate_training_prefix(image_ids, registered_ids, *, count=2048):
    """Require the exact immutable source prefix and reject reserved global IDs."""
    actual = [int(value) for value in image_ids]
    registered = [int(value) for value in registered_ids]
    if count <= 0 or len(registered) < count or len(set(registered)) != len(registered):
        raise ValueError("registered source TRAIN order is too short or contains duplicate IDs")
    if len(actual) != count or actual != registered[:count]:
        raise ValueError("supervised IDs must exactly equal the registered source TRAIN prefix")
    if any(1000 <= image_id <= 1639 or 90640 <= image_id <= 90959 for image_id in actual):
        raise ValueError("supervised training prefix contains a reserved evaluation ID")
    return actual


def modal_train_targets32(masks):
    """Production modal 4x4 conversion; retain ties/mixed patches and BG."""
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)
    if masks.ndim == 4 and tuple(masks.shape[1:]) == (PIXEL_SIZE, PIXEL_SIZE, 1):
        masks = masks[..., 0]
    if masks.ndim != 3 or tuple(masks.shape[1:]) != (PIXEL_SIZE, PIXEL_SIZE):
        raise ValueError("TRAIN masks must be [B,128,128] or [B,128,128,1]")
    if masks.is_floating_point() or masks.is_complex() or masks.dtype == torch.bool:
        raise ValueError("instance masks must contain integer IDs")
    if torch.any(masks < 0):
        raise ValueError("instance IDs must be nonnegative; background is zero")
    pooled = clevr_mask_patch(masks, PATCH)
    labels = pooled["patch_labels"]
    if tuple(labels.shape[1:]) != (GRID, GRID):
        raise AssertionError("production modal target converter returned an unexpected grid")
    return labels.reshape(labels.shape[0], GRID * GRID).to(torch.int64)


def _component_correlations(component_spikes, *, settle=SETTLE, eps=EPS):
    if (not torch.is_tensor(component_spikes) or component_spikes.ndim != 4
            or component_spikes.shape[1] != COMPONENTS):
        raise ValueError("component spikes must have shape [B,4,N,T]")
    if not 0 <= int(settle) < component_spikes.shape[-1]:
        raise ValueError("settle must leave at least one temporal sample")
    if not torch.isfinite(component_spikes).all():
        raise FloatingPointError("component spike trace contains nonfinite values")
    settled = component_spikes.float()[..., int(settle):]
    centered = settled - settled.mean(dim=-1, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    rho = torch.stack([
        (normalized[:, component] @ normalized[:, component].transpose(-1, -2)).clamp(-1.0, 1.0)
        for component in range(COMPONENTS)
    ], dim=1)
    return rho


def production_q_and_signed_rho(trace, *, settle=SETTLE):
    """Return the untouched production Q and its signed per-component Pearson terms."""
    spikes = trace.get("spikes")
    components = trace.get("component_spikes")
    if (not torch.is_tensor(spikes) or spikes.ndim != 3 or not torch.is_tensor(components)
            or components.ndim != 4 or tuple(components.shape[:1]) != tuple(spikes.shape[:1])
            or components.shape[2] != spikes.shape[1] or components.shape[3] != spikes.shape[2]):
        raise ValueError("trace must include matching spikes[B,N,T] and component_spikes[B,4,N,T]")
    q = spike_synchrony_affinity(spikes, components, settle=settle, affinity_mode="spike")
    rho = _component_correlations(components, settle=settle)
    reconstructed = rho.clamp_min(0.0).prod(dim=1)
    if not torch.allclose(q, reconstructed, rtol=0, atol=0):
        raise AssertionError("signed-rho decomposition no longer matches production actual-spike Q")
    return q, rho


def balanced_relation_objective(q, rho, labels):
    """Compute fixed per-image class-balanced L_Q + L_positive.

    Pair classes are unordered off-diagonal pairs. Each nonempty class is first
    averaged within image; nonempty class means are averaged per image; images
    are then averaged. Empty classes add neither value nor denominator.
    """
    if not torch.is_tensor(q) or q.ndim != 3 or q.shape[-1] != q.shape[-2]:
        raise ValueError("Q must have shape [B,N,N]")
    if not torch.is_tensor(rho) or rho.ndim != 4 or rho.shape[1] != COMPONENTS or rho.shape[0] != q.shape[0] or rho.shape[2:] != q.shape[1:]:
        raise ValueError("rho must have shape [B,4,N,N] matching Q")
    if not torch.is_tensor(labels) or labels.ndim != 2 or labels.shape != q.shape[:2]:
        raise ValueError("labels must have shape [B,N] matching Q")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool or torch.any(labels < 0):
        raise ValueError("labels must be nonnegative integer instance IDs")
    if not torch.isfinite(q).all() or not torch.isfinite(rho).all():
        raise FloatingPointError("Q or component correlations contain nonfinite values")
    if torch.any(q < 0) or torch.any(q > 1) or torch.any(rho < -1) or torch.any(rho > 1):
        raise ValueError("Q/rho values lie outside their production ranges")

    batch, nodes = labels.shape
    if nodes < 2:
        raise ValueError("relation objective requires at least two patches")
    rows, cols = torch.triu_indices(nodes, nodes, offset=1, device=q.device)
    left = labels[:, rows]
    right = labels[:, cols]
    left_fg, right_fg = left > 0, right > 0
    same_fg = left_fg & right_fg & (left == right)
    different_fg = left_fg & right_fg & (left != right)
    fg_bg = left_fg ^ right_fg
    bg_bg = (~left_fg) & (~right_fg)
    class_masks = (same_fg, different_fg, fg_bg, bg_bg)

    q_values = q[:, rows, cols]
    rho_values = rho[:, :, rows, cols].permute(0, 2, 1)  # [B, unordered pairs, 4]
    targets = (torch.ones_like(q_values), torch.zeros_like(q_values),
               torch.zeros_like(q_values), torch.ones_like(q_values))
    q_class_means, q_present = [], []
    for pair_mask, target in zip(class_masks, targets):
        count = pair_mask.sum(dim=1)
        error = (q_values - target).square()
        mean = (error * pair_mask.to(error.dtype)).sum(dim=1) / count.clamp_min(1).to(error.dtype)
        q_class_means.append(mean)
        q_present.append((count > 0).to(error.dtype))
    q_means = torch.stack(q_class_means, dim=1)
    q_mask = torch.stack(q_present, dim=1)
    q_per_image = (q_means * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp_min(1)
    loss_q = q_per_image.mean()

    pos_means, pos_present = [], []
    for pair_mask in (same_fg, bg_bg):
        count = pair_mask.sum(dim=1)
        error = (1.0 - rho_values).square().mean(dim=-1)
        mean = (error * pair_mask.to(error.dtype)).sum(dim=1) / count.clamp_min(1).to(error.dtype)
        pos_means.append(mean)
        pos_present.append((count > 0).to(error.dtype))
    pos_means = torch.stack(pos_means, dim=1)
    pos_mask = torch.stack(pos_present, dim=1)
    positive_per_image = (pos_means * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    loss_positive = positive_per_image.mean()
    total = loss_q + loss_positive
    details = {
        "loss_q": loss_q,
        "loss_positive": loss_positive,
        "pair_counts": torch.stack([mask.sum(dim=1) for mask in class_masks], dim=1),
        "per_image_loss_q": q_per_image,
        "per_image_loss_positive": positive_per_image,
    }
    return total, details


def supervised_relation_loss(trace, masks_or_labels, *, settle=SETTLE):
    """Convenience path from actual traces and either TRAIN masks or 32-grid IDs."""
    q, rho = production_q_and_signed_rho(trace, settle=settle)
    shape = tuple(masks_or_labels.shape)
    is_pixel_mask = ((len(shape) == 3 and shape[1:] == (PIXEL_SIZE, PIXEL_SIZE))
                     or (len(shape) == 4 and shape[1:] == (PIXEL_SIZE, PIXEL_SIZE, 1)))
    if is_pixel_mask:
        labels = modal_train_targets32(masks_or_labels)
    else:
        labels = masks_or_labels
    labels = labels.to(device=q.device) if torch.is_tensor(labels) else torch.as_tensor(labels, device=q.device)
    loss, details = balanced_relation_objective(q, rho, labels)
    details = dict(details, q=q, rho=rho, labels=labels)
    return loss, details
