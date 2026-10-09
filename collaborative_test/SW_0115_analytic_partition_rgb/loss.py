"""RGB reconstruction credit through the registered hard spike partition.

The forward prediction uses the exact production components.  The straight-
through assignment contributes an approximate gradient to the live, actual-
component spike affinity; it does not alter the classifier or add prototypes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components


GRID_SIZE = 16
N_PATCHES = GRID_SIZE * GRID_SIZE
RGB_CHANNELS = 3
ASSIGNMENT_TEMPERATURE = 0.1
CLASSIFIER_THRESHOLD = 0.5
MIN_GROUP_SIZE = 2


def production_partition(activity, component_activity, *, settle=32):
    """Return per-image production labels and one-hot H matrices.

    ``activity`` is [B,N,T], ``component_activity`` is [B,D,N,T]. The
    production classifier owns background selection and drops components
    smaller than two. All omitted patches therefore remain in label/column 0.
    K is allowed to vary by image; each returned H is [N,K_image].
    """
    if activity.ndim != 3 or activity.shape[1] != N_PATCHES:
        raise ValueError("activity must have shape [B,256,T]")
    if (component_activity.ndim != 4
            or component_activity.shape[0] != activity.shape[0]
            or component_activity.shape[2] != N_PATCHES
            or component_activity.shape[-1] != activity.shape[-1]):
        raise ValueError("component_activity must have shape [B,D,256,T]")

    # Classification is intentionally nondifferentiable and matches the
    # registered CPU classifier invocation used by the training/eval code.
    groups = spike_synchrony_components(
        activity.detach().cpu(),
        synchrony_threshold=CLASSIFIER_THRESHOLD,
        min_group_size=MIN_GROUP_SIZE,
        settle=settle,
        components=component_activity.detach().cpu(),
        background="largest_component",
        affinity_mode="spike",
        spatial_grid_size=GRID_SIZE,
    )
    labels = spatial_components_to_patch_labels(
        groups, GRID_SIZE, device=activity.device
    ).reshape(activity.shape[0], N_PATCHES)
    hard = []
    for row in labels:
        k = int(row.max().item()) + 1
        h = F.one_hot(row, num_classes=k).to(dtype=activity.dtype)
        if not torch.equal(h.sum(dim=-1), torch.ones_like(row, dtype=h.dtype)):
            raise AssertionError("production hard partition is not one-hot")
        hard.append(h)
    return labels, hard


def assignment_weights(q, hard, *, temperature=ASSIGNMENT_TEMPERATURE):
    """Build the fixed-temperature Q-derived STE assignment for one image."""
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must have shape [N,N]")
    if hard.ndim != 2 or hard.shape[0] != q.shape[0] or hard.shape[1] < 1:
        raise ValueError("hard must have shape [N,K] with K>=1")
    if temperature != ASSIGNMENT_TEMPERATURE:
        raise ValueError("assignment temperature is preregistered at 0.1")
    h = hard.detach().to(device=q.device, dtype=q.dtype)
    eye = torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
    q0 = q * (1.0 - eye)
    count = h.sum(dim=0)
    denom = (count.unsqueeze(0) - h).clamp_min(1.0)
    affinity = (q0 @ h) / denom
    p = torch.softmax(affinity / ASSIGNMENT_TEMPERATURE, dim=-1)
    w = h + (p - p.detach())
    if not torch.equal(w.detach(), h):
        raise AssertionError("STE changed the classifier's hard forward partition")
    return w, p


def reconstruct_image(q, hard, rgb, *, variance_floor=1e-6):
    """Return normalized native-patch RGB loss and its hard-forward prediction.

    ``rgb`` is detached [N,3] native 8x8 patch means in [0,1]. The ordinary
    channel/spatial mean variance is detached and clamped as registered.
    """
    if rgb.ndim != 2 or rgb.shape != (N_PATCHES, RGB_CHANNELS):
        raise ValueError("rgb must have shape [256,3]")
    if variance_floor != 1e-6:
        raise ValueError("variance floor is preregistered at 1e-6")
    x = rgb.detach().to(device=q.device, dtype=q.dtype)
    w, p = assignment_weights(q, hard)
    mu = (w.transpose(0, 1) @ x) / w.sum(dim=0).unsqueeze(-1).clamp_min(1e-8)
    prediction = w @ mu
    variance = x.var(dim=0, unbiased=False).mean().detach().clamp_min(variance_floor)
    loss = (prediction - x).square().mean() / variance
    return loss, prediction, {"W": w, "P": p, "mu": mu, "variance": variance}


def batch_reconstruction_loss(q, hard_partitions, rgb):
    """Ordinary image mean, retaining K=1 and blank images in the batch."""
    if q.ndim != 3 or q.shape[1:] != (N_PATCHES, N_PATCHES):
        raise ValueError("q must have shape [B,256,256]")
    if rgb.shape != (q.shape[0], N_PATCHES, RGB_CHANNELS):
        raise ValueError("rgb must have shape [B,256,3]")
    if len(hard_partitions) != q.shape[0]:
        raise ValueError("one variable-K hard partition is required per image")
    losses, predictions, details = [], [], []
    for i, h in enumerate(hard_partitions):
        loss, pred, info = reconstruct_image(q[i], h, rgb[i])
        losses.append(loss)
        predictions.append(pred)
        details.append(info)
    return torch.stack(losses).mean(), torch.stack(losses), predictions, details


def rgb_patch_means(images):
    """RGB float images [B,3,128,128] in [0,1] -> native [B,256,3] means."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 128, 128):
        raise ValueError("images must have shape [B,3,128,128]")
    return F.avg_pool2d(images, kernel_size=8, stride=8).permute(0, 2, 3, 1).reshape(
        images.shape[0], N_PATCHES, RGB_CHANNELS
    )
