"""Loss functions for the end-to-end S2Net autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def normal_rec_loss(
    reconstruction: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Calculate pixel-wise RGB mean squared reconstruction error.

    Args:
        reconstruction:
            Reconstructed RGB images shaped ``[B, 3, H, W]``.
        target:
            Original RGB images with the same shape as ``reconstruction``.
        reduction:
            Reduction passed directly to :func:`torch.nn.functional.mse_loss`.

    Returns:
        Pixel-wise RGB mean squared reconstruction error.
    """

    return F.mse_loss(
        reconstruction,
        target,
        reduction=reduction,
    )


def mask_diversity_loss(
    masks: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Calculate mean pairwise cosine similarity between distinct masks."""

    flattened_masks = masks.flatten(start_dim=2)
    normalized_masks = F.normalize(
        flattened_masks,
        p=2,
        dim=2,
        eps=eps,
    )
    pairwise_similarity = (
        normalized_masks @ normalized_masks.transpose(1, 2)
    )

    num_masks = masks.shape[1]
    off_diagonal = ~torch.eye(
        num_masks,
        dtype=torch.bool,
        device=masks.device,
    )
    return pairwise_similarity[:, off_diagonal].mean()


def generate_edge_map(
    target: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Generate a per-image normalized edge-strength map from RGB targets."""

    horizontal_diff = (
        target[:, :, :, 1:] - target[:, :, :, :-1]
    ).abs().mean(dim=1, keepdim=True)
    vertical_diff = (
        target[:, :, 1:, :] - target[:, :, :-1, :]
    ).abs().mean(dim=1, keepdim=True)

    horizontal_diff = F.pad(horizontal_diff, (0, 1, 0, 0))
    vertical_diff = F.pad(vertical_diff, (0, 0, 0, 1))
    edge_map = horizontal_diff + vertical_diff

    edge_max = edge_map.amax(dim=(2, 3), keepdim=True)
    return edge_map / edge_max.clamp_min(eps)


def weighted_reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    edge_scale: float = 3.0,
) -> Tensor:
    """Calculate edge-weighted pixel-wise RGB reconstruction error.

    An edge map is generated only from the target image and converted into
    spatial weights ranging approximately from ``1`` to ``1 + edge_scale``.
    The same weight is broadcast across all RGB channels. The weighted squared
    error is normalized by the total spatial weight and channel count to
    retain mean-style reduction.

    Args:
        reconstruction:
            Reconstructed RGB images shaped ``[B, 3, H, W]``.
        target:
            Original RGB target images with the same shape.
        edge_scale:
            Additional weight assigned in proportion to target edge strength.

    Returns:
        Normalized edge-weighted RGB reconstruction error.
    """

    edge_map = generate_edge_map(target)
    weights = 1.0 + float(edge_scale) * edge_map
    squared_error = (reconstruction - target).pow(2)
    weighted_error = squared_error * weights

    return weighted_error.sum() / (
        weights.sum() * reconstruction.shape[1]
    )


def mask_entropy_loss(
    masks: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Calculate mean pixel-wise entropy across softmax-normalized masks."""

    log_masks = torch.log(masks.clamp_min(eps))
    entropy = -(masks * log_masks).sum(dim=1)
    return entropy.mean()


def membrane_membership_consistency_loss(
    membrane_history: Tensor,
    object_vectors: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Match pairwise membrane-history and soft-membership similarities.

    Args:
        membrane_history:
            Oscillator membrane histories shaped ``[B, N, T]``.
        object_vectors:
            Soft object-membership vectors shaped ``[B, K, N]``.
        eps:
            Numerical stability value.

    Returns:
        Scalar consistency loss.
    """

    centered_history = (
        membrane_history
        - membrane_history.mean(dim=2, keepdim=True)
    )

    normalized_history = F.normalize(
        centered_history,
        p=2,
        dim=2,
        eps=eps,
    )

    history_similarity = (
        normalized_history
        @ normalized_history.transpose(1, 2)
    )
    history_similarity = (history_similarity + 1.0) / 2.0

    membership_similarity = (
        object_vectors.transpose(1, 2)
        @ object_vectors
    )

    num_oscillators = membrane_history.shape[1]

    off_diagonal = ~torch.eye(
        num_oscillators,
        dtype=torch.bool,
        device=membrane_history.device,
    )

    difference = history_similarity - membership_similarity

    return difference[:, off_diagonal].square().mean()
