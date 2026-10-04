"""Patch-level ground truth utilities for object-grouping evaluation."""

import torch
from torch import Tensor


def clevr_mask_patch(
    masks: Tensor,
    patch_size: int | tuple[int, int],
) -> dict[str, Tensor]:
    """Assign each non-overlapping patch its most frequent CLEVR instance ID.

    Args:
        masks: Integer instance labels shaped [B, H, W] or [B, H, W, 1].
            Background has ID 0 and participates in voting.
        patch_size: Patch height/width in label-map pixels, or one integer
            for square patches. Both image dimensions must divide exactly;
            no resizing, padding, or cropping is performed.

    Returns:
        Dictionary containing tensors shaped [B, H // patch_h, W // patch_w]:
        ``patch_labels`` (int64): Most frequent ID. Ties select the smallest
        ID, so background wins whenever it is among the tied candidates.
        ``patch_purity`` (float32): Fraction of pixels with the selected ID.
        ``patch_ties`` (bool): Whether multiple IDs share the maximum count.
        All tensors remain on the input device. Low-purity and background
        patches are retained; this function does not filter evaluation data.

    The caller must align the label map and patch size with the model's output
    mask grid. Feature-map pooling receptive fields are not inferred here.
    """
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B, H, W] or [B, H, W, 1].")
    if masks.is_floating_point() or masks.is_complex() or masks.dtype == torch.bool:
        raise ValueError("masks must contain integer instance IDs.")

    patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    batch, height, width = masks.shape
    if patch_h <= 0 or patch_w <= 0 or height % patch_h or width % patch_w:
        raise ValueError("Positive patch dimensions must divide the mask height and width exactly.")

    labels = masks.to(dtype=torch.int64)
    patches = labels.unfold(1, patch_h, patch_h).unfold(2, patch_w, patch_w)
    patches = patches.flatten(start_dim=-2)
    instance_ids = torch.unique(labels, sorted=True)
    counts = (patches.unsqueeze(-1) == instance_ids).sum(dim=-2)
    max_counts, selected = counts.max(dim=-1)

    return {
        "patch_labels": instance_ids[selected],
        "patch_purity": max_counts.to(torch.float32) / (patch_h * patch_w),
        "patch_ties": (counts == max_counts.unsqueeze(-1)).sum(dim=-1) > 1,
    }


def spatial_components_to_patch_labels(
    object_groups: list[list[tuple[int, ...]]],
    patch_grid_size: int | tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Convert spike_spatial_components output to evaluation patch labels.

    Args:
        object_groups: Batch list of groups of flat oscillator indices, exactly
            as returned by spike_spatial_components. Indices use row-major
            order: index = row * grid_w + column. Groups must be disjoint.
        patch_grid_size: Grid height/width, or one integer for a square grid.
            This is the number of patches, NOT the patch size in pixels.
        device: Output device (CPU by default); use the target label device
            when passing the result to evaluate_patch_masks.

    Returns:
        Int64 labels [B, grid_h, grid_w]. Groups receive IDs 1, 2, ... in
        classifier order, independently per image. Unassigned patches are 0.
        An image with no groups is all background. No regrouping, thresholding,
        or ground-truth matching is performed. Overlapping groups are rejected
        rather than silently overwriting labels.
    """
    grid_h, grid_w = (patch_grid_size, patch_grid_size) if isinstance(patch_grid_size, int) else patch_grid_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("Patch grid dimensions must be positive.")
    labels = torch.zeros((len(object_groups), grid_h * grid_w), dtype=torch.int64, device=device)
    for batch_index, groups in enumerate(object_groups):
        for object_id, group in enumerate(groups, start=1):
            if len(group) == 0:
                continue
            indices = torch.as_tensor(group, device=labels.device)
            if indices.ndim != 1 or indices.is_floating_point() or indices.is_complex() or indices.dtype == torch.bool:
                raise ValueError("Each group must contain a one-dimensional sequence of integer indices.")
            indices = indices.to(torch.int64)
            if ((indices < 0) | (indices >= grid_h * grid_w)).any():
                raise ValueError("Oscillator index is outside the patch grid.")
            if (labels[batch_index, indices] != 0).any():
                raise ValueError("Spatial component groups must not overlap.")
            labels[batch_index, indices] = object_id
    return labels.reshape(len(object_groups), grid_h, grid_w)


def _validate_patch_labels(prediction: Tensor, target: Tensor) -> None:
    """Require nonempty, aligned integer label maps on the same device."""
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("Expected matching [B, grid_h, grid_w] label maps.")
    if prediction.numel() == 0 or prediction.device != target.device:
        raise ValueError("Label maps must be nonempty and on the same device.")
    for labels in (prediction, target):
        if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
            raise ValueError("Instance labels must have integer dtype.")


@torch.no_grad()
def patch_fg_ari(prediction: Tensor, target: Tensor, background_id: int = 0) -> Tensor:
    """Return per-image foreground ARI [B] for [B, grid_h, grid_w] labels.

    Exclude only target background. Predicted background on target foreground
    remains a cluster. IDs need not match. Fewer than two foreground patches
    gives NaN; identical degenerate partitions score 1. Returns float64 on the
    input device. No pixel expansion or purity weighting is applied.
    """
    _validate_patch_labels(prediction, target)
    scores = torch.full((target.shape[0],), float("nan"), dtype=torch.float64, device=target.device)
    for b in range(target.shape[0]):
        foreground = target[b] != background_id
        n = int(foreground.sum())
        if n < 2:
            continue
        true_ids, ti = torch.unique(target[b][foreground], return_inverse=True)
        pred_ids, pi = torch.unique(prediction[b][foreground], return_inverse=True)
        counts = torch.bincount(ti * pred_ids.numel() + pi,
                                minlength=true_ids.numel() * pred_ids.numel())
        counts = counts.reshape(true_ids.numel(), pred_ids.numel()).to(torch.float64)
        rows, columns = counts.sum(1), counts.sum(0)
        common = (counts * (counts - 1) / 2).sum()
        true_pairs = (rows * (rows - 1) / 2).sum()
        pred_pairs = (columns * (columns - 1) / 2).sum()
        expected = true_pairs * pred_pairs / (n * (n - 1) / 2)
        denominator = (true_pairs + pred_pairs) / 2 - expected
        scores[b] = (common - expected) / denominator if denominator != 0 else 1.0
    return scores


@torch.no_grad()
def patch_foreground_iou(prediction: Tensor, target: Tensor, background_id: int = 0) -> Tensor:
    """Return per-image foreground IoU [B] for aligned integer patch labels.

    Merge all non-background IDs, then count intersection and union patches.
    Both foreground sets empty scores 1; only one empty scores 0. Inputs have
    shape [B, grid_h, grid_w]; outputs are float64 on the input device.
    """
    _validate_patch_labels(prediction, target)
    pred_fg, true_fg = prediction != background_id, target != background_id
    intersection = (pred_fg & true_fg).sum((1, 2)).to(torch.float64)
    union = (pred_fg | true_fg).sum((1, 2)).to(torch.float64)
    return torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))


@torch.no_grad()
def patch_matched_object_iou(prediction: Tensor, target: Tensor, background_id: int = 0) -> Tensor:
    """Return per-image Hungarian-matched object IoU [B] for patch labels.

    Inputs: [B, grid_h, grid_w] integer labels. Maximize total one-to-one IoU,
    then divide by target object count (unmatched target objects score zero).
    Background is not matched, but false foreground increases object unions.
    Extra predictions receive no direct penalty. No target objects gives NaN;
    no predictions with nonempty targets gives 0. SciPy matches on CPU, while
    float64 results are returned on the input device.
    """
    from scipy.optimize import linear_sum_assignment

    _validate_patch_labels(prediction, target)
    scores = torch.full((target.shape[0],), float("nan"), dtype=torch.float64, device=target.device)
    for b in range(target.shape[0]):
        true_ids, pred_ids = torch.unique(target[b]), torch.unique(prediction[b])
        true_ids, pred_ids = true_ids[true_ids != background_id], pred_ids[pred_ids != background_id]
        if true_ids.numel() == 0:
            continue
        if pred_ids.numel() == 0:
            scores[b] = 0
            continue
        true_masks = (target[b].reshape(1, -1) == true_ids[:, None]).to(torch.float64)
        pred_masks = (prediction[b].reshape(1, -1) == pred_ids[:, None]).to(torch.float64)
        intersection = true_masks @ pred_masks.T
        union = true_masks.sum(1, keepdim=True) + pred_masks.sum(1)[None, :] - intersection
        iou = intersection / union
        rows, columns = linear_sum_assignment(-iou.cpu().numpy())
        rows = torch.as_tensor(rows, device=target.device)
        columns = torch.as_tensor(columns, device=target.device)
        scores[b] = iou[rows, columns].sum() / true_ids.numel()
    return scores


def evaluate_patch_masks(
    prediction: Tensor, target: Tensor, background_id: int = 0,
) -> dict[str, dict[str, Tensor]]:
    """Evaluate three metrics on aligned [B, grid_h, grid_w] patch labels.

    Return ``per_image`` ([B]), ``mean`` (scalar image-wise average), and
    ``valid_count`` (non-NaN image count) dictionaries keyed by metric name.
    Undefined scores are excluded per metric; all-undefined means remain NaN.
    No combined weighted score is formed. Each patch must have exactly one
    ID: resolve overlapping masks before calling, without using ground truth.
    """
    per_image = {
        "fg_ari": patch_fg_ari(prediction, target, background_id),
        "foreground_iou": patch_foreground_iou(prediction, target, background_id),
        "matched_object_iou": patch_matched_object_iou(prediction, target, background_id),
    }
    return {
        "per_image": per_image,
        "mean": {name: scores.nanmean() for name, scores in per_image.items()},
        "valid_count": {name: (~scores.isnan()).sum() for name, scores in per_image.items()},
    }
