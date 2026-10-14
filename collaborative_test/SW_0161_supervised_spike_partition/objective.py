"""Two TRAIN-mask objectives for an exchangeable 11-slot spike partition.

The module is intentionally data- and model-independent: it accepts learned
patch probabilities and integer TRAIN labels and never reads validation data.
"""
from __future__ import annotations

import torch

GRID = 32
PATCHES = GRID * GRID
SLOTS = 11
MIN_GROUP = 8
EPS = 1e-8
PAIR_CLASSES = ("same_fg", "different_fg", "fg_bg", "bg_bg")


def _check_probabilities(probabilities: torch.Tensor) -> None:
    if not torch.is_tensor(probabilities) or probabilities.ndim != 3:
        raise ValueError("P must have shape [B,1024,11]")
    if tuple(probabilities.shape[1:]) != (PATCHES, SLOTS):
        raise ValueError("P must have shape [B,1024,11]")
    if probabilities.shape[0] == 0:
        raise ValueError("P batch must be nonempty")
    if not probabilities.is_floating_point():
        raise ValueError("P must be floating point")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("P contains nonfinite values")
    if torch.any(probabilities < 0):
        raise ValueError("P must be nonnegative")
    sums = probabilities.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=0.0, atol=2e-5):
        raise ValueError("each patch probability row must sum to one")


def _check_labels(labels: torch.Tensor, batch: int, nodes: int) -> torch.Tensor:
    if not torch.is_tensor(labels) or labels.ndim != 2 or tuple(labels.shape) != (batch, nodes):
        raise ValueError("labels must have shape [B,1024] matching P")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("instance labels must be integer IDs")
    if torch.any(labels < 0):
        raise ValueError("instance labels must be nonnegative")
    if batch == 0:
        raise ValueError("labels batch must be nonempty")
    for row in labels:
        if torch.unique(row).numel() > SLOTS:
            raise ValueError(f"GT label count cannot exceed {SLOTS} slots")
    return labels.to(dtype=torch.int64)


def balanced_relation_loss(probabilities: torch.Tensor, labels: torch.Tensor):
    """SW155-style class-balanced coassignment MSE, averaged per image.

    Unordered off-diagonal pairs are grouped into same-FG, different-FG,
    FG/BG, and BG/BG. Nonempty pair classes receive equal weight within each
    image; images then receive equal weight.
    """
    _check_probabilities(probabilities)
    labels = _check_labels(labels, probabilities.shape[0], probabilities.shape[1]).to(probabilities.device)
    b, n, _ = probabilities.shape
    if n < 2:
        raise ValueError("relation loss requires at least two patches")
    i, j = torch.triu_indices(n, n, offset=1, device=probabilities.device)
    left, right = labels[:, i], labels[:, j]
    lf, rf = left > 0, right > 0
    masks = (
        lf & rf & (left == right),
        lf & rf & (left != right),
        lf ^ rf,
        (~lf) & (~rf),
    )
    coassign = (probabilities[:, i] * probabilities[:, j]).sum(dim=-1)
    targets = (1.0, 0.0, 0.0, 1.0)
    means, present, counts = [], [], []
    for mask, target in zip(masks, targets):
        count = mask.sum(dim=1)
        error = (coassign - target).square()
        means.append((error * mask.to(error.dtype)).sum(dim=1) / count.clamp_min(1))
        present.append((count > 0).to(error.dtype))
        counts.append(count)
    class_means = torch.stack(means, dim=1)
    class_present = torch.stack(present, dim=1)
    per_image = (class_means * class_present).sum(dim=1) / class_present.sum(dim=1).clamp_min(1)
    loss = per_image.mean()
    return loss, {
        "per_image_loss": per_image,
        "pair_counts": torch.stack(counts, dim=1),
        "coassignment": coassign,
    }


def _hungarian(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Return minimum-cost row/column pairs; SciPy is used only for matching."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("Hungarian matching requires scipy") from exc
    rows, cols = linear_sum_assignment(cost.detach().to(device="cpu", dtype=torch.float64).numpy())
    return list(zip(rows.tolist(), cols.tolist()))


def mask_hungarian_loss(probabilities: torch.Tensor, labels: torch.Tensor):
    """Per-image all-instance Hungarian CE + all-slot soft Dice.

    Every distinct label present in the TRAIN patch targets, including label 0,
    is matched to one of the 11 slots. A target with more than 11 labels is a
    hard error. The discrete matching is detached; selected CE/Dice remain
    differentiable with respect to P.
    """
    _check_probabilities(probabilities)
    labels = _check_labels(labels, probabilities.shape[0], probabilities.shape[1]).to(probabilities.device)
    b, n, k = probabilities.shape
    image_losses, image_ce, image_dice, assignments, label_counts = [], [], [], [], []
    for batch_index in range(b):
        row_labels = labels[batch_index]
        gt_ids = torch.unique(row_labels, sorted=True)
        if gt_ids.numel() == 0 or gt_ids.numel() > k:
            raise ValueError(f"GT label count must be in [1,{k}] for every image")
        targets = torch.stack([(row_labels == value) for value in gt_ids], dim=0).to(probabilities.dtype)
        p = probabilities[batch_index]
        eps = torch.finfo(p.dtype).eps
        logp = p.clamp_min(eps).log()
        # [instances, slots], detached only for the combinatorial assignment.
        ce_cost = -(targets @ logp) / targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        intersection = targets @ p
        denominator = targets.sum(dim=1, keepdim=True) + p.sum(dim=0, keepdim=True)
        dice_cost = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        pairs = _hungarian(ce_cost + dice_cost)
        matched_slots = [slot for _, slot in sorted(pairs)]
        matched_targets = [target_index for target_index, _ in sorted(pairs)]
        matched_p = p[:, matched_slots]
        matched_y = targets[matched_targets].transpose(0, 1)

        ce_per_target = -(matched_y * matched_p.clamp_min(eps).log()).sum(dim=0) / matched_y.sum(dim=0).clamp_min(1.0)
        ce = ce_per_target.mean()
        # Include all 11 slots: unmatched slots have all-zero target masks.
        full_target = torch.zeros_like(p)
        full_target[:, matched_slots] = matched_y
        dice_by_slot = 1.0 - (2.0 * (p * full_target).sum(dim=0) + 1.0) / (
            p.sum(dim=0) + full_target.sum(dim=0) + 1.0
        )
        dice = dice_by_slot.mean()
        image_ce.append(ce)
        image_dice.append(dice)
        image_losses.append(ce + dice)
        assignments.append(tuple((int(gt_ids[t]), int(slot)) for t, slot in pairs))
        label_counts.append(int(gt_ids.numel()))
    ce_mean = torch.stack(image_ce).mean()
    dice_mean = torch.stack(image_dice).mean()
    return ce_mean + dice_mean, {
        "loss_ce": ce_mean,
        "loss_dice": dice_mean,
        "per_image_loss": torch.stack(image_losses),
        "matched_labels_to_slots": assignments,
        "gt_label_counts": label_counts,
    }


def canonical_hard_labels(probabilities: torch.Tensor, *, min_group: int = MIN_GROUP) -> torch.Tensor:
    """Largest hard-assigned slot is BG; return canonical labels [B,32,32].

    Ties for the background group are broken by its first row-major patch, then
    slot ID. Groups smaller than ``min_group`` are remapped to 0. Remaining
    groups receive IDs in order of their first patch. GT is never consulted.
    """
    _check_probabilities(probabilities)
    if not isinstance(min_group, int) or isinstance(min_group, bool) or min_group < 1:
        raise ValueError("min_group must be a positive integer")
    hard = probabilities.detach().argmax(dim=-1).cpu()
    result = torch.zeros((probabilities.shape[0], PATCHES), dtype=torch.int64)
    for b, row in enumerate(hard):
        values = row.tolist()
        groups = []
        for slot in range(SLOTS):
            positions = [ix for ix, value in enumerate(values) if value == slot]
            if positions:
                groups.append((slot, len(positions), positions[0]))
        if not groups:
            continue
        bg = min(groups, key=lambda entry: (-entry[1], entry[2], entry[0]))[0]
        foreground = sorted(
            (entry for entry in groups if entry[0] != bg and entry[1] >= min_group),
            key=lambda entry: (entry[2], entry[0]),
        )
        slot_to_id = {entry[0]: rank for rank, entry in enumerate(foreground, start=1)}
        result[b] = torch.tensor([slot_to_id.get(slot, 0) for slot in values], dtype=torch.int64)
    return result.reshape(probabilities.shape[0], GRID, GRID)
