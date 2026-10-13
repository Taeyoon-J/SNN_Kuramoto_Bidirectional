"""Detached deterministic seed selection from actual-spike Q and current QCC."""
from __future__ import annotations

import torch

GRID = 32
NODES = GRID * GRID
MAX_FOREGROUND_SEEDS = 10


def _validate_inputs(actual_q, qcc_labels):
    if not torch.is_tensor(actual_q) or actual_q.ndim != 3:
        raise ValueError("actual_q must have shape [B,1024,1024]")
    if actual_q.shape[0] < 1 or actual_q.shape[1:] != (NODES, NODES):
        raise ValueError("actual_q must use the fixed native32 32x32 patch grid")
    if not actual_q.is_floating_point():
        raise TypeError("actual_q must be floating point")
    if not torch.isfinite(actual_q).all():
        raise ValueError("actual_q must be finite")
    if torch.any(actual_q < 0) or torch.any(actual_q > 1):
        raise ValueError("actual-spike Q values must lie in [0,1]")
    if not torch.allclose(actual_q, actual_q.transpose(1, 2), rtol=1e-5, atol=1e-6):
        raise ValueError("actual-spike Q must be symmetric")
    if not torch.is_tensor(qcc_labels):
        qcc_labels = torch.as_tensor(qcc_labels)
    if qcc_labels.ndim != 3 or tuple(qcc_labels.shape) != (actual_q.shape[0], GRID, GRID):
        raise ValueError("QCC labels must have shape [B,32,32]")
    if qcc_labels.is_floating_point() or qcc_labels.is_complex() or qcc_labels.dtype == torch.bool:
        raise TypeError("QCC labels must have integer dtype")
    if torch.any(qcc_labels < 0):
        raise ValueError("QCC labels must be nonnegative integer region IDs")
    # Transfer once, explicitly detached, and use CPU float64 for deterministic
    # comparisons and exact tie handling independent of CUDA reduction order.
    q = actual_q.detach().to(device="cpu", dtype=torch.float64)
    labels = qcc_labels.detach().to(device="cpu", dtype=torch.int64).reshape(-1, NODES)
    return q, labels


def _smallest_index_at_max(values, candidates):
    maximum = values.max()
    tied = candidates[values == maximum]
    return int(tied.min().item())


def select_seed_ids(actual_q: torch.Tensor, qcc_labels) -> list[list[int]]:
    """Select `[background, foreground...]` patch IDs for each batch image.

    Background label 0 is preferred when present. Otherwise the largest QCC
    region is background, tied by its smallest patch index. Its seed is the
    patch with greatest mean Q to that region. The number of foreground seeds
    is the number of other distinct QCC regions, capped at ten; foreground
    seed patches themselves are selected by deterministic farthest-point
    distance over `1-Q`, starting from the background medoid.
    """
    q, labels = _validate_inputs(actual_q, qcc_labels)
    output = []
    for image_q, image_labels in zip(q, labels):
        unique, counts = torch.unique(image_labels, sorted=True, return_counts=True)
        regions = {int(label.item()): int(count.item()) for label, count in zip(unique, counts)}
        first = {label: int(torch.nonzero(image_labels == label, as_tuple=False)[0, 0].item())
                 for label in regions}
        if 0 in regions:
            background_label = 0
        else:
            background_label = min(regions, key=lambda label: (-regions[label], first[label]))

        background_patches = torch.nonzero(image_labels == background_label, as_tuple=False).reshape(-1)
        if background_patches.numel() == 0:
            raise AssertionError("selected QCC background region unexpectedly has no patches")
        bg_to_region = image_q.index_select(0, background_patches).index_select(1, background_patches)
        bg_mean = bg_to_region.mean(dim=1)
        background_seed = _smallest_index_at_max(bg_mean, background_patches)

        foreground_count = min(MAX_FOREGROUND_SEEDS,
                               sum(1 for label in regions if label != background_label))
        foreground_patches = torch.nonzero(image_labels != background_label, as_tuple=False).reshape(-1)
        if foreground_count > foreground_patches.numel():
            raise ValueError("foreground seed count exceeds available non-background patches")
        selected = [background_seed]
        if foreground_count:
            q_from_bg = image_q[background_seed].index_select(0, foreground_patches)
            first_distance = 1.0 - q_from_bg
            selected.append(_smallest_index_at_max(first_distance, foreground_patches))

            while len(selected) < foreground_count + 1:
                selected_tensor = torch.as_tensor(selected, dtype=torch.long)
                remaining_mask = torch.ones(NODES, dtype=torch.bool)
                remaining_mask[background_patches] = False
                remaining_mask[selected_tensor] = False
                candidates = torch.nonzero(remaining_mask, as_tuple=False).reshape(-1)
                if candidates.numel() == 0:
                    raise ValueError("not enough unselected foreground patches for registered seed count")
                q_to_selected = image_q.index_select(0, selected_tensor).index_select(1, candidates)
                min_distance = (1.0 - q_to_selected).min(dim=0).values
                selected.append(_smallest_index_at_max(min_distance, candidates))
        output.append(selected)
    return output
