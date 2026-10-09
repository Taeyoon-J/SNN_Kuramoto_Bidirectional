"""Loss-free GT-conditioned allowed-patch-label oracle scorer.

This diagnostic is not a standard metric: it uses each patch's pixel-GT label
set to resolve the target labels before computing the existing patch FG-ARI.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch, patch_fg_ari


def _integer_tensor(value, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must contain integer labels")
        tensor = value.detach().to(device="cpu", dtype=torch.int64).clone()
    else:
        array = np.asarray(value)
        if array.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer labels")
        tensor = torch.as_tensor(array, dtype=torch.int64, device="cpu").clone()
    return tensor


def score_allowed_patch_labels(prediction_labels, pixel_ground_truth_labels) -> dict:
    """Compute one image's allowed-label oracle FG-ARI and diagnostics.

    `prediction_labels` is an integer [16,16] patch grid. Ground truth is an
    integer [128,128] pixel-label map (or [128,128,1]). Each patch's allowed
    set contains every pixel label present in its 8x8 region, including 0.

    Candidate labels (0 for predicted background, otherwise the group's
    assigned GT ID) are checked uniformly against each patch's allowed set.
    A disallowed candidate falls back to the modal GT label, so a predicted
    background patch on a pure-object footprint can retain a foreground target.
    Acceptance rate measures candidate compatibility before that fallback.
    Sorted non-background GT IDs are matched one-to-one to predicted groups with a maximum-weight
    rectangular Hungarian assignment; weights count patches where that GT ID
    is in the patch's allowed set. Predicted groups are ordered by first
    row-major patch occurrence, making exact assignment ties independent of
    arbitrary predicted-ID names. SciPy resolves remaining equal optima
    deterministically from these stable row/column orders; no epsilon changes
    the integer objective. Unmatched/zero-overlap assignments receive distinct
    negative sentinels, which are never in any allowed set.

    For each patch, the resolved target is its candidate if accepted,
    otherwise the original 8x8 modal GT ID (ties use the
    existing `clevr_mask_patch` rule: smallest ID). The returned ARI is the
    existing `patch_fg_ari(prediction, resolved_target)` calculation, but is
    explicitly an oracle diagnostic rather than a standard FG-ARI result.
    Undefined scores (<2 resolved foreground patches) are returned as None
    with `score_valid=False`, so callers can serialize strict JSON safely.
    Inputs are not modified.
    """
    pred = _integer_tensor(prediction_labels, "prediction_labels")
    pixels = _integer_tensor(pixel_ground_truth_labels, "pixel_ground_truth_labels")
    if tuple(pred.shape) != (16, 16):
        raise ValueError("prediction_labels must have shape [16,16]")
    if pixels.ndim == 3 and tuple(pixels.shape) == (128, 128, 1):
        pixels = pixels[..., 0].clone()
    if tuple(pixels.shape) != (128, 128):
        raise ValueError("pixel_ground_truth_labels must have shape [128,128] or [128,128,1]")
    if bool((pred < 0).any()) or bool((pixels < 0).any()):
        raise ValueError("predicted and GT instance IDs must be nonnegative")

    majority = clevr_mask_patch(pixels.unsqueeze(0), 8)["patch_labels"][0]
    pixel_np = pixels.numpy()
    allowed_sets = []
    for row in range(16):
        for col in range(16):
            patch = pixel_np[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
            allowed_sets.append(set(int(x) for x in np.unique(patch)))

    flat_pred = pred.reshape(-1).numpy()
    first_index = {}
    for index, label in enumerate(flat_pred.tolist()):
        if label != 0 and label not in first_index:
            first_index[label] = index
    pred_groups = sorted(first_index, key=lambda label: first_index[label])
    gt_ids = sorted(int(x) for x in torch.unique(pixels).tolist() if int(x) != 0)

    weights = np.zeros((len(pred_groups), len(gt_ids)), dtype=np.int64)
    gt_column = {label: index for index, label in enumerate(gt_ids)}
    group_row = {label: index for index, label in enumerate(pred_groups)}
    for patch_index, label in enumerate(flat_pred.tolist()):
        if label == 0:
            continue
        row = group_row[label]
        for gt_id in allowed_sets[patch_index]:
            col = gt_column.get(gt_id)
            if col is not None:
                weights[row, col] += 1

    mapping = {}
    matched_counts = {}
    if weights.size:
        rows, cols = linear_sum_assignment(weights, maximize=True)
        for row, col in zip(rows.tolist(), cols.tolist()):
            label, gt_id = pred_groups[row], gt_ids[col]
            overlap = int(weights[row, col])
            if overlap > 0:
                mapping[label] = gt_id
                matched_counts[label] = overlap

    sentinel = -1
    for label in pred_groups:
        if label not in mapping:
            mapping[label] = sentinel
            matched_counts[label] = 0
            sentinel -= 1

    resolved = torch.zeros((16, 16), dtype=torch.int64)
    allowed_flat = []
    resolved_flat = []
    accepted = 0
    for patch_index, label in enumerate(flat_pred.tolist()):
        allowed = allowed_sets[patch_index]
        candidate = 0 if label == 0 else mapping[label]
        is_accepted = candidate in allowed
        accepted += int(is_accepted)
        target = candidate if is_accepted else int(majority.reshape(-1)[patch_index])
        resolved.reshape(-1)[patch_index] = target
        resolved_flat.append(target)
        allowed_flat.append(allowed)

    if any(target not in allowed for target, allowed in zip(resolved_flat, allowed_flat)):
        raise AssertionError("Resolved target is not present in its patch's allowed-label set")
    resolved_fg = int((resolved != 0).sum())
    score_tensor = patch_fg_ari(pred.unsqueeze(0), resolved.unsqueeze(0))[0]
    score_valid = bool(torch.isfinite(score_tensor))
    score = float(score_tensor) if score_valid else None
    if score_valid and not math.isfinite(score):
        raise FloatingPointError("patch_fg_ari returned a nonfinite valid score")

    mapping_rows = [
        {"predicted_id": int(label), "mapped_gt_id": int(mapping[label]),
         "first_patch_index": int(first_index[label]),
         "overlap_patch_count": int(matched_counts[label])}
        for label in pred_groups
    ]
    return {
        "metric_name": "allowed-label oracle FG-ARI (one-to-one membership matching)",
        "is_standard_metric": False,
        "allowed_label_oracle_fg_ari": score,
        "score_valid": score_valid,
        "all_patch_acceptance_rate": accepted / 256.0,
        "accepted_patch_count": accepted,
        "patch_count": 256,
        "original_majority_foreground_patch_count": int((majority != 0).sum()),
        "resolved_foreground_patch_count": resolved_fg,
        "changed_gt_patch_count": int((resolved != majority).sum()),
        "mapping_counts": mapping_rows,
        "resolved_target_labels": resolved.reshape(-1).tolist(),
    }
