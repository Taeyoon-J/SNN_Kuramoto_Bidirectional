"""GT-last fixed-support FG-ARI bound helpers.

This module does not read a dataset. Callers must supply already-authorized
target labels and a frozen predicted-foreground support mask.
"""
from __future__ import annotations

import numpy as np


def adjusted_rand_from_labels(target: np.ndarray, prediction: np.ndarray) -> float:
    """Match the production ARI convention for one nonempty 1-D population."""
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape or target.size < 2:
        return float("nan")
    _, ti = np.unique(target, return_inverse=True)
    _, pi = np.unique(prediction, return_inverse=True)
    counts = np.bincount(
        ti * (pi.max() + 1) + pi,
        minlength=(ti.max() + 1) * (pi.max() + 1),
    ).reshape(ti.max() + 1, pi.max() + 1).astype(np.float64)
    rows, cols = counts.sum(1), counts.sum(0)
    choose2 = lambda x: x * (x - 1.0) / 2.0
    a = choose2(counts).sum()
    c = choose2(rows).sum()
    b = choose2(cols).sum()
    total = choose2(float(target.size))
    expected = b * c / total
    denominator = 0.5 * (b + c) - expected
    return float((a - expected) / denominator) if denominator != 0 else 1.0


def fixed_support_ari_upper_bound(
    target_labels: np.ndarray,
    predicted_foreground: np.ndarray,
    *,
    background_id: int = 0,
) -> float:
    """Certified FG-ARI upper bound with fixed zero/nonzero membership.

    The relaxation permits arbitrarily many, possibly disconnected, positive
    groups. The population is target foreground, exactly as production FG-ARI.
    Returns NaN when production FG-ARI is undefined (fewer than two patches).
    """
    target_labels = np.asarray(target_labels)
    predicted_foreground = np.asarray(predicted_foreground, dtype=bool)
    if target_labels.shape != predicted_foreground.shape:
        raise ValueError("target and support shapes differ")
    target_fg = target_labels != background_id
    target = target_labels[target_fg]
    if target.size < 2:
        return float("nan")
    support = predicted_foreground[target_fg]
    _, gt_inverse = np.unique(target, return_inverse=True)
    oracle = np.zeros(target.size, dtype=np.int64)
    oracle[support] = gt_inverse[support] + 1
    return adjusted_rand_from_labels(gt_inverse, oracle)
