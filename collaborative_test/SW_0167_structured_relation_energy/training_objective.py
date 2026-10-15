"""SW0167 training objective with the registered gold-partition tie rule."""
from __future__ import annotations

import numpy as np
import torch

from collaborative_test.SW_0167_structured_relation_energy import structured_energy


def structured_hinge_gold_ties(e_theta, e0, target, support):
    """Return the structured hinge, selecting G* when the oracle score ties.

    This keeps SW0167's existing unordered-pair weights, detached source
    normalization, full-grid loss-augmented energy, and original-coordinate
    four-connected solver. At exact score ties, the registered gold partition
    wins and contributes a live zero loss/gradient.
    """
    if (not torch.is_tensor(e_theta) or not torch.is_tensor(e0)
            or e_theta.ndim != 2 or e_theta.shape != e0.shape
            or e_theta.shape[0] != e_theta.shape[1]):
        raise ValueError("candidate and source energies must be same-size square tensors")
    support_tensor = torch.as_tensor(support, dtype=torch.bool, device=e_theta.device)
    target_tensor = torch.as_tensor(target, device=e_theta.device)
    if (support_tensor.ndim != 2 or target_tensor.shape != support_tensor.shape
            or support_tensor.numel() != e_theta.shape[0]):
        raise ValueError("target/support must match the energy's original grid")
    if not torch.isfinite(e_theta).all() or not torch.isfinite(e0).all():
        raise ValueError("candidate/source energies must be finite")

    same, weights, idx = structured_energy.pair_targets(target_tensor, support_tensor)
    if idx.numel() < 2 or not bool(weights.any()):
        return e_theta.sum() * 0.0
    idx = idx.to(device=e_theta.device)
    et = e_theta.index_select(0, idx).index_select(1, idx)
    ebase = e0.to(device=e_theta.device).index_select(0, idx).index_select(1, idx)
    upper = torch.triu(torch.ones_like(ebase, dtype=torch.bool), diagonal=1)
    z = ebase.detach()[upper].abs().sum().clamp_min(1.0)
    y = same.to(device=e_theta.device)
    pair_weight = weights.to(device=e_theta.device, dtype=e_theta.dtype)

    support_cpu = support_tensor.detach().cpu().numpy().astype(bool, copy=False)
    n = e_theta.shape[0]
    full_augmented = torch.zeros((n, n), device=e_theta.device, dtype=e_theta.dtype)
    aug_upper = et.detach()[upper] + z * pair_weight[upper] * (1.0 - 2.0 * y[upper])
    compact_aug = torch.zeros_like(et.detach())
    compact_aug[upper] = aug_upper
    compact_aug = compact_aug + compact_aug.T
    full_augmented[idx[:, None], idx[None, :]] = compact_aug
    predicted = structured_energy.solve_energy(
        full_augmented.detach().cpu().numpy(), support_cpu.astype(np.int32),
        grid_shape=tuple(int(x) for x in support_cpu.shape))["labels"].reshape(-1)
    predicted_on_support = predicted[idx.detach().cpu().numpy()]
    y_hat = torch.as_tensor(
        predicted_on_support[:, None] == predicted_on_support[None, :],
        device=e_theta.device, dtype=e_theta.dtype)
    score = (et[upper] * y_hat[upper]).sum() / z
    distance = (pair_weight[upper] * (y_hat[upper] - y[upper]).abs()).sum()
    gold_score = (et[upper] * y[upper]).sum() / z
    augmented_score = score + distance
    if float(augmented_score.detach()) > float(gold_score.detach()):
        return augmented_score - gold_score
    return e_theta.sum() * 0.0
