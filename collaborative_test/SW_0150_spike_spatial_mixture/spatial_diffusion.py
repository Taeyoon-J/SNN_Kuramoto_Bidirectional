"""Differentiable four-neighbor diffusion of fixed, detached seed identities."""
from __future__ import annotations

import torch

GRID = 32
NODES = GRID * GRID
STEPS = 64
EDGE_EPS = 1e-8


def _checked_seed_rows(seed_ids, batch_size):
    if not isinstance(seed_ids, (list, tuple)) or len(seed_ids) != batch_size:
        raise ValueError("seed_ids must provide one ordered seed list per Q batch row")
    normalized = []
    for row in seed_ids:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 11:
            raise ValueError("each image requires a background seed and at most ten foreground seeds")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("seed IDs must be integer patch indices")
            if value < 0 or value >= NODES:
                raise ValueError("seed ID is outside the fixed 32x32 patch grid")
            values.append(value)
        if len(set(values)) != len(values):
            raise ValueError("seed IDs must be unique within each image")
        normalized.append(values)
    return normalized


def spatial_diffusion(actual_q: torch.Tensor, seed_ids, *, steps: int = STEPS) -> torch.Tensor:
    """Return row-stochastic P[B,1024,Kmax] from actual-spike Q and fixed seeds.

    Seeds are ordered with the selected background medoid first, so active
    assignment column zero is background. Seed selection is external and is
    intentionally nondifferentiable; gradients flow through Q on grid edges.
    A one-seed image is exactly all-background and has no Q credit.
    """
    if not torch.is_tensor(actual_q) or actual_q.ndim != 3:
        raise ValueError("actual_q must have shape [B,1024,1024]")
    if actual_q.shape[0] < 1 or actual_q.shape[1:] != (NODES, NODES):
        raise ValueError("actual_q must use the fixed 32x32 native patch grid")
    if not actual_q.is_floating_point():
        raise TypeError("actual_q must be floating point")
    if not torch.isfinite(actual_q).all():
        raise ValueError("actual_q must be finite")
    if torch.any(actual_q < 0) or torch.any(actual_q > 1):
        raise ValueError("actual_spike Q values must lie in [0,1]")
    if not torch.allclose(actual_q, actual_q.transpose(1, 2), rtol=1e-5, atol=1e-6):
        raise ValueError("actual_spike Q must be symmetric")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != STEPS:
        raise ValueError("registered spatial diffusion uses exactly 64 steps")
    seeds = _checked_seed_rows(seed_ids, actual_q.shape[0])
    batch, _, _ = actual_q.shape
    max_seeds = max(map(len, seeds))
    device, dtype = actual_q.device, actual_q.dtype

    # Static source-to-neighbor indices; invalid edge positions are masked so
    # no grid wraparound or self-loop uses Q's diagonal.
    ids = torch.arange(NODES, device=device).reshape(GRID, GRID)
    neighbor_indices = torch.zeros((NODES, 4), device=device, dtype=torch.long)
    edge_mask = torch.zeros((NODES, 4), device=device, dtype=torch.bool)
    neighbor_grid = (torch.roll(ids, shifts=-1, dims=0),
                     torch.roll(ids, shifts=1, dims=0),
                     torch.roll(ids, shifts=-1, dims=1),
                     torch.roll(ids, shifts=1, dims=1))
    masks_grid = (torch.ones_like(ids, dtype=torch.bool),
                  torch.ones_like(ids, dtype=torch.bool),
                  torch.ones_like(ids, dtype=torch.bool),
                  torch.ones_like(ids, dtype=torch.bool))
    masks_grid[0][-1, :] = False
    masks_grid[1][0, :] = False
    masks_grid[2][:, -1] = False
    masks_grid[3][:, 0] = False
    for direction in range(4):
        neighbor_indices[:, direction] = neighbor_grid[direction].reshape(-1)
        edge_mask[:, direction] = masks_grid[direction].reshape(-1)

    active = torch.zeros((batch, max_seeds), device=device, dtype=dtype)
    seed_rows = torch.zeros((batch, NODES), device=device, dtype=torch.bool)
    seed_onehot = torch.zeros((batch, NODES, max_seeds), device=device, dtype=dtype)
    for b, row in enumerate(seeds):
        k = len(row)
        active[b, :k] = 1
        idx = torch.as_tensor(row, device=device, dtype=torch.long)
        seed_rows[b, idx] = True
        seed_onehot[b, idx, torch.arange(k, device=device)] = 1

    counts = active.sum(dim=1, keepdim=True)
    p = active[:, None, :].expand(batch, NODES, max_seeds) / counts[:, None, :]
    p = torch.where(seed_rows.unsqueeze(-1), seed_onehot, p)
    if max_seeds == 1:
        # Exact registered K=1 behavior; no diffusion or Q gradient is needed.
        return torch.ones((batch, NODES, 1), device=device, dtype=dtype)

    rows = torch.arange(NODES, device=device)[:, None]
    weights = actual_q[:, rows, neighbor_indices]  # [B,N,4], diagonal is not gathered
    weights = (weights + EDGE_EPS) * edge_mask.to(dtype)[None, :, :]
    denominator = 1.0 + weights.sum(dim=-1, keepdim=True)
    flat_neighbors = neighbor_indices.reshape(-1)
    seed_keep = (~seed_rows).unsqueeze(-1)
    for _ in range(STEPS):
        neighbor_p = p.index_select(1, flat_neighbors).reshape(batch, NODES, 4, max_seeds)
        updated = (p + (weights.unsqueeze(-1) * neighbor_p).sum(dim=2)) / denominator
        updated = updated.clamp_min(0)
        row_sum = updated.sum(dim=-1, keepdim=True)
        if not torch.isfinite(row_sum).all() or torch.any(row_sum <= 0):
            raise FloatingPointError("diffusion produced an invalid assignment row")
        p = torch.where(seed_keep, updated / row_sum, seed_onehot)

    # Preserve exact no-credit behavior for one-seed images even when batched
    # beside images with multiple seeds (which take the differentiable path).
    single_seed = counts.squeeze(1) == 1
    exact_background = torch.zeros_like(p)
    exact_background[..., 0] = 1
    p = torch.where(single_seed[:, None, None], exact_background, p)

    if not torch.isfinite(p).all() or torch.any(p < 0):
        raise FloatingPointError("diffusion assignments must be finite and nonnegative")
    if not torch.allclose(p.sum(dim=-1), torch.ones((batch, NODES), device=device, dtype=dtype),
                          rtol=0, atol=1e-6):
        raise FloatingPointError("diffusion assignment rows do not sum to one")
    return p
