"""Cross-view partition alignment and source area-mass anchor."""
from __future__ import annotations

import torch

GRID = 32
NODES = GRID * GRID
MAX_SLOTS = 11
EPS = 1e-8


def horizontal_flip_indices(indices: torch.Tensor) -> torch.Tensor:
    """Map canonical flattened patch indices into a horizontally flipped view."""
    if not torch.is_tensor(indices) or indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be an integer tensor")
    if indices.numel() and ((indices < 0).any() or (indices >= NODES).any()):
        raise ValueError("seed index outside the 32x32 grid")
    row = torch.div(indices, GRID, rounding_mode="floor")
    col = indices % GRID
    return row * GRID + (GRID - 1 - col)


def unflip_probabilities(p: torch.Tensor) -> torch.Tensor:
    if p.ndim != 3 or p.shape[1] != NODES:
        raise ValueError("P must have shape [B,1024,K]")
    return p.reshape(p.shape[0], GRID, GRID, p.shape[2]).flip(2).reshape_as(p)


def _validate_probabilities(name: str, p: torch.Tensor) -> None:
    if not torch.is_tensor(p) or p.ndim != 3 or p.shape[1] != NODES:
        raise ValueError(f"{name} must have shape [B,1024,K]")
    if p.shape[0] < 1 or not 1 <= p.shape[2] <= MAX_SLOTS:
        raise ValueError(f"{name} batch/slot dimensions are invalid")
    if not p.is_floating_point() or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    sums = p.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} rows must sum to one")


def _pad_slots(mass: torch.Tensor) -> torch.Tensor:
    if mass.shape[-1] > MAX_SLOTS:
        raise ValueError("mass distribution exceeds registered 11 slots")
    return torch.nn.functional.pad(mass, (0, MAX_SLOTS - mass.shape[-1]))


def _mass_kl(source_mass: torch.Tensor, current_p: torch.Tensor) -> torch.Tensor:
    if source_mass.ndim == 3:
        source_mass = source_mass.mean(dim=1)
    if (source_mass.ndim != 2 or source_mass.shape[0] != current_p.shape[0]
            or not 1 <= source_mass.shape[-1] <= MAX_SLOTS):
        raise ValueError("frozen source mass must have shape [B,K] or assignments [B,1024,K]")
    if (not source_mass.is_floating_point() or not torch.isfinite(source_mass).all()
            or (source_mass < 0).any()):
        raise ValueError("frozen source masses must be finite and nonnegative")
    source_mass = source_mass.detach()
    if not torch.allclose(source_mass.sum(-1), torch.ones_like(source_mass.sum(-1)), rtol=1e-5, atol=1e-6):
        raise ValueError("frozen source masses must sum to one")
    # Sorting makes this term invariant to arbitrary slot names; JS still
    # preserves the registered seed correspondence across the two views.
    source = _pad_slots(source_mass.sort(dim=-1, descending=True).values)
    current = current_p.mean(dim=1).sort(dim=-1, descending=True).values
    current = _pad_slots(current)
    source = source / source.sum(dim=-1, keepdim=True).clamp_min(EPS)
    current = current / current.sum(dim=-1, keepdim=True).clamp_min(EPS)
    terms = torch.where(source > 0, source * (source.clamp_min(EPS).log() - current.clamp_min(EPS).log()), 0.0)
    return terms.sum(dim=-1).mean()


def alignment_js(canonical: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
    """Mean patchwise JS after mapping transformed assignments back to canonical."""
    _validate_probabilities("canonical P", canonical)
    _validate_probabilities("transformed P", transformed)
    if canonical.shape != transformed.shape:
        raise ValueError("paired views must have matching batch, nodes, and slots")
    q = unflip_probabilities(transformed)
    midpoint = (canonical + q) * 0.5
    left = torch.where(canonical > 0, canonical * (canonical.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log()), 0.0)
    right = torch.where(q > 0, q * (q.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log()), 0.0)
    return (0.5 * (left + right).sum(dim=-1)).mean()


def partition_objective(canonical: torch.Tensor, transformed: torch.Tensor,
                        source_p: torch.Tensor, *, mode: str = "candidate") -> dict[str, torch.Tensor]:
    """Compute candidate (JS + mass KL) or matched marginal-only control."""
    _validate_probabilities("canonical P", canonical)
    _validate_probabilities("transformed P", transformed)
    if source_p.ndim == 3:
        _validate_probabilities("frozen source P", source_p)
        if canonical.shape[:2] != source_p.shape[:2]:
            raise ValueError("source and paired-view batch/node dimensions must match")
    elif source_p.ndim == 2:
        if source_p.shape[0] != canonical.shape[0]:
            raise ValueError("source mass and paired views must have matching batch dimensions")
    else:
        raise ValueError("frozen source must be [B,K] mass or [B,1024,K] assignments")
    if canonical.shape != transformed.shape:
        raise ValueError("paired views must have matching batch, nodes, and slots")
    if canonical.device != transformed.device or canonical.device != source_p.device:
        raise ValueError("all assignment tensors must share a device")
    align = alignment_js(canonical, transformed)
    mass = 0.5 * (_mass_kl(source_p, canonical) + _mass_kl(source_p, transformed))
    if mode == "candidate":
        total = align + mass
    elif mode == "control":
        total = mass
    else:
        raise ValueError("mode must be 'candidate' or 'control'")
    return {"total": total, "alignment_js": align, "mass_kl": mass}
