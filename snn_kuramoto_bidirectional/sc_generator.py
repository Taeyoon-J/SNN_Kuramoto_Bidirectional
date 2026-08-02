"""Generate image-specific structural connectivity for patch oscillators."""

import math
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor


GridSize: TypeAlias = int | tuple[int, int]


def generate_fixed_connectivity(
    grid_size: GridSize,
    *,
    self_connectivity: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create deterministic distance-decayed patch connectivity ``[N, N]``.

    Patch indices use row-major order. The decay is calibrated so axial
    neighbors have connectivity 1.0 and diagonal neighbors have connectivity
    0.7. ``self_connectivity`` makes the diagonal convention explicit.
    """
    grid_h, grid_w = _parse_grid_size(grid_size)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point dtype.")
    if self_connectivity < 0.0:
        raise ValueError("self_connectivity must be nonnegative.")

    rows = torch.arange(grid_h, device=device, dtype=dtype)
    columns = torch.arange(grid_w, device=device, dtype=dtype)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    coordinates = torch.stack(
        (row_grid.reshape(-1), column_grid.reshape(-1)),
        dim=1,
    )
    distances = torch.cdist(coordinates, coordinates, p=2)

    decay_rate = -math.log(0.7) / (math.sqrt(2.0) - 1.0)
    connectivity = torch.exp(
        -decay_rate * (distances - 1.0).clamp_min(0.0)
    )

    diagonal = torch.eye(
        grid_h * grid_w,
        device=device,
        dtype=torch.bool,
    )
    diagonal_values = torch.full_like(connectivity, float(self_connectivity))
    return torch.where(diagonal, diagonal_values, connectivity)


def generate_image_modulation(
    image: Tensor,
    grid_size: GridSize,
    *,
    sigma_color: float = 0.25,
    m_min: float = 0.5,
) -> Tensor:
    """Create RGB-similarity modulation ``[B, N, N]`` from original images.

    Images are adaptively averaged onto the patch grid. Pairwise patch-color
    distances are converted to Gaussian similarities and rescaled to
    ``[m_min, 1]``.
    """
    if image.ndim != 4:
        raise ValueError("image must have shape [B, 3, H, W].")
    if image.shape[1] != 3:
        raise ValueError("image must contain exactly three RGB channels.")
    grid_h, grid_w = _parse_grid_size(grid_size)
    if sigma_color <= 0.0:
        raise ValueError("sigma_color must be positive.")
    if not 0.0 <= m_min <= 1.0:
        raise ValueError("m_min must be in [0, 1].")

    working_image = (
        image
        if image.is_floating_point()
        else image.to(dtype=torch.get_default_dtype())
    )
    pooled = F.adaptive_avg_pool2d(working_image, (grid_h, grid_w))
    patch_rgb = pooled.permute(0, 2, 3, 1).reshape(image.shape[0], -1, 3)

    color_difference = patch_rgb.unsqueeze(2) - patch_rgb.unsqueeze(1)
    squared_distance = color_difference.square().sum(dim=-1)
    similarity = torch.exp(
        -squared_distance / (2.0 * float(sigma_color) ** 2)
    )
    similarity = 0.5 * (similarity + similarity.transpose(1, 2))
    return float(m_min) + (1.0 - float(m_min)) * similarity


def generate_sc(
    image: Tensor,
    grid_size: GridSize,
    *,
    sigma_color: float = 0.25,
    m_min: float = 0.5,
    self_connectivity: float = 0.0,
) -> Tensor:
    """Generate image-specific patch connectivity ``[B, N, N]``.

    The result is the element-wise product of fixed spatial connectivity and
    original-image RGB modulation.
    """
    modulation = generate_image_modulation(
        image,
        grid_size,
        sigma_color=sigma_color,
        m_min=m_min,
    )
    fixed_connectivity = generate_fixed_connectivity(
        grid_size,
        self_connectivity=self_connectivity,
        device=modulation.device,
        dtype=modulation.dtype,
    )
    return fixed_connectivity.unsqueeze(0) * modulation


def _parse_grid_size(grid_size: GridSize) -> tuple[int, int]:
    if isinstance(grid_size, int):
        grid_h = grid_w = int(grid_size)
    elif isinstance(grid_size, tuple) and len(grid_size) == 2:
        grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    else:
        raise ValueError("grid_size must be a positive int or a pair of ints.")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive.")
    return grid_h, grid_w
