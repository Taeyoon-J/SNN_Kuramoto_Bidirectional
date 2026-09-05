"""Hierarchical object-mask classification from multi-level spike histories."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_LEVEL_GRID_SIZES = (
    (16, 16),
    (8, 8),
    (4, 4),
    (2, 2),
)


def classify_hierarchical_spikes(
    spike_levels: list[Tensor] | tuple[Tensor, ...],
    *,
    spike_threshold: float = 0.0,
    output_size: int | tuple[int, int] = 128,
    level_grid_sizes: tuple[tuple[int, int], ...] = DEFAULT_LEVEL_GRID_SIZES,
) -> tuple[Tensor, Tensor]:
    """Create one hierarchical object mask per time step.

    Args:
        spike_levels:
            Fine-to-coarse spike histories shaped ``[B, N_level, T]`` for
            Level 1 through Level 4.
        spike_threshold:
            A spike is active when its absolute value is greater than this
            threshold. The default treats every nonzero spike as active.
        output_size:
            Spatial size of the returned image masks.
        level_grid_sizes:
            Fine-to-coarse patch-grid shapes corresponding to spike_levels.

    Returns:
        A pair ``(object_masks, valid_objects)``. ``object_masks`` has shape
        ``[B, T, 1, output_height, output_width]``. All active roots at the
        same time step are merged into that time step's object. A finest-level
        patch survives only when an active parent-to-child path exists through
        every coarser level. ``valid_objects`` has shape ``[B, T]`` and marks
        masks containing at least one surviving finest-level patch.
    """
    if len(spike_levels) != len(level_grid_sizes):
        raise ValueError(
            f"Expected {len(level_grid_sizes)} spike levels, "
            f"but received {len(spike_levels)}."
        )

    output_height, output_width = _pair(output_size, "output_size")
    normalized_grids = tuple(
        _pair(grid_size, "level_grid_size")
        for grid_size in level_grid_sizes
    )

    batch_size = None
    num_steps = None
    level_activity = []

    for level_index, (spikes, grid_size) in enumerate(
        zip(spike_levels, normalized_grids),
        start=1,
    ):
        if spikes.dim() != 3:
            raise ValueError(
                f"Level {level_index} spikes must have shape [B, N, T]."
            )

        current_batch, num_oscillators, current_steps = spikes.shape
        expected_oscillators = grid_size[0] * grid_size[1]
        if num_oscillators != expected_oscillators:
            raise ValueError(
                f"Level {level_index} contains {num_oscillators} oscillators, "
                f"but grid {grid_size} requires {expected_oscillators}."
            )

        if batch_size is None:
            batch_size = current_batch
            num_steps = current_steps
        elif current_batch != batch_size or current_steps != num_steps:
            raise ValueError(
                "All spike levels must have the same batch size and number "
                "of time steps."
            )

        active = spikes.abs() > float(spike_threshold)
        active = active.permute(0, 2, 1).reshape(
            current_batch,
            current_steps,
            grid_size[0],
            grid_size[1],
        )
        level_activity.append(active)

    reachable = level_activity[-1]
    for fine_level_index in range(len(level_activity) - 2, -1, -1):
        fine_activity = level_activity[fine_level_index]
        fine_grid = normalized_grids[fine_level_index]
        expanded_parent = _resize_boolean_grid(reachable, fine_grid)
        reachable = fine_activity & expanded_parent

    valid_objects = reachable.flatten(start_dim=2).any(dim=2)
    object_masks = _resize_boolean_grid(
        reachable,
        (output_height, output_width),
    )
    object_masks = object_masks.unsqueeze(2).to(dtype=torch.float32)
    return object_masks, valid_objects


def _resize_boolean_grid(mask: Tensor, output_size: tuple[int, int]) -> Tensor:
    batch_size, num_steps, height, width = mask.shape
    resized = F.interpolate(
        mask.reshape(batch_size * num_steps, 1, height, width).float(),
        size=output_size,
        mode="nearest",
    )
    return resized.reshape(
        batch_size,
        num_steps,
        output_size[0],
        output_size[1],
    ).bool()


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} values must be positive.")
        return height, width
    raise ValueError(f"{name} must be an int or a pair of ints.")
