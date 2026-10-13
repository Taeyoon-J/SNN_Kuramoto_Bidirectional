"""Source16-to-native64 model-state mapping used by the resource probe only.

The deployed ``resolution64`` module is the frozen CPU Slot64 scoring helper.
This separate module owns candidate-model construction details so the completed
Slot baseline and scorer fingerprints stay unchanged.
"""
from __future__ import annotations

import numpy as np

from collaborative_test.SW_0154_native64_resolution_compare import resolution64

GRID = resolution64.GRID
NODES = resolution64.NODES
OLD_NODES = resolution64.OLD_NODES
OLD_GRID = 16
NODE_RANKS = {
    "kuramoto.omega": 2,
    "kuramoto.kappa": 2,
    "dendric_layer.tau_n": 2,
    "membrane_layer.tau_m": 1,
}
PAIR_KEYS = {"kuramoto.direction_learner"}


def parent_index_for_grid(grid_size=GRID, *, device=None):
    """Row-major candidate-grid cells to their exact native16 parent cells."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < OLD_GRID or grid_size % OLD_GRID:
        raise ValueError("candidate grid must be an integer multiple of native16")
    scale = grid_size // OLD_GRID
    if device is None:
        rows = np.arange(grid_size, dtype=np.int64) // scale
        cols = np.arange(grid_size, dtype=np.int64) // scale
        return rows[:, None] * OLD_GRID + cols[None, :]
    import torch
    rows = torch.arange(grid_size, device=device) // scale
    cols = torch.arange(grid_size, device=device) // scale
    return rows[:, None] * OLD_GRID + cols[None, :]


def configure_grid_hparams(hp, *, grid_size, graph_top_k):
    """Set the dimension-coupled fields before constructing a candidate core."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < OLD_GRID or grid_size % OLD_GRID:
        raise ValueError("candidate grid must be an integer multiple of native16")
    nodes = grid_size * grid_size
    if isinstance(graph_top_k, bool) or not isinstance(graph_top_k, int) or not 1 <= graph_top_k <= nodes:
        raise ValueError("graph_top_k must be in [1, grid_size**2]")
    hp.num_regions = nodes
    hp.k = float(nodes)
    hp.graph_top_k = graph_top_k
    hp.graph_spatial_grid_size = grid_size
    hp.spike_spatial_grid_size = grid_size
    hp.gamma_patch_grid_size = grid_size
    return hp.validate()


def grid_distance(grid_size=GRID, units_per_cell=0.25, *, device=None, dtype=None):
    """Euclidean row-major grid distances in the registered spatial units."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    if not np.isfinite(units_per_cell) or units_per_cell <= 0:
        raise ValueError("units_per_cell must be finite and positive")
    if device is None:
        rows = np.repeat(np.arange(grid_size, dtype=np.float32), grid_size)
        cols = np.tile(np.arange(grid_size, dtype=np.float32), grid_size)
        dy = (rows[:, None] - rows[None, :]) * float(units_per_cell)
        dx = (cols[:, None] - cols[None, :]) * float(units_per_cell)
        return np.sqrt(dx * dx + dy * dy).astype(dtype or np.float32, copy=False)
    import torch
    rows = torch.arange(grid_size, device=device, dtype=dtype or torch.float32).repeat_interleave(grid_size)
    cols = torch.arange(grid_size, device=device, dtype=dtype or torch.float32).repeat(grid_size)
    return torch.sqrt(((rows[:, None] - rows[None, :]) * units_per_cell).square()
                      + ((cols[:, None] - cols[None, :]) * units_per_cell).square())


def convert_state_dict16_to_grid(source, target_template, *, grid_size=GRID,
                                 units_per_cell=0.25):
    """Strictly map registered spatial state; copy shared tensors unchanged."""
    if not isinstance(source, dict) or not isinstance(target_template, dict):
        raise TypeError("source and target core state dictionaries are required")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < OLD_GRID or grid_size % OLD_GRID:
        raise ValueError("candidate grid must be an integer multiple of native16")
    nodes = grid_size * grid_size
    parent = parent_index_for_grid(grid_size).reshape(-1)
    if set(source) != set(target_template):
        raise ValueError("source/target state keys differ")
    result = {}
    for key, target in target_template.items():
        value = source[key]
        if key == "sc":
            if tuple(target.shape) != (nodes, nodes):
                raise ValueError("candidate structural matrix has an unexpected shape")
            if hasattr(target, "new_zeros"):
                mapped = target.new_zeros((nodes, nodes))
                mapped.fill_diagonal_(1)
            else:
                mapped = np.eye(nodes, dtype=np.asarray(target).dtype)
        elif key == "graph_generator.grid_distance":
            if tuple(target.shape) != (nodes, nodes):
                raise ValueError("candidate distance buffer has an unexpected shape")
            mapped = grid_distance(grid_size, units_per_cell,
                                   device=getattr(target, "device", None),
                                   dtype=getattr(target, "dtype", None))
        elif key in NODE_RANKS:
            if value.ndim != NODE_RANKS[key] or value.shape[0] != OLD_NODES:
                raise ValueError(f"{key} is not a registered native16 node tensor")
            if hasattr(value, "index_select"):
                import torch
                index = torch.as_tensor(parent, device=value.device, dtype=torch.long)
                mapped = value.index_select(0, index)
            else:
                mapped = np.asarray(value)[parent]
        elif key in PAIR_KEYS:
            if tuple(value.shape) != (OLD_NODES, OLD_NODES):
                raise ValueError(f"{key} is not a registered native16 pair tensor")
            if hasattr(value, "index_select"):
                import torch
                index = torch.as_tensor(parent, device=value.device, dtype=torch.long)
                mapped = value.index_select(0, index).index_select(1, index)
            else:
                array = np.asarray(value)
                mapped = array[np.ix_(parent, parent)]
        else:
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"unregistered spatial state change for {key}")
            mapped = value
        if hasattr(target, "to"):
            mapped = mapped.to(device=target.device, dtype=target.dtype).clone()
        else:
            mapped = np.asarray(mapped, dtype=np.asarray(target).dtype).copy()
        if tuple(mapped.shape) != tuple(target.shape):
            raise ValueError(f"mapped state shape mismatch for {key}")
        result[key] = mapped
    return result


def convert_state_dict16_to64(source, target_template):
    """Apply the registered 16-to-64 parent and physical distance mapping."""
    return convert_state_dict16_to_grid(source, target_template,
                                        grid_size=GRID, units_per_cell=0.25)
