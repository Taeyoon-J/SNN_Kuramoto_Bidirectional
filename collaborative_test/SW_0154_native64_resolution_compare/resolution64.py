"""Strict CPU helpers for mapping native16 node parameters to a native64 grid.

This module deliberately does not construct a 4096-node model. Physical constants
and graph settings are a separate protocol decision; only the registered spatial
parent mapping and 2x2 pixel modal conversion are implemented here.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

OLD_GRID = 16
GRID = 64
NODES = GRID * GRID
OLD_NODES = OLD_GRID * OLD_GRID
UPSCALE = GRID // OLD_GRID


def parent_index(*, device=None):
    """Return row-major64 -> row-major16 parent indices (16 children per parent)."""
    if device is None:
        rows = np.arange(GRID, dtype=np.int64) // UPSCALE
        cols = np.arange(GRID, dtype=np.int64) // UPSCALE
        return rows[:, None] * OLD_GRID + cols[None, :]
    import torch
    rows = torch.arange(GRID, device=device) // UPSCALE
    cols = torch.arange(GRID, device=device) // UPSCALE
    return rows[:, None] * OLD_GRID + cols[None, :]


def map_node_parameter(value, *, name="node parameter"):
    """Expand a native16 node-leading tensor by exact nearest parent replication."""
    if not hasattr(value, "shape") or len(value.shape) < 1 or value.shape[0] != OLD_NODES:
        raise ValueError(f"{name} must have leading dimension {OLD_NODES}")
    index = parent_index(device=value.device) if hasattr(value, "device") else parent_index().reshape(-1)
    if hasattr(value, "index_select"):
        mapped = value.index_select(0, index.to(value.device).reshape(-1))
    else:
        mapped = np.asarray(value)[index.reshape(-1)]
    if mapped.shape[0] != NODES:
        raise AssertionError("native64 node mapping emitted the wrong node count")
    return mapped


def map_pair_parameter(value, *, name="pair parameter"):
    """Expand a native16 [256,256] matrix using the same parent on both axes."""
    if tuple(value.shape) != (OLD_NODES, OLD_NODES):
        raise ValueError(f"{name} must have shape [{OLD_NODES},{OLD_NODES}]")
    index = parent_index(device=value.device).reshape(-1) if hasattr(value, "device") else parent_index().reshape(-1)
    if hasattr(value, "index_select"):
        mapped = value.index_select(0, index.to(value.device)).index_select(1, index.to(value.device))
    else:
        array = np.asarray(value)
        mapped = array[np.ix_(index, index)]
    if tuple(mapped.shape) != (NODES, NODES):
        raise AssertionError("native64 pair mapping emitted the wrong shape")
    return mapped


def modal_native64(pixel_labels):
    """Direct modal 2x2 pooling from fixed320 native128 integer labels to64."""
    repo = Path(__file__).resolve().parents[2]
    for path in (repo, repo / "collaborative_test"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from collaborative_test.SW_0135_native32_spike_binding.evaluation_contract32 import (
        COUNT, IMAGE_SIZE,
    )
    from snn_kuramoto_bidirectional.evaluation import clevr_mask_patch
    labels = np.asarray(pixel_labels)
    if labels.shape == (COUNT, IMAGE_SIZE, IMAGE_SIZE, 1):
        labels = labels[..., 0]
    if labels.shape != (COUNT, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("native pixel labels must have fixed shape [320,128,128]")
    if (not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_)
            or np.any(labels < 0)):
        raise ValueError("native pixel labels must be nonnegative integers")
    pooled = clevr_mask_patch(__import__("torch").from_numpy(np.ascontiguousarray(labels)), 2)["patch_labels"]
    result = pooled.cpu().numpy()
    if result.shape != (COUNT, GRID, GRID):
        raise AssertionError("production modal converter did not emit [320,64,64]")
    return result.astype(np.int64, copy=False)

