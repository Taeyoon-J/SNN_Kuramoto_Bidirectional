"""Temporary top-k recomputation that restores the trained graph module."""

import torch


def adjacency_at_top_k(graph_generator, gamma, top_k):
    """Recompute learned adjacency changing only top_k; always restore original."""
    original = int(graph_generator.top_k)
    top_k = int(top_k)
    if top_k < 1 or top_k > gamma.shape[-1]:
        raise ValueError("top_k must lie in [1, number of graph nodes]")
    try:
        graph_generator.top_k = top_k
        adjacency = graph_generator(gamma)
    finally:
        graph_generator.top_k = original
    if int(graph_generator.top_k) != original:
        raise AssertionError("graph generator top_k was not restored")
    expected = (gamma.shape[0], gamma.shape[-1], gamma.shape[-1])
    if tuple(adjacency.shape) != expected:
        raise ValueError(f"graph output must have shape {expected}, got {tuple(adjacency.shape)}")
    if not torch.isfinite(adjacency).all() or torch.any(adjacency < 0):
        raise FloatingPointError("learned graph adjacency must be finite and nonnegative")
    if not torch.allclose(adjacency, adjacency.transpose(1, 2), atol=1e-6, rtol=1e-6):
        raise ValueError("learned graph adjacency must be symmetric")
    return adjacency
