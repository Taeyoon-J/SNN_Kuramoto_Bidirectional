"""Fixed label-free RGB bilateral adjacency and learned-graph blending."""

import math

import torch


def rgb_bilateral_graph(images, learned_graph_generator, grid_size=16,
                        color_sigma=2.0, spatial_sigma=2.0, scale_floor=4.0):
    """Build symmetric top-k RGB/spatial graph at learned graph's coupling scale.

    Images are RGB `[B,3,H,W]` in 0..255 units. Per-image robust normalization
    uses patch-mean channel medians and 1.4826*MAD with a fixed intensity floor.
    Bilateral affinity is `exp(-d_color^2/(2*sigma_c^2)) *
    exp(-d_grid^2/(2*sigma_s^2))`; row-normalized top-k weights use the trained
    graph generator's `top_k` and `exp(log_coupling_gain)`.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B,3,H,W]")
    if not hasattr(learned_graph_generator, "top_k") or not hasattr(learned_graph_generator, "log_coupling_gain"):
        raise ValueError("learned graph generator must expose top_k and log_coupling_gain")
    for name, value in (("color_sigma", color_sigma), ("spatial_sigma", spatial_sigma),
                        ("scale_floor", scale_floor)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    height, width = images.shape[-2:]
    if grid_size < 2 or height % grid_size or width % grid_size:
        raise ValueError("image dimensions must divide evenly into grid_size >= 2")
    patch_h, patch_w = height // grid_size, width // grid_size
    patch_rgb = torch.nn.functional.avg_pool2d(
        images.float(), (patch_h, patch_w), (patch_h, patch_w)
    ).permute(0, 2, 3, 1).reshape(images.shape[0], grid_size * grid_size, 3)
    median = patch_rgb.median(dim=1, keepdim=True).values
    mad = (patch_rgb - median).abs().median(dim=1, keepdim=True).values
    robust_scale = (1.4826 * mad).clamp_min(float(scale_floor))
    normalized = ((patch_rgb - median) / robust_scale).clamp(-8.0, 8.0)
    color_d2 = torch.cdist(normalized, normalized).square()

    coords = torch.stack(torch.meshgrid(
        torch.arange(grid_size, device=images.device, dtype=torch.float32),
        torch.arange(grid_size, device=images.device, dtype=torch.float32),
        indexing="ij",
    ), dim=-1).reshape(-1, 2)
    grid_d2 = torch.cdist(coords, coords).square().unsqueeze(0)
    affinity = torch.exp(-color_d2 / (2.0 * float(color_sigma) ** 2))
    affinity = affinity * torch.exp(-grid_d2 / (2.0 * float(spatial_sigma) ** 2))

    num_nodes = grid_size * grid_size
    k = min(int(learned_graph_generator.top_k), num_nodes)
    if k < 1:
        raise ValueError("learned graph top_k must be positive")
    values, indices = affinity.topk(k, dim=-1)
    weights = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    gain = learned_graph_generator.log_coupling_gain.detach().exp().to(
        device=images.device, dtype=affinity.dtype
    )
    weights = weights * gain
    adjacency = torch.zeros_like(affinity).scatter_(-1, indices, weights)
    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    if not torch.isfinite(adjacency).all() or torch.any(adjacency < 0):
        raise FloatingPointError("RGB graph contains invalid weights")
    return adjacency


def blend_adjacencies(learned, rgb, alpha):
    """Convex blend; alpha 0 returns the exact learned tensor unchanged."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0,1]")
    if learned.shape != rgb.shape or learned.ndim != 3 or learned.shape[-1] != learned.shape[-2]:
        raise ValueError("learned and RGB graphs must share [B,N,N] shape")
    if alpha == 0.0:
        return learned
    if alpha == 1.0:
        return rgb
    return (1.0 - alpha) * learned + alpha * rgb
