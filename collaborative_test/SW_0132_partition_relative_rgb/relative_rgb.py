"""Partition-relative full-resolution RGB decoder for SW0132.

The hard forward assignment is the SW0106 H while its soft assignment retains
the registered straight-through gradient. Geometry and pooled content remain
live with respect to W; only gamma content is detached.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import (
    assignment_weights,
)

GRID = 16
PATCH = 8
IMAGE = GRID * PATCH
FEATURES = 8
CELL_VARIANCE = ((PATCH ** 2 - 1) / 12.0) * (2.0 / IMAGE) ** 2
PIXEL_CHUNK = 1024


class RelativeRGBDecoder(nn.Module):
    """SW0106-sized shared MLP receiving 8D content and 2D relative position."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid(),
        )

    def forward(self, content, relative_xy):
        if content.ndim != 2 or content.shape[-1] != FEATURES:
            raise ValueError("content must have shape [K,8]")
        if relative_xy.ndim != 3 or relative_xy.shape[0] != content.shape[0] \
                or relative_xy.shape[-1] != 2:
            raise ValueError("relative coordinates must have shape [K,P,2]")
        k, pixels, _ = relative_xy.shape
        expanded = content[:, None, :].expand(k, pixels, FEATURES)
        return self.net(torch.cat((expanded, relative_xy), dim=-1))


def patch_centers(device, dtype):
    coordinate = (torch.arange(GRID, device=device, dtype=dtype) + 0.5) * (2.0 / GRID) - 1.0
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def pixel_centers(device, dtype):
    coordinate = (torch.arange(IMAGE, device=device, dtype=dtype) + 0.5) * (2.0 / IMAGE) - 1.0
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def patch_pixel_weights(weights):
    """Repeat row-major 16x16 patch weights over each native 8x8 pixel cell."""
    if weights.ndim != 2 or weights.shape[0] != GRID * GRID:
        raise ValueError("patch weights must have shape [256,K]")
    k = weights.shape[1]
    return weights.reshape(GRID, GRID, k).repeat_interleave(PATCH, dim=0) \
        .repeat_interleave(PATCH, dim=1).reshape(IMAGE * IMAGE, k)


def partition_geometry(weights, device=None, dtype=None):
    """Return live slot mass, centroid and footprint-corrected standard deviation."""
    if weights.ndim != 2 or weights.shape[0] != GRID * GRID:
        raise ValueError("weights must have shape [256,K]")
    device = weights.device if device is None else device
    dtype = weights.dtype if dtype is None else dtype
    centers = patch_centers(device, dtype)
    w = weights.to(device=device, dtype=dtype)
    raw_mass = w.sum(dim=0)
    mass = raw_mass.clamp_min(1.0)
    centroid = (w.transpose(0, 1) @ centers) / mass[:, None]
    delta = centers[:, None, :] - centroid[None, :, :]
    variance = (w[:, :, None] * delta.square()).sum(dim=0) / mass[:, None]
    variance = variance + CELL_VARIANCE
    return raw_mass, centroid, variance.sqrt()


def assignment_to_weights(q, hard, temperature=0.10):
    """Use the frozen SW0106 hard-H straight-through assignment implementation."""
    return assignment_weights(q, hard, credit=True, temperature=temperature)


def _render_chunk(pixel_weights, content, centroid, scale, xy, decoder):
    relative = (xy[:, None, :] - centroid[None, :, :]) / scale[None, :, :]
    relative = relative.transpose(0, 1)  # [K,P,2]
    slot_rgb = decoder(content, relative)
    # Each pixel's repeated W is the sole mixture weight; there is no alpha head.
    return torch.einsum("pk,kpc->pc", pixel_weights, slot_rgb)


def render_partition(weights, gamma, decoder, *, chunk_size=PIXEL_CHUNK,
                     checkpoint_chunks=True):
    """Render [128,128,3] RGB using live assignment geometry and detached gamma.

    `weights` is [256,K], `gamma` is [256,8]. Checkpointing wraps the complete
    decoder-and-mixture chunk so intermediate slot RGB activations are released.
    """
    if gamma.shape != (GRID * GRID, FEATURES):
        raise ValueError("gamma must have shape [256,8]")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    raw_mass, centroid, scale = partition_geometry(weights)
    content = (weights.transpose(0, 1) @ gamma.detach()) / raw_mass.clamp_min(1.0)[:, None]
    pixel_w = patch_pixel_weights(weights)
    xy = pixel_centers(weights.device, weights.dtype)
    pieces = []
    for start in range(0, IMAGE * IMAGE, chunk_size):
        stop = min(start + chunk_size, IMAGE * IMAGE)
        args = (pixel_w[start:stop], content, centroid, scale, xy[start:stop])
        if checkpoint_chunks and torch.is_grad_enabled():
            rendered = checkpoint(
                lambda w, z, mu, sigma, pos: _render_chunk(w, z, mu, sigma, pos, decoder),
                *args, use_reentrant=False,
            )
        else:
            rendered = _render_chunk(*args, decoder)
        pieces.append(rendered)
    return torch.cat(pieces, dim=0).reshape(IMAGE, IMAGE, 3)


def reconstruct_one(q, hard, gamma, target_rgb, decoder, *, chunk_size=PIXEL_CHUNK,
                    checkpoint_chunks=True):
    """Return full-resolution prediction, scalar image MSE and assignment details."""
    if target_rgb.shape != (IMAGE, IMAGE, 3):
        raise ValueError("target RGB must have shape [128,128,3]")
    weights, soft = assignment_to_weights(q, hard)
    prediction = render_partition(weights, gamma, decoder, chunk_size=chunk_size,
                                  checkpoint_chunks=checkpoint_chunks)
    loss = F.mse_loss(prediction, target_rgb)
    if not torch.equal(weights.detach(), hard.detach()):
        raise AssertionError("straight-through forward assignment differs from hard H")
    return prediction, loss, {
        "W": weights, "P": soft, "K": int(hard.shape[1]),
        "group_sizes": hard.sum(dim=0).detach(),
        "slot_mass": weights.sum(dim=0),
    }


def reconstruct_batch(q_batch, hard_rows, gamma_batch, target_batch, decoder, *,
                      chunk_size=PIXEL_CHUNK, checkpoint_chunks=True):
    if q_batch.ndim != 3 or gamma_batch.shape != (q_batch.shape[0], 256, FEATURES):
        raise ValueError("expected batched Q [B,256,256] and gamma [B,256,8]")
    if len(hard_rows) != q_batch.shape[0] or target_batch.shape != (q_batch.shape[0], IMAGE, IMAGE, 3):
        raise ValueError("hard partitions or RGB target batch has the wrong shape")
    results = [reconstruct_one(q_batch[i], hard_rows[i], gamma_batch[i], target_batch[i], decoder,
                               chunk_size=chunk_size, checkpoint_chunks=checkpoint_chunks)
               for i in range(q_batch.shape[0])]
    predictions = torch.stack([row[0] for row in results])
    losses = torch.stack([row[1] for row in results])
    return predictions, losses, [row[2] for row in results]
