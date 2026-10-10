"""Native-spike Slot Attention binder and relative-coordinate RGB renderer.

This module is an isolated model component for the prospective SW0134 design.
It does not alter the production core, readout, or any training runner.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

GRID = 16
PATCH = 8
IMAGE = GRID * PATCH
PATCHES = GRID * GRID
SLOTS = 11
DIM = 64
SPIKE_COMPONENTS = 4
SETTLED_STEPS = 512
PIXEL_CHUNK = 1024
CELL_VARIANCE = ((PATCH ** 2 - 1) / 12.0) * (2.0 / IMAGE) ** 2


class NativeSpikeSlotBinder(nn.Module):
    """Exchangeable 11-slot binding from raw [B,4,256,512] emitted spikes."""

    def __init__(self, *, seed=134):
        super().__init__()
        # A private RNG makes all arm instances start identically without
        # changing the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.input_projection = nn.Linear(SPIKE_COMPONENTS * SETTLED_STEPS, DIM)
            self.input_norm = nn.LayerNorm(DIM)
            self.key = nn.Linear(DIM, DIM, bias=False)
            self.query = nn.Linear(DIM, DIM, bias=False)
            self.value = nn.Linear(DIM, DIM, bias=False)
            self.gru = nn.GRUCell(DIM, DIM)
            self.slot_mlp = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU(),
                                          nn.Linear(128, DIM))
            # Shared across the exchangeable slots: only the fixed noise gives
            # slots distinct initial values; there are no per-slot categories.
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, DIM))
            self.slot_log_scale = nn.Parameter(torch.zeros(1, 1, DIM))
            noise = torch.randn(1, SLOTS, DIM)
        self.register_buffer("initial_noise", noise, persistent=True)

    def initial_slots(self, batch_size):
        return (self.slot_mu + self.slot_log_scale.exp() * self.initial_noise).expand(
            int(batch_size), -1, -1)

    def forward(self, actual_spikes, *, initial_slots=None):
        if actual_spikes.ndim != 4 or tuple(actual_spikes.shape[1:]) != (
                SPIKE_COMPONENTS, PATCHES, SETTLED_STEPS):
            raise ValueError("SW0134 binder input must be actual spikes [B,4,256,512]")
        if not torch.is_floating_point(actual_spikes) or not torch.isfinite(actual_spikes).all():
            raise ValueError("actual spike traces must be finite floating-point values")
        batch = actual_spikes.shape[0]
        patch = actual_spikes.permute(0, 2, 1, 3).reshape(batch, PATCHES, -1)
        patch = self.input_norm(self.input_projection(patch))
        slots = self.initial_slots(batch) if initial_slots is None else initial_slots
        if slots.shape != (batch, SLOTS, DIM):
            raise ValueError("initial slot state must have shape [B,11,64]")
        if slots.device != patch.device or slots.dtype != patch.dtype:
            raise ValueError("initial slots must match spike feature device and dtype")

        key = self.key(patch)
        value = self.value(patch)
        scale = math.sqrt(DIM)
        for _ in range(3):
            logits = torch.einsum("bnd,bkd->bnk", key, self.query(slots)) / scale
            patch_to_slot = torch.softmax(logits, dim=-1)
            slot_to_patch = patch_to_slot / patch_to_slot.sum(dim=1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bnk,bnd->bkd", slot_to_patch, value)
            slots = self.gru(updates.reshape(-1, DIM), slots.reshape(-1, DIM)).reshape(
                batch, SLOTS, DIM)
            slots = slots + self.slot_mlp(slots)

        final_logits = torch.einsum("bnd,bkd->bnk", key, self.query(slots)) / scale
        assignments = torch.softmax(final_logits, dim=-1)
        if not torch.isfinite(slots).all() or not torch.isfinite(assignments).all():
            raise FloatingPointError("nonfinite SW0134 binder output")
        return assignments, slots, patch


class RelativeSlotRGBDecoder(nn.Module):
    """Shared 66→64→64→3 slot RGB decoder; assignment P is the only mask mixture."""

    def __init__(self, *, seed=106):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.net = nn.Sequential(
                nn.Linear(DIM + 2, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 3), nn.Sigmoid(),
            )

    def forward(self, slot_latents, relative_xy):
        if slot_latents.ndim != 3 or slot_latents.shape[1:] != (SLOTS, DIM):
            raise ValueError("slot latents must have shape [B,11,64]")
        if relative_xy.ndim != 4 or relative_xy.shape[0] != slot_latents.shape[0] \
                or relative_xy.shape[1] != SLOTS or relative_xy.shape[-1] != 2:
            raise ValueError("relative coordinates must have shape [B,11,P,2]")
        features = torch.cat((slot_latents[:, :, None, :].expand(-1, -1,
                              relative_xy.shape[2], -1), relative_xy), dim=-1)
        return self.net(features)


def _patch_centers(device, dtype):
    axis = (torch.arange(GRID, device=device, dtype=dtype) + 0.5) * (2.0 / GRID) - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _pixel_centers(device, dtype):
    axis = (torch.arange(IMAGE, device=device, dtype=dtype) + 0.5) * (2.0 / IMAGE) - 1.0
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def slot_geometry(assignments):
    """Live P-derived centroid and scale with exact 8x8 pixel-footprint variance."""
    if assignments.ndim != 3 or assignments.shape[1:] != (PATCHES, SLOTS):
        raise ValueError("assignments must have shape [B,256,11]")
    centers = _patch_centers(assignments.device, assignments.dtype)
    mass = assignments.sum(dim=1).clamp_min(1.0)
    centroid = torch.einsum("bnk,nd->bkd", assignments, centers) / mass[:, :, None]
    delta = centers[None, :, None, :] - centroid[:, None, :, :]
    variance = (assignments[..., None] * delta.square()).sum(dim=1) / mass[:, :, None]
    scale = (variance + CELL_VARIANCE).sqrt()
    return mass, centroid, scale


def pixel_assignment_weights(assignments):
    """Expand row-major patch assignments onto native 128x128 pixels."""
    if assignments.ndim != 3 or assignments.shape[1:] != (PATCHES, SLOTS):
        raise ValueError("assignments must have shape [B,256,11]")
    batch = assignments.shape[0]
    grid = assignments.reshape(batch, GRID, GRID, SLOTS)
    return grid.repeat_interleave(PATCH, dim=1).repeat_interleave(PATCH, dim=2).reshape(
        batch, IMAGE * IMAGE, SLOTS)


def _render_chunk(pixel_weights, slots, centroid, scale, xy, decoder):
    relative = (xy[None, None, :, :] - centroid[:, :, None, :]) / scale[:, :, None, :]
    slot_rgb = decoder(slots, relative)
    return torch.einsum("bpk,bkpc->bpc", pixel_weights, slot_rgb)


def render_slot_rgb(assignments, slots, decoder, *, chunk_size=PIXEL_CHUNK,
                    checkpoint_chunks=True):
    """Render [B,128,128,3] by slot RGB mixture with live P geometry and weights."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("pixel chunk size must be a positive integer")
    if assignments.shape[0] != slots.shape[0] or slots.shape[1:] != (SLOTS, DIM):
        raise ValueError("assignments and slot latent batch/shapes differ")
    _, centroid, scale = slot_geometry(assignments)
    weights = pixel_assignment_weights(assignments)
    xy = _pixel_centers(assignments.device, assignments.dtype)
    pieces = []
    for start in range(0, IMAGE * IMAGE, chunk_size):
        stop = min(start + chunk_size, IMAGE * IMAGE)
        args = (weights[:, start:stop], slots, centroid, scale, xy[start:stop])
        if checkpoint_chunks and torch.is_grad_enabled():
            rendered = checkpoint(
                lambda w, z, mu, sigma, pos: _render_chunk(w, z, mu, sigma, pos, decoder),
                *args, use_reentrant=False)
        else:
            rendered = _render_chunk(*args, decoder)
        pieces.append(rendered)
    result = torch.cat(pieces, dim=1).reshape(assignments.shape[0], IMAGE, IMAGE, 3)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite SW0134 RGB prediction")
    return result


def canonical_hard_labels(assignments):
    """Argmax slot masks with deterministic largest-group background and IDs.

    The largest occupied slot is label zero; size ties use first occupied patch,
    then the slot index. Remaining IDs use first occupied patch, then slot index.
    Empty slots are omitted; a one-slot partition is all background.
    """
    if assignments.ndim != 3 or assignments.shape[1:] != (PATCHES, SLOTS):
        raise ValueError("assignments must have shape [B,256,11]")
    argmax = assignments.detach().argmax(dim=-1).cpu()
    output = torch.zeros((assignments.shape[0], PATCHES), dtype=torch.int64)
    for batch_index, row in enumerate(argmax):
        slots = row.tolist()
        occupied = []
        for slot in range(SLOTS):
            indices = [index for index, value in enumerate(slots) if value == slot]
            if indices:
                occupied.append((slot, len(indices), indices[0]))
        if not occupied:
            continue
        background = sorted(occupied, key=lambda item: (-item[1], item[2], item[0]))[0][0]
        foreground = sorted((entry for entry in occupied if entry[0] != background),
                            key=lambda item: (item[2], item[0]))
        output[batch_index] = torch.tensor([
            0 if value == background else next(
                label for label, entry in enumerate(foreground, start=1) if entry[0] == value)
            for value in slots], dtype=torch.int64)
    return output.reshape(assignments.shape[0], GRID, GRID)


def reconstruct_from_spikes(actual_spikes, binder, decoder, *, initial_slots=None,
                             chunk_size=PIXEL_CHUNK, checkpoint_chunks=True):
    assignments, slots, patch_features = binder(actual_spikes, initial_slots=initial_slots)
    prediction = render_slot_rgb(assignments, slots, decoder, chunk_size=chunk_size,
                                 checkpoint_chunks=checkpoint_chunks)
    labels = canonical_hard_labels(assignments)
    return prediction, assignments, slots, patch_features, labels
