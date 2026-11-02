"""Adaptive temporal assignment and slot-conditioned native-RGB decoder.

The assignment path consumes only the four actual component spike traces.
Spatial coordinates enter only the RGB decoder after the assignments form
per-slot temporal embeddings.
"""
from __future__ import annotations

import torch
from torch import nn


COMPONENTS = 4
PATCH_GRID = 16
PATCHES = PATCH_GRID * PATCH_GRID
PATCH_SIZE = 8
SLOTS = 11
TEMPORAL_WIDTH = 64
DECODER_WIDTH = 128
ASSIGNMENT_TEMPERATURE = 0.1
RMS_EPS = 1e-8


def image_from_patches(patches: torch.Tensor) -> torch.Tensor:
    """Convert [B,256,8,8,3] RGB patches into [B,3,128,128]."""
    if patches.ndim != 5 or tuple(patches.shape[1:]) != (PATCHES, PATCH_SIZE, PATCH_SIZE, 3):
        raise ValueError("patch RGB tensor must be [B,256,8,8,3]")
    b = patches.shape[0]
    return (patches.reshape(b, PATCH_GRID, PATCH_GRID, PATCH_SIZE, PATCH_SIZE, 3)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, 3, PATCH_GRID * PATCH_SIZE, PATCH_GRID * PATCH_SIZE))


def normalized_patch_centers(device=None, dtype=torch.float32) -> torch.Tensor:
    axis = (torch.arange(PATCH_GRID, device=device, dtype=dtype) + 0.5) * (2.0 / PATCH_GRID) - 1.0
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((col.reshape(-1), row.reshape(-1)), dim=-1)


def fit_spike_rms(component_spikes: torch.Tensor) -> torch.Tensor:
    """Fit the shared legacy four-channel RMS from actual TRAIN spike traces."""
    if component_spikes.ndim != 4 or component_spikes.shape[1:3] != (COMPONENTS, PATCHES):
        raise ValueError("actual component spikes must be [B,4,256,T]")
    if component_spikes.shape[0] < 1 or component_spikes.shape[-1] < 1:
        raise ValueError("spike population must be nonempty")
    if not torch.isfinite(component_spikes).all():
        raise FloatingPointError("actual spike traces contain nonfinite values")
    rms = component_spikes.float().square().mean(dim=(0, 2, 3)).sqrt()
    return torch.where(rms > RMS_EPS, rms, torch.ones_like(rms)).detach()


class AdaptiveTemporalAssignment(nn.Module):
    """Map [B,4,256,T] actual spikes to 11-way patch assignments."""

    def __init__(self, spike_rms=None):
        super().__init__()
        rms = torch.ones(COMPONENTS) if spike_rms is None else torch.as_tensor(spike_rms, dtype=torch.float32)
        rms = rms.detach().reshape(-1)
        if rms.shape != (COMPONENTS,) or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("spike_rms must contain four finite positive frozen values")
        self.register_buffer("spike_rms", rms.clone())
        self.temporal = nn.Sequential(
            nn.Conv1d(COMPONENTS, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, TEMPORAL_WIDTH, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.assignment = nn.Linear(TEMPORAL_WIDTH, SLOTS)

    def forward(self, component_spikes: torch.Tensor):
        if component_spikes.ndim != 4 or component_spikes.shape[1:3] != (COMPONENTS, PATCHES):
            raise ValueError("actual component spikes must be [B,4,256,T]")
        if component_spikes.shape[0] < 1 or component_spikes.shape[-1] < 1:
            raise ValueError("spike batch and time dimensions must be nonempty")
        if not torch.isfinite(component_spikes).all():
            raise FloatingPointError("actual spike traces contain nonfinite values")
        b, _, n, steps = component_spikes.shape
        normalized = component_spikes / self.spike_rms.to(
            device=component_spikes.device, dtype=component_spikes.dtype).view(1, COMPONENTS, 1, 1)
        temporal_input = normalized.permute(0, 2, 1, 3).reshape(b * n, COMPONENTS, steps)
        patch_embedding = self.temporal(temporal_input).mean(dim=-1).reshape(b, n, TEMPORAL_WIDTH)
        probability = torch.softmax(self.assignment(patch_embedding) / ASSIGNMENT_TEMPERATURE, dim=-1)
        slot_mass = probability.sum(dim=1).clamp_min(1e-8)
        slot_embedding = torch.bmm(probability.transpose(1, 2), patch_embedding) / slot_mass.unsqueeze(-1)
        return probability, slot_embedding, patch_embedding


class SlotSpatialRGBDecoder(nn.Module):
    """Decode each slot at each patch coordinate, then mix by assignment."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(TEMPORAL_WIDTH + 2, DECODER_WIDTH),
            nn.GELU(),
            nn.Linear(DECODER_WIDTH, DECODER_WIDTH),
            nn.GELU(),
            nn.Linear(DECODER_WIDTH, PATCH_SIZE * PATCH_SIZE * 3),
            nn.Sigmoid(),
        )

    def forward(self, probability: torch.Tensor, slot_embedding: torch.Tensor,
                patch_xy: torch.Tensor):
        if probability.ndim != 3 or tuple(probability.shape[1:]) != (PATCHES, SLOTS):
            raise ValueError("assignment probabilities must be [B,256,11]")
        if slot_embedding.shape != (probability.shape[0], SLOTS, TEMPORAL_WIDTH):
            raise ValueError("slot embeddings must be [B,11,64]")
        if patch_xy.ndim == 2:
            patch_xy = patch_xy.unsqueeze(0).expand(probability.shape[0], -1, -1)
        if patch_xy.shape != (probability.shape[0], PATCHES, 2):
            raise ValueError("patch coordinates must be [256,2] or [B,256,2]")
        if not (torch.isfinite(probability).all() and torch.isfinite(slot_embedding).all()
                and torch.isfinite(patch_xy).all()):
            raise FloatingPointError("decoder inputs must be finite")
        b = probability.shape[0]
        xy = patch_xy.to(device=slot_embedding.device, dtype=slot_embedding.dtype)
        latent = slot_embedding[:, :, None, :].expand(-1, -1, PATCHES, -1)
        coords = xy[:, None, :, :].expand(-1, SLOTS, -1, -1)
        decoded = self.network(torch.cat((latent, coords), dim=-1))
        decoded = decoded.reshape(b, SLOTS, PATCHES, PATCH_SIZE, PATCH_SIZE, 3)
        mixed = torch.einsum("bnk,bknhwc->bnhwc", probability, decoded)
        return mixed, decoded


class AdaptiveTemporalRGBModel(nn.Module):
    """End-to-end actual-spike assignment with a slot-spatial RGB decoder."""

    def __init__(self):
        super().__init__()
        self.assignment_head = AdaptiveTemporalAssignment()
        self.rgb_decoder = SlotSpatialRGBDecoder()

    def forward(self, component_spikes: torch.Tensor, patch_xy: torch.Tensor):
        probability, slot_embedding, patch_embedding = self.assignment_head(component_spikes)
        reconstructed_patches, decoded = self.rgb_decoder(probability, slot_embedding, patch_xy)
        return {
            "assignment": probability,
            "slot_embedding": slot_embedding,
            "patch_embedding": patch_embedding,
            "reconstructed_patches": reconstructed_patches,
            "decoded_slot_patches": decoded,
            "reconstructed_image": image_from_patches(reconstructed_patches),
        }
