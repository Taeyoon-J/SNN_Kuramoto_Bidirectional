"""Imagewise iterative slot binder for SW0126 (prospective, no GT inputs)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalSlotRGBBinder(nn.Module):
    """Bind four-component event histories to slots and reconstruct native RGB.

    The classifier sees only ``[B,4,N,T]`` event/gate traces. Patch coordinates
    are supplied only to the shared RGB decoder. There is no independent mask
    or alpha head: the interpolated slot assignments are the RGB mixture.
    """

    def __init__(self, max_slots=11, feature_dim=64, iterations=3,
                 patch_grid=16, image_size=128, init_seed=126):
        super().__init__()
        if max_slots != 11 or iterations != 3:
            raise ValueError("SW0126 registers exactly 11 slots and 3 binding updates")
        if feature_dim != 64 or patch_grid < 1 or image_size % patch_grid:
            raise ValueError("SW0126 requires 64-D features and divisible image/patch geometry")
        self.max_slots = int(max_slots)
        self.feature_dim = int(feature_dim)
        self.iterations = int(iterations)
        self.patch_grid = int(patch_grid)
        self.image_size = int(image_size)

        # Same seed gives the paired event/gate binders identical initialization
        # without perturbing the caller's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.temporal_projection = nn.Linear(4 * 512, self.feature_dim)
            self.slot_queries = nn.Parameter(torch.empty(self.max_slots, self.feature_dim))
            nn.init.normal_(self.slot_queries, mean=0.0, std=self.feature_dim ** -0.5)
            self.coordinate_basis = nn.Sequential(
                nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 32))
            self.slot_rgb_coefficients = nn.Linear(self.feature_dim, 32 * 3)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.slot_norm = nn.LayerNorm(self.feature_dim)

    def assignments(self, component_trace):
        if component_trace.ndim != 4 or component_trace.shape[1] != 4:
            raise ValueError("binder input must be actual [B,4,N,T] component traces")
        batch, _, patches, steps = component_trace.shape
        if patches != self.patch_grid * self.patch_grid or steps != 512:
            raise ValueError("SW0126 binder expects the full settled 512-frame trace on its patch grid")
        if not torch.isfinite(component_trace).all():
            raise ValueError("binder input contains nonfinite values")

        flattened = component_trace.permute(0, 2, 1, 3).reshape(batch, patches, 4 * steps)
        features = self.feature_norm(self.temporal_projection(flattened))
        slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        scale = math.sqrt(self.feature_dim)
        for _ in range(self.iterations):
            logits = torch.einsum("bkd,bnd->bkn", slots, features) / scale
            weights = torch.softmax(logits, dim=1)
            mass = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            update = torch.bmm(weights, features) / mass
            slots = self.slot_norm(slots + update)
        logits = torch.einsum("bkd,bnd->bkn", slots, features) / scale
        assignment = torch.softmax(logits, dim=1)
        return assignment, slots

    def forward(self, component_trace):
        assignment, slots = self.assignments(component_trace)
        batch = component_trace.shape[0]
        size = self.image_size
        axis = torch.linspace(-1.0, 1.0, size, device=slots.device, dtype=slots.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(1, 1, size * size, 2)
        coordinate_basis = self.coordinate_basis(coords.reshape(size * size, 2)).reshape(
            size, size, 32)
        slot_rgb_coeff = self.slot_rgb_coefficients(slots).reshape(
            batch, self.max_slots, 32, 3)

        patch_assignment = assignment.reshape(
            batch, self.max_slots, self.patch_grid, self.patch_grid)
        pixel_assignment = F.interpolate(patch_assignment, size=(size, size),
                                         mode="bilinear", align_corners=False)
        # Decode each slot's RGB field, then mix RGB with P. Applying sigmoid
        # after mixing logits would change the registered mixture semantics.
        slot_rgb = torch.sigmoid(torch.einsum(
            "hwf,bkfc->bkhwc", coordinate_basis, slot_rgb_coeff))
        reconstruction = torch.einsum("bkhw,bkhwc->bhwc",
                                      pixel_assignment, slot_rgb)
        return {
            "assignment": assignment,
            "slots": slots,
            "slot_rgb_coefficients": slot_rgb_coeff,
            "coordinate_basis": coordinate_basis,
            "slot_rgb": slot_rgb,
            "pixel_assignment": pixel_assignment,
            "reconstruction": reconstruction,
        }
