"""Four matched spike-assignment heads with frozen-graph relation context."""
from __future__ import annotations

import math

import torch
from torch import nn

from collaborative_test.SW_0135_native32_spike_binding.binder import (
    COMPONENTS, DIM, PATCHES, SETTLED_STEPS, SLOTS, NativeSpikeSlotBinder,
)

ARMS = ("raw_only", "Q_only", "A_only", "Q_A")


def _validate_relation(value, batch, *, name, reference):
    if not torch.is_tensor(value) or value.ndim != 3 or tuple(value.shape) != (batch, PATCHES, PATCHES):
        raise ValueError(f"{name} must have shape [B,1024,1024]")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"{name} must match the spike tensor device and dtype")
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")
    if torch.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    if name == "Q" and torch.any(value > 1):
        raise ValueError("Q must lie in the registered [0,1] affinity range")
    if not torch.allclose(value, value.transpose(-1, -2), rtol=0.0, atol=2e-6):
        raise ValueError(f"{name} must be symmetric")
    # Frozen source topology is data, not a differentiable model input.
    matrix = value.detach().clone()
    diagonal = torch.arange(PATCHES, device=matrix.device)
    matrix[:, diagonal, diagonal] = 0
    row_sum = matrix.sum(dim=-1, keepdim=True)
    return matrix / row_sum.clamp_min(1e-8)


class RelationContextAssignment(nn.Module):
    """Exchangeable 11-slot binder with two shared Q/A message updates.

    Inputs are only actual emitted component spikes and frozen source-derived Q/A.
    No RGB, coordinates, GT, or object-count input is accepted. Q/A are detached
    because this matched first-stage comparison holds the source topology fixed;
    a later jointly learned-source stage must define its own gradient contract.
    """

    def __init__(self, arm: str, *, seed: int = 163001, binder_seed: int = 161001):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.arm = arm
        self.binder = NativeSpikeSlotBinder(seed=binder_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.relation_mlp = nn.Sequential(
                nn.Linear(3 * DIM, 128), nn.ReLU(), nn.Linear(128, DIM))
            nn.init.zeros_(self.relation_mlp[-1].weight)
            nn.init.zeros_(self.relation_mlp[-1].bias)

    def forward(self, actual_spikes, q, adjacency, *, initial_slots=None):
        if (not torch.is_tensor(actual_spikes) or actual_spikes.ndim != 4
                or tuple(actual_spikes.shape[1:]) != (COMPONENTS, PATCHES, SETTLED_STEPS)):
            raise ValueError("actual_spikes must have shape [B,4,1024,512]")
        if not torch.is_floating_point(actual_spikes) or not torch.isfinite(actual_spikes).all():
            raise ValueError("actual_spikes must be finite floating point")
        batch = actual_spikes.shape[0]
        if batch < 1:
            raise ValueError("batch must be nonempty")
        q_norm = _validate_relation(q, batch, name="Q", reference=actual_spikes)
        a_norm = _validate_relation(adjacency, batch, name="A", reference=actual_spikes)

        # Preserve the registered raw-spike projection and normalization.
        patch = actual_spikes.permute(0, 2, 1, 3).reshape(batch, PATCHES, -1)
        h = self.binder.input_norm(self.binder.input_projection(patch))
        use_q = self.arm in ("Q_only", "Q_A")
        use_a = self.arm in ("A_only", "Q_A")
        for _ in range(2):
            q_message = torch.bmm(q_norm, h) if use_q else torch.zeros_like(h)
            a_message = torch.bmm(a_norm, h) if use_a else torch.zeros_like(h)
            update = self.relation_mlp(torch.cat((h, q_message, a_message), dim=-1))
            h = h + update

        # Same three shared exchangeable attention/GRU iterations as SW0135.
        slots = self.binder.initial_slots(batch) if initial_slots is None else initial_slots
        if tuple(slots.shape) != (batch, SLOTS, DIM):
            raise ValueError("initial slots must have shape [B,11,64]")
        if slots.device != h.device or slots.dtype != h.dtype:
            raise ValueError("initial slots must match the spike feature device and dtype")
        key, value = self.binder.key(h), self.binder.value(h)
        scale = math.sqrt(DIM)
        for _ in range(3):
            logits = torch.einsum("bnd,bkd->bnk", key, self.binder.query(slots)) / scale
            patch_to_slot = torch.softmax(logits, dim=-1)
            slot_to_patch = patch_to_slot / patch_to_slot.sum(dim=1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bnk,bnd->bkd", slot_to_patch, value)
            slots = self.binder.gru(updates.reshape(-1, DIM), slots.reshape(-1, DIM)).reshape(
                batch, SLOTS, DIM)
            slots = slots + self.binder.slot_mlp(slots)
        final_logits = torch.einsum("bnd,bkd->bnk", key, self.binder.query(slots)) / scale
        probabilities = torch.softmax(final_logits, dim=-1)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(slots).all():
            raise FloatingPointError("nonfinite SW0163 assignment")
        return probabilities, slots, h
