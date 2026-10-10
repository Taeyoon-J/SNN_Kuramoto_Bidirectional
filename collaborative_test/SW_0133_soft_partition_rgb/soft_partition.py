"""Genuine soft-assignment SW0106 RGB reconstruction for SW0133.

Unlike SW0132, the forward mixture uses P itself. Assignment, pooled content,
relative geometry, and pixel mixture all remain differentiable when requested.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from collaborative_test.SW_0106_spike_partition_rgb.partition_rgb import assignment_weights
from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import render_partition


def soft_weights(q, hard, *, assignment_live=True):
    """Return registered SW0106 P and either live P or detached P as W."""
    _registered_ste, p = assignment_weights(q, hard, credit=True)
    if p is None:
        raise AssertionError("registered assignment helper did not return soft P")
    weights = p if assignment_live else p.detach()
    return weights, p


def reconstruct_soft(q, hard, gamma, target_rgb, decoder, *, assignment_live=True,
                    chunk_size=1024, checkpoint_chunks=True):
    """Return full native-resolution reconstruction, loss, and assignments.

    The hard partition defines the number and identity of available slots. The
    forward RGB mixture is a genuine soft P (or detached P control), not an STE.
    """
    weights, p = soft_weights(q, hard, assignment_live=assignment_live)
    prediction = render_partition(weights, gamma, decoder, chunk_size=chunk_size,
                                  checkpoint_chunks=checkpoint_chunks)
    if target_rgb.shape != prediction.shape:
        raise ValueError("target RGB must match the rendered [128,128,3] image")
    loss = F.mse_loss(prediction, target_rgb)
    return prediction, loss, {"W": weights, "P": p, "K": int(hard.shape[1])}
