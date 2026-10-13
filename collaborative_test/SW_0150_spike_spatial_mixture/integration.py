"""Bind the SW0150 assignment kernels to one native32 source rollout."""
from __future__ import annotations

import torch

from collaborative_test.SW_0136_native32_transfer import evaluate as sw136_eval
from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145

from collaborative_test.SW_0150_spike_spatial_mixture.seed_selection import select_seed_ids
from collaborative_test.SW_0150_spike_spatial_mixture.spatial_diffusion import spatial_diffusion


ALLOWED_LIVE_TAILS = (0, 64)
EXPECTED_GAMMA_SHAPE = (8, 1024)
EXPECTED_Q_SHAPE = (1024, 1024)
EXPECTED_QCC_SHAPE = (32, 32)


def _validate_foundation(foundation):
    if not callable(getattr(foundation, "encode", None)) or not callable(
            getattr(foundation, "rollout", None)):
        raise TypeError("foundation must expose native32 encode() and rollout()")
    if not isinstance(getattr(foundation, "wrapped", None), torch.nn.Module):
        raise TypeError("foundation must carry the registered wrapped native core")


def forward_assignment(foundation, clean_rgb, *, live_tail_steps=64):
    """Produce native32 trace, actual-spike Q/QCC, deterministic seeds and live P.

    A single T=1024 source rollout feeds both the registered SW0145 actual-spike
    Q and SW0136 production QCC readout. Seed selection is detached by its
    helper; spatial diffusion remains differentiable with respect to Q.
    """
    _validate_foundation(foundation)
    if (isinstance(live_tail_steps, bool) or not isinstance(live_tail_steps, int)
            or live_tail_steps not in ALLOWED_LIVE_TAILS):
        raise ValueError("live_tail_steps must be exactly 0 or 64")
    if not torch.is_tensor(clean_rgb):
        clean_rgb = torch.as_tensor(clean_rgb)
    if (clean_rgb.dtype != torch.uint8 or clean_rgb.ndim != 4
            or tuple(clean_rgb.shape[1:]) != (128, 128, 3)
            or clean_rgb.shape[0] < 1):
        raise ValueError("clean_rgb must be NHWC uint8 [B,128,128,3]")

    gamma = foundation.encode(clean_rgb)
    if (not torch.is_tensor(gamma) or gamma.ndim != 3
            or tuple(gamma.shape[1:]) != EXPECTED_GAMMA_SHAPE
            or gamma.shape[0] != clean_rgb.shape[0]):
        raise ValueError("native32 encoder must produce gamma [B,8,1024]")
    if not torch.isfinite(gamma).all():
        raise FloatingPointError("native32 gamma contains nonfinite values")

    trace = foundation.rollout(gamma, total_steps=1024,
                               live_tail_steps=live_tail_steps)
    if not isinstance(trace, dict):
        raise TypeError("native rollout must return its trace dictionary")
    q = sw145._actual_q(trace)
    if (not torch.is_tensor(q) or q.ndim != 3
            or tuple(q.shape[1:]) != EXPECTED_Q_SHAPE
            or q.shape[0] != clean_rgb.shape[0]):
        raise ValueError("SW0145 actual-spike helper must return Q [B,1024,1024]")
    if not torch.isfinite(q).all():
        raise FloatingPointError("actual-spike Q contains nonfinite values")

    qcc = sw136_eval._qcc_labels(trace)
    qcc_tensor = torch.as_tensor(qcc)
    if (qcc_tensor.ndim != 3 or tuple(qcc_tensor.shape[1:]) != EXPECTED_QCC_SHAPE
            or qcc_tensor.shape[0] != clean_rgb.shape[0]):
        raise ValueError("SW0136 production QCC helper must return labels [B,32,32]")

    seeds = select_seed_ids(q, qcc_tensor)
    p = spatial_diffusion(q, seeds)
    if (not torch.isfinite(p).all() or torch.any(p < 0)
            or not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]),
                                  rtol=0, atol=1e-6)):
        raise FloatingPointError("native32 P must be finite, nonnegative, row-stochastic")
    primary_labels = p.argmax(dim=-1).reshape(-1, 32, 32)
    return {
        "gamma": gamma,
        "gamma_content": gamma.transpose(1, 2),
        "q": q,
        "qcc": qcc_tensor,
        "seed_ids": seeds,
        "p": p,
        "primary_labels": primary_labels,
        "trace": trace,
        "live_tail_steps": live_tail_steps,
    }
