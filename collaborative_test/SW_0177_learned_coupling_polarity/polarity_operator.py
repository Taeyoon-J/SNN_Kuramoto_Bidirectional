"""SW0177 learnable symmetric coupling-polarity operator prototype."""

from contextlib import contextmanager
import math
import types

import numpy as np  # Keep NumPy before Torch for the Windows OpenMP runtime.
import torch
from torch import nn
from torch.nn import functional as F


class SymmetricEdgeMultiplier(nn.Module):
    """Shared 33->32(SiLU)->1 head over admitted undirected graph edges."""

    def __init__(self, grid_size=32, mode="signed"):
        super().__init__()
        if mode not in ("signed", "nonnegative"):
            raise ValueError("mode must be 'signed' or 'nonnegative'")
        self.grid_size = int(grid_size)
        self.mode = mode
        self.layers = nn.Sequential(
            nn.Linear(33, 32), nn.SiLU(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, z, structural_a):
        """Return dense symmetric m, fixed to one off admitted upper edges.

        ``z`` is the normalized projection [B,N,16] captured from the actual
        graph forward. ``structural_a`` is its positive pre-floor graph [B,N,N].
        """
        if z.ndim != 3 or structural_a.ndim != 3:
            raise ValueError("z and structural_a must be rank-3 batched tensors")
        b, n, d = z.shape
        if structural_a.shape != (b, n, n) or d != 16:
            raise ValueError("expected z [B,N,16] and matching structural graph")
        if not torch.isfinite(z).all() or not torch.isfinite(structural_a).all():
            raise ValueError("non-finite graph features or support")
        if (structural_a < 0).any():
            raise ValueError("structural support must be nonnegative")
        upper = torch.triu(structural_a > 0, diagonal=1)
        batch_ix, i, j = torch.where(upper)
        multiplier = torch.ones_like(structural_a)
        if i.numel() == 0:
            return multiplier

        grid_y = torch.div(torch.arange(n, device=z.device), self.grid_size,
                           rounding_mode="floor").to(z.dtype)
        grid_x = (torch.arange(n, device=z.device) % self.grid_size).to(z.dtype)
        distance = torch.sqrt((grid_y[i] - grid_y[j]).square()
                              + (grid_x[i] - grid_x[j]).square()) / 32.0
        zi, zj = z[batch_ix, i], z[batch_ix, j]
        features = torch.cat((zi + zj, torch.abs(zi - zj), distance[:, None]), dim=-1)
        r = self.layers(features).squeeze(-1)
        m = 1.0 - 2.0 * torch.tanh(r)
        if self.mode == "nonnegative":
            m = torch.abs(m)
        # index_put preserves the autograd path from the head to both mirrors.
        multiplier[batch_ix, i, j] = m
        multiplier[batch_ix, j, i] = m
        return multiplier


def seeded_edge_multiplier(*, seed=177001, grid_size=32, mode="signed"):
    """Construct a reproducible head while restoring the caller's CPU RNG.

    Construction uses CPU parameters first; moving the returned module to a
    leased device does not consume or reset that device's RNG stream.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        return SymmetricEdgeMultiplier(grid_size=grid_size, mode=mode)


def apply_lag_multiplier(coupling, structural_a, z, head):
    """Apply the head after native preparation; preserve native A_lat exactly."""
    if not isinstance(coupling, (tuple, list)) or len(coupling) != 3:
        raise ValueError("native prepare_coupling must return (A_lat,Lcos,Lsin)")
    a_lat, lag_cos, lag_sin = coupling
    if a_lat.shape != lag_cos.shape or a_lat.shape != lag_sin.shape:
        raise ValueError("native coupling tensors must have identical shapes")
    multiplier = head(z, structural_a)
    if multiplier.shape != a_lat.shape:
        raise ValueError("multiplier/native coupling shape mismatch")
    return (a_lat, lag_cos * multiplier, lag_sin * multiplier)


@contextmanager
def install_polarity_prepare(graph, kuramoto, head):
    """Instance-scope a post-prepare lag modifier and restore exactly on exit.

    The actual graph projection is observed once via a forward hook. The
    original native prepare_coupling performs symmetrization, ReLU, floor,
    normalization, and lag construction before the learned multiplier runs.
    """
    if not isinstance(head, SymmetricEdgeMultiplier):
        raise TypeError("head must be a SymmetricEdgeMultiplier")
    captured = {}

    def capture_projection(_module, _inputs, output):
        if output.ndim != 3:
            raise ValueError("actual graph projection output must be rank three")
        captured["z"] = F.normalize(output, dim=-1)

    hook = graph.projection.register_forward_hook(capture_projection)
    original_instance_value = kuramoto.__dict__.get("prepare_coupling")
    had_instance_value = "prepare_coupling" in kuramoto.__dict__
    original_bound = kuramoto.prepare_coupling

    def wrapped(this, A, batch_size=None, num_units=None, device=None):
        native = original_bound(A, batch_size=batch_size,
                                num_units=num_units, device=device)
        if A is None:
            raise ValueError("polarity requires the image-conditioned structural graph")
        if "z" not in captured:
            raise RuntimeError("graph projection was not captured in this forward")
        z = captured["z"]
        if z.shape[0] != A.shape[0] or z.shape[1] != A.shape[1]:
            raise ValueError("captured projection and structural graph do not align")
        return apply_lag_multiplier(native, A, z, head)

    kuramoto.prepare_coupling = types.MethodType(wrapped, kuramoto)
    try:
        yield captured
    finally:
        hook.remove()
        if had_instance_value:
            kuramoto.prepare_coupling = original_instance_value
        else:
            delattr(kuramoto, "prepare_coupling")

