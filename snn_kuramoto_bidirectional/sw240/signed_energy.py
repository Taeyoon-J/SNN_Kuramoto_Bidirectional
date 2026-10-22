from __future__ import annotations
import math
import numpy as np
NATIVE_NODES=1024
Q_EPS=1e-6

def signed_pair_energy(q, adjacency, coupling_gain):
    """Build the registered symmetric pair energy from actual Q and source A."""
    q = np.asarray(q)
    adjacency = np.asarray(adjacency)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] < 1:
        raise ValueError("Q must be a nonempty square matrix")
    if adjacency.shape != q.shape:
        raise ValueError("source adjacency must match Q")
    if (not np.isfinite(q).all() or not np.isfinite(adjacency).all()
            or np.any(q < -1e-7) or np.any(q > 1 + 1e-7)):
        raise ValueError("Q and source adjacency must be finite; Q must lie in [0,1]")
    if not np.allclose(q, q.T, rtol=0, atol=2e-6):
        raise ValueError("Q must be symmetric")
    if (not np.isscalar(coupling_gain) or not math.isfinite(float(coupling_gain))
            or float(coupling_gain) <= 0):
        raise ValueError("source coupling gain must be finite and positive")

    q64 = q.astype(np.float64, copy=False)
    q_sym = 0.5 * (q64 + q64.T)
    q_bar = np.clip(q_sym, Q_EPS, 1.0 - Q_EPS)
    logit_q = np.log(q_bar / (1.0 - q_bar))
    a64 = adjacency.astype(np.float64, copy=False)
    energy = (1.0 + NATIVE_NODES * (a64 + a64.T) / (2.0 * float(coupling_gain))) * logit_q
    energy = 0.5 * (energy + energy.T)
    np.fill_diagonal(energy, 0.0)
    if not np.isfinite(energy).all():
        raise ValueError("signed pair energy became nonfinite")
    return energy
