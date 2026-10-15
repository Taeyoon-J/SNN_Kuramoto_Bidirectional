"""SW0175's registered signed-component maximin-route objective."""
from __future__ import annotations

import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as routes


def native_component_rho_q(spikes: torch.Tensor, component_spikes: torch.Tensor,
                           *, settle: int = 512, eps: float = 1e-8):
    """Return live component correlations and production positive-product Q.

    `component_spikes` is the actual emitted `[B,4,N,T]` trace. This mirrors
    `spike_synchrony_affinity(..., components=..., affinity_mode='spike')`
    operation-for-operation; `rho` is returned as `[B,4,N,N]` for the new
    positive route penalty, while Q retains the production product arithmetic.
    """
    if not torch.is_tensor(spikes) or not torch.is_tensor(component_spikes):
        raise TypeError("spikes and component_spikes must be tensors")
    if spikes.ndim != 3:
        raise ValueError("spikes must have shape [B,N,T]")
    if component_spikes.ndim != 4:
        raise ValueError("component_spikes must have shape [B,4,N,T]")
    batch, nodes, steps = spikes.shape
    if (component_spikes.shape[0] != batch or component_spikes.shape[1] != 4
            or component_spikes.shape[2] != nodes or component_spikes.shape[3] != steps):
        raise ValueError("component_spikes geometry must match [B,4,N,T] spikes")
    if isinstance(settle, bool) or not isinstance(settle, int) or not 0 <= settle < steps:
        raise ValueError("settle must be an integer in [0,T)")
    if not np.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    components = component_spikes[..., settle:]

    def correlate(value):
        value = value.float()
        value = value - value.mean(dim=-1, keepdim=True)
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(float(eps))
        return (value @ value.transpose(-1, -2)).clamp(-1.0, 1.0)

    per_component = [correlate(components[:, d]) for d in range(4)]
    # Keep the production stack -> positive clamp -> product sequence exactly.
    production_stack = torch.stack(per_component).clamp_min(0.0)
    q = production_stack.prod(dim=0)
    rho = torch.stack(per_component, dim=1)
    return rho, q


def critical_component_loss(q: torch.Tensor, rho: torch.Tensor, target, support,
                            *, plan=None):
    """Pair-mass weighted bottleneck positive route plus unchanged native Q² negatives.

    `q` is native positive-product Q `[N,N]`; `rho` is the live signed
    per-component correlation tensor `[4,N,N]` from the same emitted spikes.
    Route construction remains the frozen SW0168 `forest_routes` on detached Q.
    """
    if not torch.is_tensor(q) or not torch.is_tensor(rho):
        raise TypeError("q and rho must be tensors")
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must have shape [N,N]")
    if rho.shape != (4, q.shape[0], q.shape[1]):
        raise ValueError("rho must have shape [4,N,N] matching q")
    if q.device != rho.device:
        raise ValueError("q and rho must be on the same device")
    if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(rho).all()):
        raise ValueError("q and rho must be finite")
    if bool(((q < 0) | (q > 1)).any()) or bool(((rho < -1) | (rho > 1)).any()):
        raise ValueError("q must be in [0,1] and signed rho in [-1,1]")
    plan = routes.forest_routes(q.detach(), target, support) if plan is None else plan
    terms = []
    if plan["positive_pair_mass"]:
        edges = plan["positive_edges"]
        ij = torch.as_tensor([(i, j) for i, j, _ in edges], dtype=torch.long, device=q.device)
        weights = torch.as_tensor([w for _, _, w in edges], dtype=torch.float64, device=q.device)
        rho_edges = rho[:, ij[:, 0], ij[:, 1]]
        # amax deliberately retains PyTorch's equal-subgradient rule on ties.
        bottleneck_penalty = (1.0 - rho_edges).square().amax(dim=0)
        terms.append((weights * bottleneck_penalty).sum() / int(plan["positive_pair_mass"]))
    if plan["negative_pair_mass"]:
        values, weights = routes._edge_values(q, plan["negative_edges"])
        terms.append((weights * values.square()).sum() / int(plan["negative_pair_mass"]))
    if not terms:
        return q.sum() * 0.0
    return torch.stack(terms).mean()


def native_q_loss(q: torch.Tensor, target, support, *, plan=None):
    """Matched control objective: original pair-weighted `(1-Q)^2` and `Q^2`."""
    if not torch.is_tensor(q) or q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square tensor")
    plan = routes.forest_routes(q.detach(), target, support) if plan is None else plan
    terms = []
    if plan["positive_pair_mass"]:
        values, weights = routes._edge_values(q, plan["positive_edges"])
        terms.append((weights * (1.0 - values).square()).sum() / int(plan["positive_pair_mass"]))
    if plan["negative_pair_mass"]:
        values, weights = routes._edge_values(q, plan["negative_edges"])
        terms.append((weights * values.square()).sum() / int(plan["negative_pair_mass"]))
    if not terms:
        return q.sum() * 0.0
    return torch.stack(terms).mean()
