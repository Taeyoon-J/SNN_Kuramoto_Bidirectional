from __future__ import annotations
import numpy as np
import torch

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
