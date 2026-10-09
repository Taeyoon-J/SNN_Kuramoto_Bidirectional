"""GT-free temporal prototypes over frozen SW0097 actual component spikes."""
from __future__ import annotations

import torch


COMPONENTS = 4
PATCHES = 256
SLOTS = 11
TEMPERATURE = 0.1
AFFINITY_STOP = 0.50
MAX_PROTOTYPES = 11
REFINEMENT_STEPS = 3
RMS_EPS = 1e-8


def center_scale_traces(component_spikes: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
    """Center actual traces over time and apply the frozen legacy TRAIN RMS."""
    if (component_spikes.ndim != 4 or component_spikes.shape[1:3] != (COMPONENTS, PATCHES)
            or component_spikes.shape[-1] < 2):
        raise ValueError("component_spikes must be [B,4,256,T>=2]")
    rms = torch.as_tensor(rms, device=component_spikes.device, dtype=component_spikes.dtype).reshape(-1)
    if rms.shape != (COMPONENTS,) or not torch.isfinite(rms).all() or (rms <= 0).any():
        raise ValueError("frozen legacy RMS must contain four finite positive values")
    if not torch.isfinite(component_spikes).all():
        raise FloatingPointError("component spike traces must be finite")
    return (component_spikes - component_spikes.mean(dim=-1, keepdim=True)) / rms.view(1, 4, 1, 1)


def _select_anchors(x: torch.Tensor, q: torch.Tensor, stop_threshold=AFFINITY_STOP,
                    max_prototypes=MAX_PROTOTYPES):
    """Select deterministic distinct-trace anchors from detached production Q."""
    if x.ndim != 3 or x.shape[:2] != (COMPONENTS, PATCHES):
        raise ValueError("one trace tensor must have shape [4,256,T]")
    if q.shape != (PATCHES, PATCHES) or not torch.isfinite(q).all():
        raise ValueError("production affinity must be finite [256,256]")
    # Anchor selection is explicitly detached. Move just these compact
    # selection tensors once, avoiding a device synchronization for every
    # exact-duplicate comparison in the deterministic patch loop.
    q_detached = q.detach().to(device="cpu")
    x_rows = x.detach().permute(1, 0, 2).reshape(PATCHES, -1).to(device="cpu").contiguous()
    # Torch equality treats +/-0 as equal; canonicalize zero before byte keys.
    canonical_rows = torch.where(x_rows == 0, torch.zeros_like(x_rows), x_rows)
    trace_keys = [row.numpy().tobytes() for row in canonical_rows]
    first = int(torch.argmax(q_detached.sum(dim=1)))  # first index wins ties
    anchors = [first]
    while len(anchors) < max_prototypes:
        selected_keys = {trace_keys[patch] for patch in anchors}
        # Keep the lowest patch ID as the representative of each exact trace.
        representatives = {}
        for patch, key in enumerate(trace_keys):
            if key not in representatives:
                representatives[key] = patch
        candidates = [patch for key, patch in representatives.items()
                      if key not in selected_keys]
        if not candidates:
            break
        candidate_index = torch.as_tensor(candidates, device=q_detached.device, dtype=torch.long)
        max_affinity = q_detached.index_select(0, candidate_index)[:, anchors].max(dim=1).values
        # candidates are ascending patch IDs, so argmin gives deterministic lowest-ID ties.
        nearest, position = max_affinity.min(dim=0)
        if float(nearest) >= stop_threshold:
            break
        anchors.append(candidates[int(position)])
    return anchors


def _assign_one(x: torch.Tensor, q: torch.Tensor, temperature=TEMPERATURE,
                stop_threshold=AFFINITY_STOP, max_prototypes=MAX_PROTOTYPES):
    anchors = _select_anchors(x, q, stop_threshold, max_prototypes)
    # Initial centers remain live: selection is detached, prototype learning is not.
    prototypes = x[:, anchors, :].permute(1, 0, 2)
    patch_traces = x.permute(1, 0, 2)

    def score(current):
        return -(patch_traces[:, None] - current[None]).square().mean(dim=(2, 3))

    for _ in range(REFINEMENT_STEPS):
        probability = torch.softmax(score(prototypes) / temperature, dim=-1)
        mass = probability.sum(dim=0).clamp_min(1.0)
        prototypes = torch.einsum("nk,ndt->kdt", probability, patch_traces) / mass[:, None, None]
    probability = torch.softmax(score(prototypes) / temperature, dim=-1)
    if len(anchors) < SLOTS:
        probability = torch.nn.functional.pad(probability, (0, SLOTS - len(anchors)))
    return probability, anchors, prototypes


def temporal_prototype_assignment(component_spikes: torch.Tensor, rms: torch.Tensor,
                                  production_q: torch.Tensor, *, settle: int,
                                  stop_threshold=AFFINITY_STOP):
    """Return padded [B,256,11] assignment, selected anchors, and live prototypes.

    ``production_q`` must be the registered detached positive four-component
    product affinity computed from the same actual rollout. The caller retains
    responsibility for using the production classifier for the comparison arm.
    """
    if (component_spikes.ndim != 4 or component_spikes.shape[1:3] != (4, PATCHES)
            or not 0 <= settle < component_spikes.shape[-1]):
        raise ValueError("invalid actual component trace shape or settle index")
    if production_q.shape != (component_spikes.shape[0], PATCHES, PATCHES):
        raise ValueError("production_q must be [B,256,256] from this rollout")
    x = center_scale_traces(component_spikes[..., settle:], rms)
    assignments, anchors, prototypes = [], [], []
    for batch in range(x.shape[0]):
        p, anchor, centers = _assign_one(x[batch], production_q[batch].detach(),
                                         stop_threshold=stop_threshold)
        assignments.append(p)
        anchors.append(anchor)
        prototypes.append(centers)
    probability = torch.stack(assignments)
    if (tuple(probability.shape) != (x.shape[0], PATCHES, SLOTS)
            or not torch.isfinite(probability).all()
            or (probability < 0).any()):
        raise AssertionError("temporal prototype readout returned invalid probabilities")
    return probability, anchors, prototypes
