"""SW0178 full-label leave-one-out temporal class competition."""
from __future__ import annotations

import numpy as np  # Initialize NumPy before torch on Windows.
import torch


def native_component_rho_q(spikes: torch.Tensor, component_spikes: torch.Tensor,
                           *, settle: int = 512):
    """Return signed per-component rho and exact production positive-product Q."""
    from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import (
        native_component_rho_q as production_rho_q,
    )
    return production_rho_q(spikes, component_spikes, settle=settle)


def leave_one_out_class_loss(affinity: torch.Tensor, labels: torch.Tensor,
                             *, temperature: float = 0.1) -> torch.Tensor:
    """Class-balanced leave-one-out CE for one or more full labeled images.

    Args:
        affinity: finite `[B,N,N]` pair affinities.
        labels: integer `[B,N]` full modal labels, including class 0.
        temperature: positive finite scalar.

    Singleton query pixels are omitted, while singleton classes remain in the
    denominator for other queries. Images with no valid queries contribute a
    differentiable zero to the batch mean.
    """
    if not torch.is_tensor(affinity) or affinity.ndim != 3:
        raise ValueError("affinity must have shape [B,N,N]")
    if affinity.shape[1] != affinity.shape[2]:
        raise ValueError("affinity must be square over nodes")
    if not torch.is_tensor(labels) or labels.ndim != 2:
        raise ValueError("labels must have shape [B,N]; flatten modal grids explicitly at the caller")
    batch, nodes = labels.shape
    if affinity.shape != (batch, nodes, nodes):
        raise ValueError("labels and affinity geometry differ")
    if labels.dtype == torch.bool or labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError("labels must have an integer dtype")
    if labels.device != affinity.device:
        raise ValueError("labels and affinity must be on the same device")
    if not bool(torch.isfinite(affinity).all()):
        raise ValueError("affinity must be finite")
    if isinstance(temperature, bool) or not np.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    image_losses = []
    for image in range(batch):
        y = labels[image]
        classes = torch.unique(y, sorted=True)
        class_losses = []
        for query_class in classes:
            query_indices = torch.nonzero(y == query_class, as_tuple=False).flatten()
            if query_indices.numel() < 2:
                continue
            query_loss = []
            for query in query_indices:
                class_logits = []
                class_ids = []
                for class_id in classes:
                    members = torch.nonzero(y == class_id, as_tuple=False).flatten()
                    if class_id == query_class:
                        members = members[members != query]
                    if members.numel() == 0:
                        continue
                    class_logits.append(affinity[image, query, members].mean())
                    class_ids.append(class_id)
                if len(class_ids) < 2:
                    continue
                logits = torch.stack(class_logits) / float(temperature)
                target_pos = torch.nonzero(torch.stack(class_ids) == query_class,
                                           as_tuple=False).flatten()
                if target_pos.numel() != 1:
                    raise AssertionError("valid query must have exactly one own-class prototype")
                query_loss.append(torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), target_pos[:1].long()))
            if query_loss:
                class_losses.append(torch.stack(query_loss).mean())
        image_losses.append(torch.stack(class_losses).mean() if class_losses else affinity[image].sum() * 0.0)
    return torch.stack(image_losses).mean()


def temporal_affinity_losses(spikes: torch.Tensor, component_spikes: torch.Tensor,
                             labels: torch.Tensor, *, settle: int = 512,
                             temperature: float = 0.1):
    """Return matched signed-mean-rho and production-Q class losses plus tensors."""
    rho, q = native_component_rho_q(spikes, component_spikes, settle=settle)
    mean_rho = rho.mean(dim=1)
    return {
        "mean_rho_ce": leave_one_out_class_loss(mean_rho, labels, temperature=temperature),
        "product_q_ce": leave_one_out_class_loss(q, labels, temperature=temperature),
        "rho": rho,
        "q": q,
    }
