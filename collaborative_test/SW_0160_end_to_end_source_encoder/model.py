"""SW0160 matched source-encoder training configuration (no runner side effects)."""
from __future__ import annotations

import torch

from collaborative_test.SW_0152_source_preserving_partition_equivariance.deep_encoder import (
    SourcePreservingDeepEncoder,
)

ARMS = ("frozen_base_joint", "full_encoder_joint")
LEARNING_RATES = {
    "source_base": 3e-5,
    "spatial_residual": 3e-6,
    "graph": 3e-5,
    "native_core": 3e-5,
    "integration_tensors": 1e-3,
}


def install_encoder(foundation, arm: str, *, residual_seed: int = 160001):
    """Install identical zero-output encoder residual; unfreeze base only by arm."""
    if arm not in ARMS:
        raise ValueError("unregistered SW0160 arm")
    if isinstance(foundation.encoder, SourcePreservingDeepEncoder):
        raise ValueError("SW0160 encoder already installed")
    device = next(foundation.encoder.parameters()).device
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(residual_seed))
        wrapper = SourcePreservingDeepEncoder(foundation.encoder, freeze_base=False).to(device)
    wrapper.base_encoder.requires_grad_(arm == "full_encoder_joint")
    wrapper.residual.requires_grad_(True)
    wrapper.train()
    # The source encoder has no running-statistics layers; keep its registered
    # training mode in both arms and do not change normalization/clipping.
    foundation.encoder = wrapper
    return wrapper


def trainable_families(foundation, arm: str):
    if arm not in ARMS or not isinstance(foundation.encoder, SourcePreservingDeepEncoder):
        raise ValueError("invalid SW0160 arm or encoder")
    encoder = foundation.encoder
    graph = foundation.wrapped.core.graph_generator
    if graph is None or getattr(graph, "_sw0143_geodesic_arm", None) != "T":
        raise ValueError("SW0160 requires the registered SW0143-T graph")
    graph_params = []
    for name, parameter in graph.named_parameters():
        parameter.requires_grad_(name != "geodesic_contrast")
        if name != "geodesic_contrast":
            graph_params.append(parameter)
    native = [p for name, p in foundation.wrapped.core.named_parameters()
              if not name.startswith("graph_generator.")]
    integrations = [foundation.wrapped.a_d, foundation.wrapped.a_m, foundation.wrapped.b]
    base = list(encoder.base_encoder.parameters())
    residual = list(encoder.residual.parameters())
    encoder.residual.requires_grad_(True)
    for p in base:
        p.requires_grad_(arm == "full_encoder_joint")
    families = {
        "source_base": base if arm == "full_encoder_joint" else [],
        "spatial_residual": residual,
        "graph": graph_params,
        "native_core": native,
        "integration_tensors": integrations,
    }
    selected = [p for values in families.values() for p in values]
    if not selected or len({id(p) for p in selected}) != len(selected):
        raise ValueError("SW0160 optimizer parameter families are empty or duplicated")
    if any(not p.requires_grad for p in selected):
        raise ValueError("SW0160 selected parameter is unexpectedly frozen")
    if arm == "frozen_base_joint" and any(p.requires_grad for p in base):
        raise ValueError("matched control source base must remain frozen")
    return families


def make_optimizer(families):
    if set(families) != set(LEARNING_RATES):
        raise ValueError("SW0160 optimizer family schema changed")
    groups = []
    for name, rate in LEARNING_RATES.items():
        params = list(families[name])
        if params:
            groups.append({"name": name, "params": params, "lr": rate})
    return torch.optim.Adam(groups)
