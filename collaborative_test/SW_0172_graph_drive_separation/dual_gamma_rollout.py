"""SW0172 dual-gamma routing over the frozen SW0134 native rollout."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0162_connected_signed_partition.preflight import (
    capture_source_coupling,
)
from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import (
    ContextualFeatureEncoder,
)


def encode_gamma_pair(found, rgb_uint8: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute source/context gamma from one shared frozen-base feature pass."""
    if not isinstance(getattr(found, "encoder", None), ContextualFeatureEncoder):
        raise TypeError("found.encoder must be the registered SW0171 contextual encoder")
    if not torch.is_tensor(rgb_uint8) or rgb_uint8.ndim != 4:
        raise ValueError("RGB must be a [B,128,128,3] tensor")
    if tuple(rgb_uint8.shape[1:]) != (128, 128, 3) or rgb_uint8.dtype != torch.uint8:
        raise ValueError("RGB must be uint8 [B,128,128,3]")
    encoder = found.encoder
    parameter = next(encoder.base_encoder.parameters())
    if parameter.dtype != torch.float32:
        raise TypeError("registered source kernels must be float32")
    images = rgb_uint8.to(device=parameter.device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    base_features = encoder.base_encoder(images)
    residual = encoder.context(images)[:, :, 1:-1, 1:-1]
    if tuple(base_features.shape[1:]) != (8, 126, 126) or residual.shape != base_features.shape:
        raise ValueError("base/context feature maps must both be [B,8,126,126]")
    base_gamma = _features_to_gamma(found, base_features)
    context_gamma = _features_to_gamma(found, base_features + residual)
    if base_gamma.shape != context_gamma.shape or tuple(base_gamma.shape[1:]) != (8, 1024):
        raise ValueError("native gamma tensors must be [B,8,1024]")
    return {"base_features": base_features, "context_residual": residual,
            "context_features": base_features + residual,
            "gamma_base": base_gamma, "gamma_context": context_gamma}


def _features_to_gamma(found, features: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(found.feature_mean, device=features.device, dtype=features.dtype)
    std = torch.as_tensor(found.feature_std, device=features.device, dtype=features.dtype)
    if mean.numel() == 8 and std.numel() == 8:
        mean, std = mean.reshape(1, 8, 1, 1), std.reshape(1, 8, 1, 1)
    clip = float(found.feature_clip)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("feature statistics must be finite with positive standard deviations")
    if not torch.isfinite(torch.tensor(clip)) or clip <= 0:
        raise ValueError("feature clip must be finite and positive")
    normalized = ((features - mean) / std).clamp(-clip, clip)
    gamma = found.patcher(normalized)
    if gamma.dtype != torch.float32 or not torch.isfinite(gamma).all():
        raise ValueError("native gamma must be finite float32")
    return gamma


@contextmanager
def _capture_graph(core):
    graph = getattr(core, "graph_generator", None)
    projection = getattr(graph, "projection", None)
    if projection is None:
        raise ValueError("SW0172 requires the native graph projection")
    drive_module = getattr(core, "gamma_to_drive", None)
    if drive_module is None:
        raise ValueError("SW0172 source core has no native GammaToDrive module")
    captured_projection, captured_adjacency, captured_drive = [], [], []
    handles = [
        projection.register_forward_hook(
            lambda _module, _inputs, output: captured_projection.append(output)),
        graph.register_forward_hook(
            lambda _module, _inputs, output: captured_adjacency.append(output)),
        drive_module.register_forward_hook(
            lambda _module, _inputs, output: captured_drive.append(output)),
    ]
    try:
        yield captured_projection, captured_adjacency, captured_drive
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _route_drive_input(core, drive_gamma):
    """Temporarily route only GammaToDrive's first input, restoring exact state."""
    module = core.gamma_to_drive
    if not isinstance(module, torch.nn.Module):
        raise TypeError("core.gamma_to_drive must be the registered module")
    if not torch.is_tensor(drive_gamma):
        raise TypeError("drive_gamma must be a tensor")
    # Do not mutate the module's registered hooks; return a removable handle.
    def replace_gamma(_module, args):
        if not args:
            raise ValueError("GammaToDrive forward requires gamma as its first argument")
        return (drive_gamma, *args[1:])
    handle = module.register_forward_pre_hook(replace_gamma)
    try:
        yield handle
    finally:
        handle.remove()


def run_dual_gamma_rollout(wrapped, gamma_graph: torch.Tensor,
                           gamma_drive: torch.Tensor, *, total_steps=None,
                           live_tail_steps=64) -> dict:
    """Run the frozen SW0134 rollout with graph and direct-drive gammas split.

    ``gamma_graph`` always feeds the actual native graph. The pre-hook changes
    only the first input to the actual ``GammaToDrive`` module. The frozen
    rollout remains the single source of recurrent dynamics and trace assembly.
    """
    if not torch.is_tensor(gamma_graph) or not torch.is_tensor(gamma_drive):
        raise TypeError("both gamma inputs must be tensors")
    if gamma_graph.shape != gamma_drive.shape or gamma_graph.ndim != 3:
        raise ValueError("graph and drive gamma must share [B,8,1024] geometry")
    if tuple(gamma_graph.shape[1:]) != (8, 1024):
        raise ValueError("gamma inputs must have native [B,8,1024] shape")
    if gamma_graph.dtype != torch.float32 or gamma_drive.dtype != torch.float32:
        raise TypeError("gamma inputs must be float32")
    if gamma_graph.device != gamma_drive.device:
        raise ValueError("gamma inputs must share a device")
    if not torch.isfinite(gamma_graph).all() or not torch.isfinite(gamma_drive).all():
        raise ValueError("gamma inputs must be finite")
    with _capture_graph(wrapped.core) as outputs:
        with _route_drive_input(wrapped.core, gamma_drive):
            trace, native_coupling = capture_source_coupling(
                wrapped.core.kuramoto,
                lambda: late_rollout(wrapped, gamma_graph, total_steps=total_steps,
                                     live_tail_steps=live_tail_steps))
    projections, adjacencies, drives = outputs
    if len(projections) != 1 or len(adjacencies) != 1 or len(drives) != 1:
        raise RuntimeError(
            "expected one actual graph projection/forward/drive call, observed "
            f"{len(projections)}/{len(adjacencies)}/{len(drives)}")
    projection_raw = projections[0]
    z = F.normalize(projection_raw, dim=-1)
    if not torch.isfinite(z).all():
        raise ValueError("captured native graph projection produced non-finite z")
    return {"trace": trace, "z": z, "projection_raw": projection_raw,
            "graph_adjacency": adjacencies[0], "adjacency": native_coupling,
            "drive": drives[0],
            "gamma_graph": gamma_graph, "gamma_drive": gamma_drive}
