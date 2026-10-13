"""SW0152 training primitives over the frozen SW0135 native32 foundation."""
from __future__ import annotations

import torch

from collaborative_test.SW_0145_native32_boundary_credit_probe import run as sw145
from collaborative_test.SW_0150_spike_spatial_mixture.spatial_diffusion import spatial_diffusion
from collaborative_test.SW_0152_source_preserving_partition_equivariance.deep_encoder import SourcePreservingDeepEncoder
from collaborative_test.SW_0152_source_preserving_partition_equivariance.partition_objective import horizontal_flip_indices, partition_objective
from collaborative_test.SW_0152_source_preserving_partition_equivariance.view_augmentation import make_views
from collaborative_test.SW_0150_spike_spatial_mixture.integration import forward_assignment


def install_deep_encoder(foundation: object) -> SourcePreservingDeepEncoder:
    """Wrap the already-loaded source encoder with an exact-zero spatial head."""
    if isinstance(foundation.encoder, SourcePreservingDeepEncoder):
        raise ValueError("SW0152 deep encoder is already installed")
    model = SourcePreservingDeepEncoder(foundation.encoder, freeze_base=True)
    # New residual layers follow the already loaded source encoder's device.
    model = model.to(next(foundation.encoder.parameters()).device)
    foundation.encoder = model
    return model


def make_paired_views(rgb_uint8: torch.Tensor, image_ids: list[int], *, source_seed: int):
    if rgb_uint8.dtype != torch.uint8 or rgb_uint8.ndim != 4 or rgb_uint8.shape[0] != len(image_ids):
        raise ValueError("paired view input must be uint8 NHWC and align with image IDs")
    clean, augmented, records = [], [], []
    for image, image_id in zip(rgb_uint8, image_ids):
        c, a, record = make_views(image, source_seed=source_seed, image_id=int(image_id))
        clean.append(c); augmented.append(a); records.append(record)
    return torch.stack(clean), torch.stack(augmented), records


def paired_assignments(foundation: object, clean_rgb: torch.Tensor,
                       transformed_rgb: torch.Tensor, *, live_tail_steps: int = 64):
    """One canonical QCC seed selection, bijectively mapped to the flipped view."""
    if clean_rgb.shape != transformed_rgb.shape:
        raise ValueError("paired RGB views must have the same batch shape")
    canonical = forward_assignment(foundation, clean_rgb, live_tail_steps=live_tail_steps)
    mapped = [horizontal_flip_indices(torch.tensor(row, dtype=torch.int64)).tolist()
              for row in canonical["seed_ids"]]
    gamma_t = foundation.encode(transformed_rgb)
    trace_t = foundation.rollout(gamma_t, total_steps=1024, live_tail_steps=live_tail_steps)
    q_t = sw145._actual_q(trace_t)
    p_t = spatial_diffusion(q_t, mapped)
    if p_t.shape[0] != clean_rgb.shape[0] or p_t.shape[1] != 1024:
        raise ValueError("transformed assignment does not match native32 grid")
    return canonical, {"gamma": gamma_t, "q": q_t, "p": p_t, "trace": trace_t,
                       "seed_ids": mapped}


def pair_loss(canonical_result: dict, transformed_result: dict,
              source_mass: torch.Tensor, *, arm: str) -> dict[str, torch.Tensor]:
    mode = {"equivariance": "candidate", "mass_control": "control"}.get(arm)
    if mode is None:
        raise ValueError("arm must be equivariance or mass_control")
    return partition_objective(canonical_result["p"], transformed_result["p"],
                               source_mass, mode=mode)


def trainable_families(foundation: object, arm: str):
    if arm not in ("equivariance", "mass_control"):
        raise ValueError("unregistered SW0152 arm")
    encoder = foundation.encoder
    if not isinstance(encoder, SourcePreservingDeepEncoder):
        raise ValueError("SW0152 residual encoder is not installed")
    core = foundation.wrapped.core
    return {
        "spatial_residual": list(encoder.residual.parameters()),
        "graph": list(core.graph_generator.parameters()),
        "native_core": [p for name, p in core.named_parameters()
                        if not name.startswith("graph_generator.")],
        "integration_scalars": [foundation.wrapped.a_d, foundation.wrapped.a_m,
                                foundation.wrapped.b],
    }


def configure_trainability(foundation: object, arm: str):
    families = trainable_families(foundation, arm)
    foundation.encoder.requires_grad_(False)
    foundation.encoder.residual.requires_grad_(True)
    foundation.encoder.train()
    foundation.encoder.base_encoder.eval()
    foundation.wrapped.requires_grad_(True)
    foundation.wrapped.core.graph_generator.requires_grad_(True)
    foundation.wrapped.train()
    parameters = [p for group in families.values() for p in group]
    if (not parameters or any(not p.requires_grad for p in parameters)
            or len({id(p) for p in parameters}) != len(parameters)):
        raise ValueError("SW0152 selected trainable families are invalid")
    return families


def make_optimizer(families: dict[str, list[torch.nn.Parameter]]):
    required = {"spatial_residual", "graph", "native_core", "integration_scalars"}
    if set(families) != required:
        raise ValueError("optimizer groups must match the registered four trainable families")
    rows = [
        {"name": "spatial_residual", "params": families["spatial_residual"], "lr": 3e-6},
        {"name": "graph", "params": families["graph"], "lr": 3e-5},
        {"name": "native_core", "params": families["native_core"], "lr": 3e-5},
        {"name": "integration_scalars", "params": families["integration_scalars"], "lr": 1e-3},
    ]
    return torch.optim.Adam(rows)


def state_sha(state_or_module) -> str:
    import hashlib
    import numpy as np
    state = state_or_module.state_dict() if isinstance(state_or_module, torch.nn.Module) else state_or_module
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(parameters) -> str:
    return state_sha({str(index): parameter for index, parameter in enumerate(parameters)})


def color_bin_row_permutation(rgb_uint8: torch.Tensor, p: torch.Tensor):
    """Within-image RGB 8-bin/channel row shuffle used only as TRAIN diagnostic."""
    if (rgb_uint8.dtype != torch.uint8 or rgb_uint8.ndim != 4
            or tuple(rgb_uint8.shape[1:]) != (128, 128, 3)):
        raise ValueError("diagnostic RGB must be uint8 [B,128,128,3]")
    if p.ndim != 3 or p.shape[:2] != (rgb_uint8.shape[0], 1024):
        raise ValueError("diagnostic P must be [B,1024,K]")
    patches = rgb_uint8.reshape(rgb_uint8.shape[0], 32, 4, 32, 4, 3).float().mean((2, 4)) / 255.0
    bins = torch.floor(patches.clamp(0, 1) * 8).to(torch.int64).clamp_max(7)
    codes = bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]
    codes = codes.reshape(rgb_uint8.shape[0], 1024).to(p.device)
    shuffled = p.clone()
    eligible = moved = 0
    for b in range(p.shape[0]):
        for color in torch.unique(codes[b], sorted=True):
            nodes = torch.nonzero(codes[b] == color, as_tuple=False).flatten()
            if nodes.numel() < 2:
                continue
            eligible += int(nodes.numel())
            values = p[b].index_select(0, nodes)
            shifted = values.roll(1, dims=0)
            shuffled[b, nodes] = shifted
            moved += int((shifted.argmax(-1) != values.argmax(-1)).sum())
    return shuffled, {"eligible_patch_count": eligible, "changed_hard_assignment_count": moved,
                      "moved_fraction": moved / eligible if eligible else 0.0}


def logical_update(*, optimizer, family_map, compute_micro, wrapped, micro_count: int = 4):
    """Backpropagate independent microbatch graphs before one clipped Adam step."""
    if micro_count != 4:
        raise ValueError("SW0152 uses exactly four microbatches per logical update")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for micro in range(micro_count):
        result = compute_micro(micro)
        loss = result["total"]
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("SW0152 objective must be a finite scalar")
        (loss / micro_count).backward()
        losses.append(float(loss.detach().cpu()))
        del result, loss
    norms = {}
    for name, parameters in family_map.items():
        sq = 0.0
        found = False
        for parameter in parameters:
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(f"nonfinite gradient in {name}")
                sq += float(grad.double().square().sum().cpu())
                found = True
        norms[name] = sq ** 0.5 if found else 0.0
    active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not active or any(norms[name] <= 0 for name in family_map):
        raise FloatingPointError("SW0152 update lacks finite positive selected-family gradient")
    torch.nn.utils.clip_grad_norm_(active, 1.0)
    optimizer.step()
    with torch.no_grad():
        wrapped.project_integrations_()
    return {"mean_loss": sum(losses) / micro_count, "family_gradient_norms": norms}
