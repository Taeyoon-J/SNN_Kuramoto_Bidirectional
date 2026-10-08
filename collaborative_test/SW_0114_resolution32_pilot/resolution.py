"""Strict spatial warm-start and memory-bounded geodesic helpers for SW0114."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import types


def parent_index(device=None):
    """Map each row-major 32x32 location to its 16x16 parent."""
    row = torch.arange(32, device=device) // 2
    col = torch.arange(32, device=device) // 2
    return (row[:, None] * 16 + col[None, :]).reshape(-1)


def grid_distance(grid_size: int, units_per_cell: float, *, device=None, dtype=torch.float32):
    axis = torch.arange(grid_size, device=device, dtype=dtype) * units_per_cell
    rows = torch.arange(grid_size, device=device).repeat_interleave(grid_size)
    cols = torch.arange(grid_size, device=device).repeat(grid_size)
    dy = (rows[:, None] - rows[None, :]).to(dtype) * units_per_cell
    dx = (cols[:, None] - cols[None, :]).to(dtype) * units_per_cell
    return torch.sqrt(dx.square() + dy.square())


def convert_state_dict(source, target_template, *, candidate: bool):
    """Build a strict target state; only declared per-node tensors are replicated."""
    mapped = {}
    p = parent_index(next(iter(source.values())).device)
    replicated = {
        "kuramoto.omega": (0,),
        "kuramoto.kappa": (0,),
        "dendric_layer.tau_n": (0,),
        "membrane_layer.tau_m": (0,),
    }
    matrix_keys = {"kuramoto.direction_learner"}
    for key, target in target_template.items():
        if key == "sc":
            if not candidate:
                if key not in source or source[key].shape != target.shape:
                    raise ValueError("control SC shape changed")
                mapped[key] = source[key].detach().clone()
            else:
                mapped[key] = torch.eye(1024, dtype=target.dtype, device=target.device)
            continue
        if key == "graph_generator.grid_distance":
            if not candidate:
                mapped[key] = source[key].detach().clone()
            else:
                mapped[key] = grid_distance(32, .5, device=target.device, dtype=target.dtype)
            continue
        if key not in source:
            raise KeyError(f"source checkpoint lacks target key {key}")
        value = source[key]
        if candidate and key in replicated:
            expected_rank = 1 if key == "membrane_layer.tau_m" else 2
            if value.ndim != expected_rank or value.shape[0] != 256 or target.shape[0] != 1024:
                raise ValueError(f"unexpected per-node tensor shape for {key}: {value.shape} -> {target.shape}")
            value = value.index_select(0, p.to(value.device))
        elif candidate and key in matrix_keys:
            if value.shape != (256, 256) or target.shape != (1024, 1024):
                raise ValueError(f"unexpected pairwise tensor shape for {key}")
            value = value.index_select(0, p.to(value.device)).index_select(1, p.to(value.device))
        if value.shape != target.shape:
            raise ValueError(f"unregistered shape conversion for {key}: {value.shape} -> {target.shape}")
        mapped[key] = value.detach().to(device=target.device, dtype=target.dtype).clone()
    if set(source) != set(target_template):
        raise ValueError(f"state key sets differ: missing={set(target_template)-set(source)}, extra={set(source)-set(target_template)}")
    return mapped


def geodesic_distance_chunked(z, euclidean, contrast, *, steps=3, radius=1.5,
                              temperature=.5, cap=16., chunk=32, log4=False):
    """Exact row-chunked relaxation; retain full intermediate-node reduction."""
    if z.ndim != 3 or euclidean.ndim != 3 or euclidean.shape[1] != euclidean.shape[2]:
        raise ValueError("expected z[B,N,D], euclidean[B,N,N]")
    n = z.shape[1]
    if euclidean.shape[0] not in (1, z.shape[0]) or euclidean.shape[1:] != (n, n):
        raise ValueError("geodesic geometry does not match embeddings")
    if chunk < 1 or temperature <= 0 or steps < 0:
        raise ValueError("invalid relaxation parameters")
    similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1., 1.)
    step = euclidean * (1. + F.softplus(torch.as_tensor(contrast, device=z.device, dtype=z.dtype))
                       * (1. - similarity))
    distance = torch.where(euclidean <= radius, step, torch.full_like(step, cap))
    correction = temperature * torch.log(torch.tensor(4., device=z.device, dtype=z.dtype)) if log4 else 0.
    for _ in range(steps):
        rows = []
        for start in range(0, n, chunk):
            left = distance[:, start:start + chunk, :].unsqueeze(2)
            right = distance.unsqueeze(1)
            relaxed = -temperature * torch.logsumexp(-(left + right) / temperature, dim=-1)
            relaxed = relaxed + correction
            rows.append(torch.minimum(distance[:, start:start + chunk, :], relaxed))
        distance = torch.cat(rows, dim=1)
    return distance.clamp(max=cap)


def attach_chunked_geodesic(graph, *, chunk=32, log4=False):
    """Replace only the graph's relaxation implementation; preserve its parameters."""
    def _method(self, z, euclidean):
        return geodesic_distance_chunked(
            z, euclidean, self.geodesic_contrast, steps=self.geodesic_steps,
            radius=self.geodesic_radius, temperature=self.geodesic_temperature,
            cap=self.geodesic_cap, chunk=chunk, log4=log4)
    graph._geodesic_distance = types.MethodType(_method, graph)
    return graph


def build_core(seed_state, *, grid_size: int, device="cpu"):
    """Instantiate registered legacy core and strict-load its 16/32 warm-start."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "collaborative_test")]
    from SW_0094_aligned_joint_pilot.run import hparams
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
    if grid_size not in (16, 32):
        raise ValueError("registered grids are 16 or 32")
    hp = hparams("raw")
    hp.num_regions = grid_size * grid_size
    hp.k = float(grid_size * grid_size)
    hp.graph_top_k = 32 if grid_size == 16 else 128
    hp.spike_spatial_grid_size = grid_size
    hp.gamma_patch_grid_size = grid_size
    hp.validate()
    core = S2NetCore(hp, device=device).to(device)
    mapped = convert_state_dict(seed_state, core.state_dict(), candidate=(grid_size == 32))
    core.load_state_dict(mapped, strict=True)
    if grid_size == 32:
        attach_chunked_geodesic(core.graph_generator, chunk=32, log4=True)
    core.graph_generator.requires_grad_(False)
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core
