"""Strict 16→32 source mapping and checkpointed native32 geodesic operator."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

OLD_GRID = 16
GRID = 32
NODES = GRID * GRID
ROW_CHUNK = 32


def parent_index(*, device=None):
    """Spatially map each row-major32 node to its 16-grid parent."""
    row = torch.arange(GRID, device=device) // 2
    col = torch.arange(GRID, device=device) // 2
    return (row[:, None] * OLD_GRID + col[None, :]).reshape(-1)


def grid_distance(grid_size: int, units_per_cell: float, *, device=None, dtype=torch.float32):
    if not isinstance(grid_size, int) or grid_size < 1 or units_per_cell <= 0:
        raise ValueError("grid size and cell distance must be positive")
    row = torch.arange(grid_size, device=device).repeat_interleave(grid_size)
    col = torch.arange(grid_size, device=device).repeat(grid_size)
    dy = (row[:, None] - row[None, :]).to(dtype) * units_per_cell
    dx = (col[:, None] - col[None, :]).to(dtype) * units_per_cell
    return torch.sqrt(dx.square() + dy.square())


def convert_state_dict(source, target_template):
    """Strictly map declared256-node tensors, shared state unchanged, SC identity."""
    if not isinstance(source, dict) or not isinstance(target_template, dict):
        raise TypeError("source and target state dictionaries are required")
    if set(source) != set(target_template):
        raise ValueError(f"source/target state keys differ: missing={set(target_template)-set(source)}, "
                         f"extra={set(source)-set(target_template)}")
    if not source:
        raise ValueError("empty source state")
    device = next(iter(target_template.values())).device
    p = parent_index(device=device)
    node_ranks = {"kuramoto.omega": 2, "kuramoto.kappa": 2,
                  "dendric_layer.tau_n": 2, "membrane_layer.tau_m": 1}
    pair_keys = {"kuramoto.direction_learner"}
    mapped = {}
    for key, target in target_template.items():
        value = source[key]
        if key == "sc":
            expected = (GRID * GRID, GRID * GRID)
            if tuple(target.shape) != expected:
                raise ValueError(f"candidate structural identity shape is invalid: {target.shape}")
            mapped[key] = torch.eye(*expected, device=target.device, dtype=target.dtype)
        elif key == "graph_generator.grid_distance":
            if tuple(target.shape) != (NODES, NODES):
                raise ValueError("candidate graph grid-distance shape is invalid")
            mapped[key] = grid_distance(GRID, .5, device=target.device, dtype=target.dtype)
        elif key in node_ranks:
            if value.ndim != node_ranks[key] or value.shape[0] != OLD_GRID * OLD_GRID \
                    or target.shape[0] != NODES:
                raise ValueError(f"unregistered source node mapping for {key}: {tuple(value.shape)} → {tuple(target.shape)}")
            converted = value.index_select(0, p.to(value.device))
            if converted.shape != target.shape:
                raise ValueError(f"mapped node shape mismatch for {key}")
            mapped[key] = converted.to(device=target.device, dtype=target.dtype).clone()
        elif key in pair_keys:
            if tuple(value.shape) != (OLD_GRID * OLD_GRID, OLD_GRID * OLD_GRID) \
                    or tuple(target.shape) != (NODES, NODES):
                raise ValueError(f"unregistered pairwise source mapping for {key}")
            pp = p.to(value.device)
            converted = value.index_select(0, pp).index_select(1, pp)
            mapped[key] = converted.to(device=target.device, dtype=target.dtype).clone()
        else:
            if value.shape != target.shape:
                raise ValueError(f"unregistered shape change for {key}: {tuple(value.shape)} → {tuple(target.shape)}")
            mapped[key] = value.to(device=target.device, dtype=target.dtype).clone()
    return mapped


def geodesic_distance_chunked(z, euclidean, contrast, *, steps=3, radius=1.5,
                              temperature=.5, cap=16., chunk=ROW_CHUNK, log4=True):
    """Exact min-plus soft relaxation, checkpointed by output-row chunks.

    For each output pair(i,j), every intermediate node k remains in the
    log-sum-exp. Checkpointing the entire chunk relaxation bounds saved
    intermediates while preserving value and gradient semantics.
    """
    if z.ndim != 3 or euclidean.ndim not in (2, 3):
        raise ValueError("expected z[B,N,D] and euclidean[N,N] or [1|B,N,N]")
    batch, n, _ = z.shape
    if euclidean.ndim == 2:
        euclidean = euclidean.unsqueeze(0)
    if euclidean.shape[0] not in (1, batch) or tuple(euclidean.shape[1:]) != (n, n):
        raise ValueError("geodesic distance shape does not match embeddings")
    if not isinstance(chunk, int) or chunk < 1 or steps < 0 or temperature <= 0 or cap <= 0:
        raise ValueError("invalid geodesic relaxation settings")
    euclidean = euclidean.to(device=z.device, dtype=z.dtype)
    contrast_t = torch.as_tensor(contrast, device=z.device, dtype=z.dtype)
    similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1., 1.)
    step = euclidean * (1. + F.softplus(contrast_t) * (1. - similarity))
    distance = torch.where(euclidean <= radius, step, torch.full_like(step, cap))
    correction = temperature * math_log4(z) if log4 else 0.0

    for _ in range(steps):
        rows = []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            # Bind range/constants now: checkpoint recomputation must use this
            # exact row range, not the loop's eventual values.
            def relax_rows(value, lo=start, hi=stop, temp=temperature, corr=correction):
                # D[i,k] + D[k,j], retaining the complete intermediate axis k.
                left = value[:, lo:hi, :].unsqueeze(-1)  # [B,rows,k,1]
                right = value.unsqueeze(1)               # [B,1,k,j]
                candidates = left + right                # [B,rows,k,j]
                relaxed = -temp * torch.logsumexp(-candidates / temp, dim=2)
                relaxed = relaxed + corr
                return torch.minimum(value[:, lo:hi, :], relaxed)

            if torch.is_grad_enabled() and distance.requires_grad:
                rows.append(checkpoint(relax_rows, distance, use_reentrant=False))
            else:
                rows.append(relax_rows(distance))
        distance = torch.cat(rows, dim=1)
    return distance.clamp(max=cap)


def math_log4(reference):
    return torch.log(torch.tensor(4.0, device=reference.device, dtype=reference.dtype))


def attach_checkpointed_geodesic(graph, *, chunk=ROW_CHUNK, log4=True):
    """Attach the exact chunked relaxation without replacing graph parameters."""
    def method(self, z, euclidean):
        return geodesic_distance_chunked(
            z, euclidean, self.geodesic_contrast, steps=self.geodesic_steps,
            radius=self.geodesic_radius, temperature=self.geodesic_temperature,
            cap=self.geodesic_cap, chunk=chunk, log4=log4)
    graph._geodesic_distance = types.MethodType(method, graph)
    return graph


def build_core(source_state, *, device="cpu"):
    """Construct trainable native32 core and strictly load mapped source state."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for candidate in (str(root), str(root / "collaborative_test")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from SW_0094_aligned_joint_pilot.run import hparams
    from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

    hp = hparams("raw")
    hp.num_regions = NODES
    hp.k = float(NODES)
    hp.graph_top_k = 128
    hp.graph_spatial_grid_size = GRID
    hp.spike_spatial_grid_size = GRID
    hp.gamma_patch_grid_size = GRID
    hp.validate()
    core = S2NetCore(hp, device=device).to(device)
    converted = convert_state_dict(source_state, core.state_dict())
    core.load_state_dict(converted, strict=True)
    if core.graph_generator is None or core.graph_generator.top_k != 128:
        raise AssertionError("native32 candidate requires its registered learned graph")
    attach_checkpointed_geodesic(core.graph_generator, chunk=ROW_CHUNK, log4=True)
    # The 1024-grid graph is intentionally trainable in SW0135.
    if not any(parameter.requires_grad for parameter in core.graph_generator.parameters()):
        raise AssertionError("native32 graph parameters were unexpectedly frozen")
    return core
