"""Opt-in coordinate augmentation for the learned graph node embedding.

The legacy graph is held as a child module and its forward calculation is copied
with one addition: a zero-initialized linear projection of normalized XY patch
centers is added before the existing normalization. No S2Net or shared graph
source file is modified.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class XYGraphAdapter(nn.Module):
    def __init__(self, legacy_graph: nn.Module, grid_size: int = 16):
        super().__init__()
        if int(grid_size) != 16 or legacy_graph.projection.out_features != 16:
            raise ValueError("registered SW0110 requires a 16x16 grid and 16-d node embedding")
        self.base_graph = legacy_graph
        axis = torch.linspace(-1.0, 1.0, int(grid_size), dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("xy", torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1))
        self.xy_projection = nn.Parameter(torch.zeros(16, 2))

    @property
    def uses_feedback(self):
        return self.base_graph.uses_feedback

    def initial_alignment(self, batch_size, num_nodes, device):
        return self.base_graph.initial_alignment(batch_size, num_nodes, device)

    def update_alignment(self, alignment, theta):
        return self.base_graph.update_alignment(alignment, theta)

    def forward(self, gamma, alignment=None):
        graph = self.base_graph
        if (gamma.size(1) == graph.projection.in_features
                and gamma.size(2) != graph.projection.in_features):
            gamma = gamma.transpose(1, 2)

        xy = self.xy.to(device=gamma.device, dtype=gamma.dtype)
        node_xy = F.linear(xy, self.xy_projection.to(dtype=gamma.dtype))
        z = F.normalize(graph.projection(gamma) + node_xy.unsqueeze(0), dim=-1)
        logits = torch.bmm(z, z.transpose(1, 2)) / graph.log_temperature.exp().clamp_min(1e-3)

        if graph.spatial_rate is not None:
            rate = F.softplus(graph.spatial_rate)
            distance = graph.grid_distance.unsqueeze(0)
            if graph.geodesic_steps > 0:
                distance = graph._geodesic_distance(z, distance)
            logits = logits - rate * distance

        if graph.feedback_strength is not None and alignment is not None:
            logits = logits + graph.feedback_strength * alignment

        num_nodes = logits.size(-1)
        values, indices = logits.topk(min(graph.top_k, num_nodes), dim=-1)
        weights = values.softmax(dim=-1) * graph.log_coupling_gain.exp()
        adjacency = torch.zeros_like(logits).scatter_(-1, indices, weights)
        return 0.5 * (adjacency + adjacency.transpose(1, 2))


def attach_xy_graph(core, grid_size: int = 16):
    """Replace only the core's graph module, retaining its exact initialized weights."""
    legacy = core.graph_generator
    if legacy is None:
        raise ValueError("SW0110 requires the registered learned graph")
    reference = legacy.projection.weight
    adapter = XYGraphAdapter(legacy, grid_size=grid_size).to(
        device=reference.device, dtype=reference.dtype)
    core.graph_generator = adapter
    return core
