"""Pair-count maximin and matched local-edge losses for SW0168."""
from __future__ import annotations

import numpy as np
import torch


def _inputs(q, target, support):
    if not torch.is_tensor(q) or q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] == 0:
        raise ValueError("Q must be a nonempty square tensor")
    if not q.is_floating_point() or not torch.isfinite(q).all():
        raise ValueError("Q must be finite floating point")
    if torch.any(q < 0) or torch.any(q > 1):
        raise ValueError("native positive-clamped Q must lie in [0,1]")
    if not torch.allclose(q, q.T, rtol=0, atol=2e-6):
        raise ValueError("Q must be symmetric")
    target_array = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    t = np.asarray(target_array)
    f = np.asarray(support.detach().cpu() if torch.is_tensor(support) else support)
    if t.ndim != 2 or f.shape != t.shape or t.size != q.shape[0]:
        raise ValueError("target and support must match Q's original grid")
    if not np.issubdtype(t.dtype, np.integer) or np.any(t < 0):
        raise ValueError("target IDs must be nonnegative integers")
    if f.dtype != np.bool_ and not np.isin(f, (0, 1)).all():
        raise ValueError("support must be boolean")
    f = f.astype(bool, copy=False).reshape(-1)
    t = t.astype(np.int64, copy=False).reshape(-1)
    if np.any(t[~f] != 0):
        raise ValueError("target labels outside fixed support must be zero")
    return t, f, tuple(map(int, target_array.shape))


def _grid_edges(support, shape):
    h, w = shape
    edges = []
    for i in range(h * w):
        if not support[i]:
            continue
        r, c = divmod(i, w)
        if c + 1 < w and support[i + 1]:
            edges.append((i, i + 1))
        if r + 1 < h and support[i + w]:
            edges.append((i, i + w))
    return edges


def _support_components(nodes, edges):
    dsu = _DSU(nodes)
    for a, b in edges:
        dsu.union(a, b)
    return {node: dsu.find(node) for node in nodes}


class _DSU:
    def __init__(self, nodes, labels=None):
        labels = {} if labels is None else labels
        self.parent = {int(i): int(i) for i in nodes}
        self.size = {int(i): 1 for i in nodes}
        self.hist = {int(i): {int(labels.get(i, i)): 1} for i in nodes}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return None
        # Stable root selection keeps edge/tie behavior reproducible.
        if self.size[a] < self.size[b] or (self.size[a] == self.size[b] and a > b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        for label, count in self.hist[b].items():
            self.hist[a][label] = self.hist[a].get(label, 0) + count
        del self.size[b], self.hist[b]
        return a, b


def forest_routes(q, target, support):
    """Return detached edge routes and exact represented unordered-pair mass."""
    t, f, shape = _inputs(q, target, support)
    nodes = np.flatnonzero(f).tolist()
    edge_list = _grid_edges(f, shape)
    q_cpu = q.detach().cpu().double().numpy()
    component = _support_components(nodes, edge_list)
    label_component = {}
    for node in nodes:
        label = int(t[node])
        previous = label_component.setdefault(label, component[node])
        if previous != component[node]:
            raise ValueError("feasible target IDs must be 4-connected within fixed support")

    # Positive maximin forest is constrained within feasible target IDs.
    pos_dsu = _DSU(nodes)
    pos_edges = []
    pos_mass = 0
    same_edges = [(a, b) for a, b in edge_list if t[a] == t[b]]
    for a, b in sorted(same_edges, key=lambda e: (-float(q_cpu[e]), e[0], e[1])):
        ra, rb = pos_dsu.find(a), pos_dsu.find(b)
        if ra == rb:
            continue
        weight = pos_dsu.size[ra] * pos_dsu.size[rb]
        pos_dsu.union(ra, rb)
        pos_edges.append((a, b, weight))
        pos_mass += weight
    first_positive_root = {}
    for node in nodes:
        label = int(t[node])
        root = pos_dsu.find(node)
        if label in first_positive_root and first_positive_root[label] != root:
            raise ValueError("feasible target IDs must be 4-connected within fixed support")
        first_positive_root[label] = root

    # Pre-union each target class, then route cross-target pairs through the
    # descending-Q edge forest. Histogram overlap excludes same-target pairs.
    neg_dsu = _DSU(nodes, {node: int(t[node]) for node in nodes})
    first = {}
    for node in nodes:
        first.setdefault(int(t[node]), node)
        if first[int(t[node])] != node:
            neg_dsu.union(first[int(t[node])], node)
    neg_edges = []
    neg_mass = 0
    diff_edges = [(a, b) for a, b in edge_list if t[a] != t[b]]
    for a, b in sorted(diff_edges, key=lambda e: (-float(q_cpu[e]), e[0], e[1])):
        ra, rb = neg_dsu.find(a), neg_dsu.find(b)
        if ra == rb:
            continue
        ha, hb = neg_dsu.hist[ra], neg_dsu.hist[rb]
        represented = neg_dsu.size[ra] * neg_dsu.size[rb] - sum(
            count * hb.get(label, 0) for label, count in ha.items())
        roots = neg_dsu.union(ra, rb)
        if represented:
            neg_edges.append((a, b, int(represented)))
            neg_mass += int(represented)
    eligible_negative = 0
    for group_id in set(component.values()):
        group_nodes = [node for node in nodes if component[node] == group_id]
        counts = {}
        for node in group_nodes:
            counts[int(t[node])] = counts.get(int(t[node]), 0) + 1
        eligible_negative += len(group_nodes) * (len(group_nodes) - 1) // 2 - sum(
            count * (count - 1) // 2 for count in counts.values())
    total_counts = {}
    for node in nodes:
        total_counts[int(t[node])] = total_counts.get(int(t[node]), 0) + 1
    all_negative = len(nodes) * (len(nodes) - 1) // 2 - sum(
        count * (count - 1) // 2 for count in total_counts.values())
    return {"positive_edges": pos_edges, "positive_pair_mass": int(pos_mass),
            "negative_edges": neg_edges, "negative_pair_mass": int(neg_mass),
            "excluded_unreachable_negative_pair_mass": int(all_negative - eligible_negative),
            "local_edges": [(a, b, int(t[a] == t[b])) for a, b in edge_list],
            "support_nodes": len(nodes), "grid_shape": list(shape)}


def _edge_values(q, edges):
    if not edges:
        return q.reshape(-1)[:0], torch.empty(0, dtype=torch.float64, device=q.device)
    ij = torch.as_tensor([(a, b) for a, b, _ in edges], dtype=torch.long, device=q.device)
    counts = torch.as_tensor([w for _, _, w in edges], dtype=torch.float64, device=q.device)
    return q[ij[:, 0], ij[:, 1]], counts


def constrained_maximin_loss(q, target, support):
    """Pair-count weighted positive and negative constrained maximin loss."""
    plan = forest_routes(q, target, support)
    terms = []
    if plan["positive_pair_mass"]:
        values, weights = _edge_values(q, plan["positive_edges"])
        terms.append((weights * (1.0 - values).square()).sum() / plan["positive_pair_mass"])
    if plan["negative_pair_mass"]:
        values, weights = _edge_values(q, plan["negative_edges"])
        terms.append((weights * values.square()).sum() / plan["negative_pair_mass"])
    if not terms:
        return q.sum() * 0.0
    return torch.stack(terms).mean()


def matched_local_edge_loss(q, target, support):
    """Matched local 4-neighbor class-balanced edge objective."""
    plan = forest_routes(q, target, support)
    edges = plan["local_edges"]
    terms = []
    for same in (1, 0):
        selected = [(a, b, 1) for a, b, label in edges if label == same]
        if not selected:
            continue
        values, _ = _edge_values(q, selected)
        terms.append((1.0 - values).square().mean() if same else values.square().mean())
    if not terms:
        return q.sum() * 0.0
    return torch.stack(terms).mean()
