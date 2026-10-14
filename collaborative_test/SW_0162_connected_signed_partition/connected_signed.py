"""Deterministic grid-connected agglomeration from signed Q/coupling evidence."""
from __future__ import annotations

import heapq
import math

import numpy as np


NATIVE_NODES = 1024
Q_EPS = 1e-6


def signed_pair_energy(q, adjacency, coupling_gain):
    """Build the registered symmetric pair energy from actual Q and source A."""
    q = np.asarray(q)
    adjacency = np.asarray(adjacency)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] < 1:
        raise ValueError("Q must be a nonempty square matrix")
    if adjacency.shape != q.shape:
        raise ValueError("source adjacency must match Q")
    if (not np.isfinite(q).all() or not np.isfinite(adjacency).all()
            or np.any(q < -1e-7) or np.any(q > 1 + 1e-7)):
        raise ValueError("Q and source adjacency must be finite; Q must lie in [0,1]")
    if not np.allclose(q, q.T, rtol=0, atol=2e-6):
        raise ValueError("Q must be symmetric")
    if (not np.isscalar(coupling_gain) or not math.isfinite(float(coupling_gain))
            or float(coupling_gain) <= 0):
        raise ValueError("source coupling gain must be finite and positive")

    q64 = q.astype(np.float64, copy=False)
    q_sym = 0.5 * (q64 + q64.T)
    q_bar = np.clip(q_sym, Q_EPS, 1.0 - Q_EPS)
    logit_q = np.log(q_bar / (1.0 - q_bar))
    a64 = adjacency.astype(np.float64, copy=False)
    energy = (1.0 + NATIVE_NODES * (a64 + a64.T) / (2.0 * float(coupling_gain))) * logit_q
    energy = 0.5 * (energy + energy.T)
    np.fill_diagonal(energy, 0.0)
    if not np.isfinite(energy).all():
        raise ValueError("signed pair energy became nonfinite")
    return energy


def _grid_neighbors(index, height, width):
    row, col = divmod(index, width)
    if row:
        yield index - width
    if row + 1 < height:
        yield index + width
    if col:
        yield index - 1
    if col + 1 < width:
        yield index + 1


def connected_signed_merge(q, adjacency, coupling_gain, *, grid_shape=(32, 32)):
    """Greedily merge adjacent components by maximum positive total cross energy.

    Each proposed merge sums *all* pair energies between the two components,
    including negative evidence. Ties use the pair of smallest patch indices.
    The algorithm stops when no adjacent merge has strictly positive gain.
    """
    q = np.asarray(q)
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(type(v) is not int or v <= 0 for v in grid_shape)):
        raise ValueError("grid_shape must contain two positive integers")
    height, width = grid_shape
    n = height * width
    if q.shape != (n, n):
        raise ValueError("Q shape does not match grid_shape")
    energy = signed_pair_energy(q, adjacency, coupling_gain)

    # Cross sums are updated by one row/column addition at each union. The
    # adjacency graph remains the original four-neighbour graph quotient.
    cross = energy.copy()
    active = set(range(n))
    members = {i: {i} for i in range(n)}
    neighbors = {i: set(_grid_neighbors(i, height, width)) for i in range(n)}
    versions = [0] * n
    heap = []

    def push(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        value = float(cross[a, b])
        if value > 0.0:
            heapq.heappush(heap, (-value, a, b, versions[a], versions[b]))

    for a in range(n):
        for b in neighbors[a]:
            if a < b:
                push(a, b)

    merge_count = 0
    while heap:
        neg_gain, a, b, va, vb = heapq.heappop(heap)
        if (a not in active or b not in active or versions[a] != va or versions[b] != vb
                or b not in neighbors[a]):
            continue
        gain = float(cross[a, b])
        if not gain > 0.0:
            continue
        keep, drop = a, b  # IDs are their original minimum patch indices.
        old_neighbors = (neighbors[keep] | neighbors[drop]) - {keep, drop}

        # Sum energy to every active component, not merely current neighbours;
        # a later union may bring a distant component into contact.
        cross[keep, :] += cross[drop, :]
        cross[:, keep] += cross[:, drop]
        cross[keep, keep] = 0.0
        cross[drop, :] = 0.0
        cross[:, drop] = 0.0

        members[keep].update(members.pop(drop))
        active.remove(drop)
        versions[keep] += 1
        versions[drop] += 1
        neighbors[keep] = set(old_neighbors)
        neighbors.pop(drop)
        for other in old_neighbors:
            neighbors[other].discard(keep)
            neighbors[other].discard(drop)
            neighbors[other].add(keep)

        for other in sorted(old_neighbors):
            push(keep, other)
        merge_count += 1

    labels = np.empty(n, dtype=np.int32)
    clusters = []
    for representative in sorted(active):
        component = members[representative]
        labels[list(component)] = representative
        clusters.append({"representative": representative, "size": len(component),
                         "patches": sorted(component)})
    return {"labels": labels.reshape(height, width), "clusters": clusters,
            "merge_count": merge_count, "pair_energy": energy}


def canonical_labels(component_labels, *, min_size=8):
    """Largest component is background; tiny components fold into it.

    Equal-size largest components break ties by their smallest patch index.
    Remaining foreground IDs are ordered by their smallest patch index.
    """
    raw = np.asarray(component_labels)
    if raw.ndim != 2 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("component labels must be a nonempty integer grid")
    if type(min_size) is not int or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    flat = raw.reshape(-1)
    ids, counts = np.unique(flat, return_counts=True)
    first = {int(label): int(np.flatnonzero(flat == label)[0]) for label in ids}
    background = min((int(label) for label, count in zip(ids, counts)),
                     key=lambda label: (-int(np.count_nonzero(flat == label)), first[label]))
    sizes = {int(label): int(count) for label, count in zip(ids, counts)}
    foreground = sorted((int(label) for label in ids if int(label) != background
                         and sizes[int(label)] >= min_size), key=lambda label: first[label])
    output = np.zeros(flat.shape, dtype=np.int32)
    for new_id, label in enumerate(foreground, start=1):
        output[flat == label] = new_id
    return output.reshape(raw.shape)
