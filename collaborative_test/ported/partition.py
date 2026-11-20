from __future__ import annotations
import heapq
import numpy as np
import signed_energy as signed

def _neighbors(index: int, height: int, width: int):
    row, col = divmod(index, width)
    if row: yield index - width
    if row + 1 < height: yield index + width
    if col: yield index - 1
    if col + 1 < width: yield index + 1


def foreground_induced_merge(q, source_a, coupling_gain, source_labels, *,
                             grid_shape=(32, 32), native_nodes=None):
    """Merge only positive-evidence, four-connected groups inside source FG.

    The full native Q/A energy is constructed before inducing the Source-T
    foreground subgraph. Every foreground node survives, including singleton
    and sub-eight-patch groups; label 0 remains exactly the Source-T background.
    """
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(type(v) is not int or v <= 0 for v in grid_shape)):
        raise ValueError("grid_shape must contain two positive integers")
    h, w = grid_shape
    n = h * w
    labels = np.asarray(source_labels)
    if labels.shape != (h, w) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Source-T labels must be an integer grid matching grid_shape")
    if labels.size == 0 or np.any(labels < 0):
        raise ValueError("Source-T labels must be nonempty and nonnegative")
    q = np.asarray(q)
    a = np.asarray(source_a)
    if q.shape != (n, n) or a.shape != (n, n):
        raise ValueError("Q and actual source A must be full native-grid matrices")

    # Reuse the registered full-N operator exactly; do not recompute a reduced
    # foreground oscillator or renormalize graph weights.
    energy = signed.signed_pair_energy(q, a, coupling_gain, native_nodes=native_nodes)
    foreground = np.flatnonzero(labels.reshape(-1) > 0).astype(np.int64)
    out = np.zeros(n, dtype=np.int32)
    if foreground.size == 0:
        return {"labels": out.reshape(h, w), "clusters": [], "merge_count": 0,
                "merge_log": [], "foreground_support": foreground,
                "pair_energy": energy, "discarded_mass": 0}

    cross = energy.copy()
    active = set(int(i) for i in foreground)
    members = {int(i): {int(i)} for i in foreground}
    fset = active.copy()
    neighbors = {i: {j for j in _neighbors(i, h, w) if j in fset} for i in active}
    versions = [0] * n
    heap = []

    def push(x, y):
        if x == y: return
        if x > y: x, y = y, x
        value = float(cross[x, y])
        if value > 0.0:
            heapq.heappush(heap, (-value, x, y, versions[x], versions[y]))

    for x in sorted(active):
        for y in sorted(neighbors[x]):
            if x < y: push(x, y)

    merge_log = []
    while heap:
        neg_gain, x, y, vx, vy = heapq.heappop(heap)
        if (x not in active or y not in active or versions[x] != vx or versions[y] != vy
                or y not in neighbors[x]):
            continue
        gain = float(cross[x, y])
        if not gain > 0.0: continue
        keep, drop = x, y  # IDs are minimum original patch indices.
        touching = (neighbors[keep] | neighbors[drop]) - {keep, drop}
        cross[keep, :] += cross[drop, :]
        cross[:, keep] += cross[:, drop]
        cross[keep, keep] = 0.0
        cross[drop, :] = 0.0
        cross[:, drop] = 0.0
        members[keep].update(members.pop(drop))
        active.remove(drop)
        versions[keep] += 1
        versions[drop] += 1
        neighbors[keep] = set(touching)
        neighbors.pop(drop)
        for other in touching:
            neighbors[other].discard(keep)
            neighbors[other].discard(drop)
            neighbors[other].add(keep)
        for other in sorted(touching): push(keep, other)
        merge_log.append({"patch_pair": [x, y], "gain": gain})

    groups = [members[rep] for rep in sorted(active)]
    groups.sort(key=min)
    clusters = []
    for label_id, patches in enumerate(groups, start=1):
        ordered = sorted(patches)
        out[ordered] = label_id
        clusters.append({"label": label_id, "representative": ordered[0],
                         "size": len(ordered), "patches": ordered})
    if not np.array_equal(out > 0, labels.reshape(-1) > 0):
        raise AssertionError("foreground support changed during induced partition")
    return {"labels": out.reshape(h, w), "clusters": clusters,
            "merge_count": len(merge_log), "merge_log": merge_log,
            "foreground_support": foreground, "pair_energy": energy,
            "discarded_mass": 0}
