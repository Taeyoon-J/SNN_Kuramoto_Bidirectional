import itertools

import torch
import torch.nn.functional as F


def spike_rhythm(
    spikes,
    threshold=0.8,
    min_group_size=2,
    return_all_groups=False,
    eps=1e-8,
):
    """
    Group oscillators by cosine similarity between their spike histories.

    Args:
        spikes:
            Tensor shaped [B, num_oscillators, T].
        threshold:
            Minimum cosine similarity for two oscillators to be considered
            rhythmically similar.
        min_group_size:
            Minimum number of oscillators in one object group.
        return_all_groups:
            If False, return maximal valid groups only.
            If True, also return smaller valid subgroups.
        eps:
            Numerical stability value for cosine similarity.

    Returns:
        List with length B. Each item is a list of tuples containing
        oscillator indices that form one object group.
    """
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, num_oscillators, T]. Use B=1 for one sample.")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2.")

    spikes = spikes.float()
    similarity = _pairwise_cosine_similarity(spikes, eps=eps)
    groups = [
        _find_similarity_groups(
            similarity[b],
            threshold=threshold,
            min_group_size=min_group_size,
            return_all_groups=return_all_groups,
        )
        for b in range(spikes.size(0))
    ]

    return groups


def spike_interval(
    core_out,
    interval_size,
    threshold=0.5,
    min_group_size=1,
    include_partial=True,
):
    """
    Detect active oscillators per temporal interval from membrane histories.

    Args:
        core_out:
            Tensor shaped [B, num_oscillators, T].
        interval_size:
            Number of time steps per interval.
        threshold:
            Oscillators whose interval-mean membrane value is greater than or
            equal to this threshold are treated as detecting an object in that
            interval.
        min_group_size:
            Minimum number of active oscillators required for an interval group
            to be returned. Use 1 to keep single-oscillator detections.
        include_partial:
            If True, include the final shorter interval when T is not divisible
            by interval_size. If False, discard it.

    Returns:
        List with length B. Each item is a list of unique tuples containing
        oscillator indices that form one object group.
    """
    if core_out.dim() != 3:
        raise ValueError("core_out must have shape [B, num_oscillators, T]. Use B=1 for one sample.")
    if interval_size <= 0:
        raise ValueError("interval_size must be positive.")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")

    core_out = core_out.float()
    intervals = _make_intervals(
        num_steps=core_out.size(2),
        interval_size=int(interval_size),
        include_partial=include_partial,
    )

    if not intervals:
        return [[] for _ in range(core_out.size(0))]

    interval_means = torch.stack(
        [
            core_out[:, :, start:end].mean(dim=2)
            for start, end in intervals
        ],
        dim=1,
    )
    active_mask = interval_means >= float(threshold)

    groups = []
    for batch_idx in range(core_out.size(0)):
        batch_groups = []
        seen_groups = set()
        for interval_idx in range(len(intervals)):
            active_indices = torch.nonzero(
                active_mask[batch_idx, interval_idx],
                as_tuple=False,
            ).flatten().tolist()
            group = tuple(active_indices)
            if len(group) >= int(min_group_size) and group not in seen_groups:
                batch_groups.append(group)
                seen_groups.add(group)
        groups.append(batch_groups)

    return groups


def spike_spatial_components(
    activity,
    patch_grid_size,
    threshold=0.5,
    min_group_size=2,
    activity_source="spikes",
    time_aggregate="max",
):
    """
    Detect object-like groups as spatial connected components on a patch grid.

    Args:
        activity:
            Tensor shaped [B, num_oscillators, T]. This can be binary spikes,
            membrane values, or sigmoid-normalized membrane activity.
        patch_grid_size:
            Integer grid size or (height, width). The product must equal
            num_oscillators.
        threshold:
            Active threshold after temporal aggregation.
        min_group_size:
            Minimum connected-component size.
        activity_source:
            Metadata only; accepted for API symmetry with visualization.
        time_aggregate:
            "max" marks a patch active if it is active at any time.
            "mean" marks a patch active by its mean activity over time.

    Returns:
        List with length B. Each item is a list of tuples containing oscillator
        indices for spatially contiguous active patch components.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, num_oscillators, T].")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be positive.")
    if activity_source not in {"spikes", "membrane", "sigmoid_membrane"}:
        raise ValueError('activity_source must be "spikes", "membrane", or "sigmoid_membrane".')
    if time_aggregate not in {"max", "mean"}:
        raise ValueError('time_aggregate must be "max" or "mean".')

    grid_h, grid_w = _parse_grid_size(patch_grid_size)
    if activity.size(1) != grid_h * grid_w:
        raise ValueError(
            f"activity has {activity.size(1)} oscillators, but patch grid "
            f"{grid_h}x{grid_w} has {grid_h * grid_w}."
        )

    activity = activity.float()
    if time_aggregate == "max":
        patch_scores = activity.max(dim=2).values
    else:
        patch_scores = activity.mean(dim=2)
    active = patch_scores >= float(threshold)

    groups = []
    for batch_idx in range(active.size(0)):
        mask = active[batch_idx].view(grid_h, grid_w)
        components = _spatial_components(mask)
        batch_groups = []
        for component in components:
            if len(component) < int(min_group_size):
                continue
            indices = tuple(sorted(row * grid_w + col for row, col in component))
            batch_groups.append(indices)
        groups.append(batch_groups)
    return groups


def _pairwise_cosine_similarity(spikes, eps=1e-8):
    left = spikes.unsqueeze(2)
    right = spikes.unsqueeze(1)
    return F.cosine_similarity(left, right, dim=-1, eps=eps)


def _make_intervals(num_steps, interval_size, include_partial):
    full_end = (num_steps // interval_size) * interval_size
    intervals = [
        (start, start + interval_size)
        for start in range(0, full_end, interval_size)
    ]
    if include_partial and full_end < num_steps:
        intervals.append((full_end, num_steps))
    return intervals


def _parse_grid_size(value):
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("patch_grid_size must be positive.")
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError("patch_grid_size values must be positive.")
        return height, width
    raise ValueError("patch_grid_size must be an int or a pair of ints.")


def _spatial_components(mask):
    height, width = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    components = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not bool(mask[row, col]):
                continue
            stack = [(row, col)]
            visited[row, col] = True
            component = []
            while stack:
                cur_row, cur_col = stack.pop()
                component.append((cur_row, cur_col))
                for next_row, next_col in (
                    (cur_row - 1, cur_col),
                    (cur_row + 1, cur_col),
                    (cur_row, cur_col - 1),
                    (cur_row, cur_col + 1),
                ):
                    if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                        continue
                    if visited[next_row, next_col] or not bool(mask[next_row, next_col]):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            components.append(component)
    return sorted(components, key=len, reverse=True)


def _find_similarity_groups(
    similarity,
    threshold,
    min_group_size,
    return_all_groups,
):
    adjacency = similarity >= float(threshold)
    adjacency.fill_diagonal_(False)

    maximal_groups = _maximal_cliques(adjacency)
    maximal_groups = [
        group for group in maximal_groups
        if len(group) >= int(min_group_size)
    ]

    if not return_all_groups:
        return maximal_groups

    all_groups = set()
    for group in maximal_groups:
        for size in range(int(min_group_size), len(group) + 1):
            all_groups.update(itertools.combinations(group, size))

    return sorted(all_groups, key=lambda item: (len(item), item))


def _maximal_cliques(adjacency):
    num_nodes = adjacency.size(0)
    neighbors = {
        node: set(torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist())
        for node in range(num_nodes)
    }

    cliques = []
    _bron_kerbosch(
        current=set(),
        candidates=set(range(num_nodes)),
        excluded=set(),
        neighbors=neighbors,
        cliques=cliques,
    )

    return [tuple(sorted(clique)) for clique in cliques]


def _bron_kerbosch(current, candidates, excluded, neighbors, cliques):
    if not candidates and not excluded:
        cliques.append(current)
        return

    for node in list(candidates):
        _bron_kerbosch(
            current=current | {node},
            candidates=candidates & neighbors[node],
            excluded=excluded & neighbors[node],
            neighbors=neighbors,
            cliques=cliques,
        )
        candidates.remove(node)
        excluded.add(node)
