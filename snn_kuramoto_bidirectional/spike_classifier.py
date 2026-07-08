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
