
if __package__:
    from . import error_bound
else:
    import error_bound

import itertools
import math

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
    error_bound.validate_spike_classifier_spike_rhythm(spikes, min_group_size)

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
    error_bound.validate_spike_classifier_spike_interval(core_out, interval_size, min_group_size)

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
    error_bound.validate_spike_classifier_spike_spatial_components(activity, min_group_size, activity_source, time_aggregate)

    grid_h, grid_w = _parse_grid_size(patch_grid_size)
    error_bound.validate_spike_classifier_spike_spatial_components_2(grid_h, grid_w, activity)

    activity = activity.float()
    if time_aggregate == "max":
        patch_scores = activity.max(dim=2).values
    else:  # readout mode: mean (README 4.3 training default)
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
        error_bound.validate_loss_function_parse_grid_size(value)
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        error_bound.validate_loss_function_parse_grid_size_2(height, width)
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


def spatial_gaussian_kernel(num_nodes, grid_size, sigma, device=None, dtype=None):
    """Return G_ij=exp(-d(i,j)^2/(2 sigma^2)) on a regular patch grid."""
    if isinstance(grid_size, int):
        height = width = int(grid_size)
    else:
        if len(grid_size) != 2:
            raise ValueError("grid_size must be an int or (height, width).")
        height, width = int(grid_size[0]), int(grid_size[1])
    if height <= 0 or width <= 0 or height * width != int(num_nodes):
        raise ValueError("grid_size dimensions must be positive and match num_nodes.")
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("spatial sigma must be positive.")
    dtype = dtype or torch.float32
    if math.isinf(sigma):
        return torch.ones(num_nodes, num_nodes, device=device, dtype=dtype)
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    coordinates = torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1)
    distance_squared = torch.cdist(coordinates, coordinates).square()
    return torch.exp(-distance_squared / (2.0 * sigma * sigma))


def permute_spatial_kernel(kernel, permutation):
    """Apply one node permutation to both axes of a square spatial kernel."""
    if kernel.dim() != 2 or kernel.size(0) != kernel.size(1):
        raise ValueError("kernel must be a square [N, N] tensor.")
    permutation = torch.as_tensor(permutation)
    if permutation.dim() != 1 or permutation.numel() != kernel.size(0):
        raise ValueError("permutation must contain exactly N node indices.")
    if permutation.dtype == torch.bool or permutation.is_floating_point():
        raise ValueError("permutation indices must be integers.")
    permutation = permutation.to(device=kernel.device, dtype=torch.long)
    expected = torch.arange(kernel.size(0), device=kernel.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("permutation must contain every node index exactly once.")
    return kernel.index_select(0, permutation).index_select(1, permutation)


def seeded_spatial_permutation(num_nodes, seed):
    """Return a reproducible node permutation independent of CUDA RNG state."""
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError("spatial permutation seed must be a non-negative integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(num_nodes), generator=generator)


def spike_synchrony_affinity(activity, components=None, settle=0, eps=1e-8,
                             spatial_sigma=None, spatial_grid_size=16,
                             affinity_mode="spike",
                             spatial_permutation_seed=None):
    """Build spike synchrony S, S*Gaussian, or Gaussian-only affinity."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if affinity_mode not in {"spike", "spike_binary", "spike_spatial", "spike_spatial_permuted", "spatial_only"}:
        raise ValueError("Unknown affinity_mode.")
    spatial_modes = {"spike_spatial", "spike_spatial_permuted", "spatial_only"}
    if affinity_mode in spatial_modes and spatial_sigma is None:
        raise ValueError("spatial_sigma is required for spatial affinity modes.")
    if affinity_mode == "spike_spatial_permuted" and spatial_permutation_seed is None:
        raise ValueError("spatial_permutation_seed is required for permuted spatial affinity.")
    activity = activity.float()
    if int(settle) > 0:
        activity = activity[:, :, int(settle):]
        if components is not None:
            components = components[..., int(settle):]

    def correlate(x):
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return (x @ x.transpose(-1, -2)).clamp(-1.0, 1.0)

    kernel = None
    if affinity_mode in spatial_modes:
        kernel = spatial_gaussian_kernel(
            activity.size(1), spatial_grid_size, spatial_sigma,
            device=activity.device, dtype=activity.dtype,
        )
    if affinity_mode == "spike_spatial_permuted":
        permutation = seeded_spatial_permutation(activity.size(1), spatial_permutation_seed)
        kernel = permute_spatial_kernel(kernel, permutation)
    if affinity_mode == "spatial_only":
        return kernel.unsqueeze(0).expand(activity.size(0), -1, -1)
    if affinity_mode == "spike_binary" and components is not None:
        # Ablation control: retain only whether each component crossed its
        # membrane threshold, then use the same per-component correlation rule.
        components = (components != 0).to(dtype=activity.dtype)
    if components is None:
        similarity = correlate(activity)
    else:
        if components.dim() != 4 or components.size(0) != activity.size(0) \
                or components.size(2) != activity.size(1):
            raise ValueError("components must have shape [B, D, N, T].")
        per = [correlate(components[:, d].float()) for d in range(components.size(1))]
        similarity = torch.stack(per).clamp_min(0.0).prod(dim=0)
    if affinity_mode in {"spike_spatial", "spike_spatial_permuted"}:
        similarity = similarity * kernel.unsqueeze(0)
    return similarity


def spike_synchrony_components(
    activity,
    foreground_threshold=0.5,
    synchrony_threshold=0.5,
    min_group_size=1,
    settle=0,
    components=None,
    background="largest_component",
    synchrony_quantile=0.35,
    target_foreground=None,
    eps=1e-8,
    spatial_sigma=None,
    spatial_grid_size=16,
    affinity_mode="spike",
    spatial_permutation_seed=None,
):
    """
    Group oscillators that spike together, as connected components of synchrony.

    This is the readout the architecture names: units firing together are one
    object. It differs from ``spike_spatial_components``, which thresholds
    activity and then takes spatially connected components -- that groups two
    touching objects into one however differently they fire, and uses no
    synchrony at all.

    Nothing here needs the true object count. The number of groups falls out of
    the connectivity, and both thresholds are chosen on validation, so the output
    is a prediction rather than an oracle result.

    Args:
        activity: [B, N, T] spikes or membrane. The synchrony between two units
            is the correlation of their traces.
        background: how background is decided. "largest_component" calls the
            biggest synchrony component background, which is what the data says:
            ranked by mean PLV, the least-synchronised patches recover the
            foreground at IoU 0.503 against 0.062 for chance, because the ~230
            background patches form one enormous synchronous mass and objects are
            what fails to join it. "activity" uses foreground_threshold on the
            mean firing rate instead, which is near useless -- that ranking
            recovers the foreground at 0.146.
        foreground_threshold: only used when background="activity".
        synchrony_quantile: with background="hybrid", the fraction of units the
            synchrony ranking lets through as candidate foreground before the
            component rule is applied. Chosen on validation.
        target_foreground: when set, the synchrony threshold is calibrated per
            image so the groups cover about this fraction of units, and
            synchrony_threshold is ignored. A fixed threshold lands in a
            different place for every model: across three seeds of one setting
            the predicted foreground came out at 0.179, 0.363 and 0.312 of
            patches, and foreground IoU tracked that inversely -- 0.520, 0.214,
            0.363 -- so most of the seed variance was the threshold, not the
            model. The fraction is chosen on validation, never from an image's
            own labels, so this stays a prediction.
        synchrony_threshold: two foreground units are linked above this.
        min_group_size: components smaller than this are dropped to background.
        settle: leading steps to discard as transient.
        components: optional [B, D, N, T] per-component activity. When given,
            synchrony is measured per component and combined by product, which is
            what the phase readout does and is worth 0.202 foreground ARI there.

    Returns:
        List of length B; each item is a list of tuples of oscillator indices,
        disjoint, ordered by descending size. Matches the input
        ``spatial_components_to_patch_labels`` expects.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if not 0.0 <= float(synchrony_threshold) <= 1.0:
        raise ValueError("synchrony_threshold must lie in [0, 1].")
    if affinity_mode not in {"spike", "spike_binary", "spike_spatial", "spike_spatial_permuted", "spatial_only"}:
        raise ValueError("Unknown affinity_mode.")
    if affinity_mode in {"spike_spatial", "spike_spatial_permuted", "spatial_only"} and spatial_sigma is None:
        raise ValueError("spatial_sigma is required for spatial affinity modes.")

    activity = activity.float()
    if int(settle) > 0:
        activity = activity[:, :, int(settle):]
        if components is not None:
            components = components[..., int(settle):]
    similarity = spike_synchrony_affinity(
        activity, components=components, settle=0, eps=eps,
        spatial_sigma=spatial_sigma, spatial_grid_size=spatial_grid_size,
        affinity_mode=affinity_mode,
        spatial_permutation_seed=spatial_permutation_seed,
    )

    if background not in {"largest_component", "activity", "hybrid"}:
        raise ValueError('background must be "largest_component", "activity" or "hybrid".')
    if background == "activity":
        foreground = activity.mean(dim=-1) >= float(foreground_threshold)
    elif background == "hybrid":
        # Two signals, kept separate because they carry different information.
        # Ranked by mean synchrony to everything else, the least synchronised
        # patches recover the true foreground at IoU 0.400 against 0.062 for
        # chance -- the background is one large synchronous mass and objects are
        # what fails to join it. That is a per-unit score; the component rule
        # below reads structure instead and on its own reaches 0.41 to 0.55. A
        # unit has to pass both: quiet enough in the ranking, and outside the
        # biggest component.
        degree = similarity.mean(dim=-1)
        cut = torch.quantile(degree, float(synchrony_quantile), dim=-1, keepdim=True)
        foreground = degree <= cut
    else:
        foreground = torch.ones(activity.shape[:2], dtype=torch.bool,
                                device=activity.device)
    def components_at(b, threshold):
        """Groups for image b at one threshold, background already removed."""
        active = foreground[b]
        adjacency = (similarity[b] >= threshold) & active.unsqueeze(1) & active.unsqueeze(0)
        seen = torch.zeros(activity.size(1), dtype=torch.bool)
        found = []
        for start in range(activity.size(1)):
            if seen[start] or not bool(active[start]):
                continue
            frontier, member = [start], [start]
            seen[start] = True
            while frontier:
                node = frontier.pop()
                for n in (adjacency[node] & ~seen).nonzero(as_tuple=True)[0].tolist():
                    seen[n] = True
                    member.append(n)
                    frontier.append(n)
            if len(member) >= int(min_group_size):
                found.append(tuple(sorted(member)))
        found.sort(key=len, reverse=True)
        if background in {"largest_component", "hybrid"} and found:
            found = found[1:]
        return found

    if target_foreground is not None:
        goal = float(target_foreground) * activity.size(1)
        out = []
        for b in range(activity.size(0)):
            low, high, best = 0.0, 1.0, None
            tolerance = max(1.0, 0.1 * goal)
            # the covered count falls as the threshold drops, because more units
            # join the one component that is then called background. Seven rounds
            # resolve the threshold to about 1/128, and the search stops early
            # once it is within a tenth of the goal -- each round is a BFS per
            # image, so this is the cost of the whole evaluation.
            for _ in range(7):
                mid = 0.5 * (low + high)
                found = components_at(b, mid)
                covered = sum(len(g) for g in found)
                gap = abs(covered - goal)
                if best is None or gap < best[0]:
                    best = (gap, found)
                if gap <= tolerance:
                    break
                if covered > goal:
                    high = mid
                else:
                    low = mid
            out.append(best[1])
        return out

    linked = similarity >= float(synchrony_threshold)

    out = []
    for b in range(activity.size(0)):
        active = foreground[b]
        adjacency = linked[b] & active.unsqueeze(1) & active.unsqueeze(0)
        seen = torch.zeros(activity.size(1), dtype=torch.bool)
        groups = []
        for start in range(activity.size(1)):
            if seen[start] or not bool(active[start]):
                continue
            # breadth-first over the synchrony graph
            frontier = [start]
            seen[start] = True
            member = [start]
            while frontier:
                node = frontier.pop()
                neighbours = (adjacency[node] & ~seen).nonzero(as_tuple=True)[0]
                for n in neighbours.tolist():
                    seen[n] = True
                    member.append(n)
                    frontier.append(n)
            if len(member) >= int(min_group_size):
                groups.append(tuple(sorted(member)))
        groups.sort(key=len, reverse=True)
        if background in {"largest_component", "hybrid"} and groups:
            # the biggest synchronous mass is the background, and drops out
            groups = groups[1:]
        out.append(groups)
    return out
