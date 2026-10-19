import torch


def hierarchical_spike_match(
    level_spikes,
    grid_sizes,
    threshold=0.5,
    min_group_size=1,
    require_all_parents=True,
    time_aggregate="any",
):
    """
    Group fine-level oscillators by matching their spikes with coarser parents.

    Args:
        level_spikes:
            Sequence of tensors ordered from lowest/fine level to
            highest/coarse level. Each tensor must have shape [B, N, T].
        grid_sizes:
            Sequence of grid sizes ordered the same way as level_spikes.
            Example: [(4, 4), (2, 2)].
        threshold:
            Values greater than or equal to this threshold are treated as spikes.
        min_group_size:
            Minimum number of lowest-level oscillators in a returned group.
        require_all_parents:
            If True, every coarser parent level must spike at the same time.
            If False, at least one coarser parent level is enough.
        time_aggregate:
            "any" returns groups that appear at any time step.
            "per_time" returns one group list per time step.

    Returns:
        List with length B. For time_aggregate="any", each item is a list of
        dictionaries:

            {
                "fine_indices": (...),
                "level_indices": ((...), (...), ...),
                "times": (...),
            }

        "fine_indices" are oscillator indices from the lowest/fine level.
        "level_indices" stores the matched oscillator indices at every level.

    Example:
        With grid_sizes [(4, 4), (2, 2)], zero-based coarse index 0 covers
        fine indices (0, 1, 4, 5). In one-based notation, coarse cell 1 covers
        fine cells 1, 2, 5, 6.
    """
    _validate_inputs(level_spikes, grid_sizes, min_group_size, time_aggregate)

    binary_levels = [(spikes.float() >= float(threshold)) for spikes in level_spikes]
    batch_size = binary_levels[0].size(0)
    num_steps = binary_levels[0].size(2)
    parent_maps = build_parent_index_maps(grid_sizes)

    results = []
    for batch_idx in range(batch_size):
        if time_aggregate == "per_time":
            batch_result = []
            for time_idx in range(num_steps):
                batch_result.append(
                    _groups_for_time(
                        binary_levels=binary_levels,
                        parent_maps=parent_maps,
                        batch_idx=batch_idx,
                        time_idx=time_idx,
                        min_group_size=min_group_size,
                        require_all_parents=require_all_parents,
                    )
                )
        else:
            by_hierarchy = {}
            for time_idx in range(num_steps):
                for group in _groups_for_time(
                    binary_levels=binary_levels,
                    parent_maps=parent_maps,
                    batch_idx=batch_idx,
                    time_idx=time_idx,
                    min_group_size=min_group_size,
                    require_all_parents=require_all_parents,
                ):
                    key = group["level_indices"]
                    if key not in by_hierarchy:
                        by_hierarchy[key] = {
                            "fine_indices": set(),
                            "level_indices": key,
                            "times": set(),
                        }
                    by_hierarchy[key]["fine_indices"].update(group["fine_indices"])
                    by_hierarchy[key]["times"].add(int(time_idx))

            batch_result = [
                {
                    "fine_indices": tuple(sorted(value["fine_indices"])),
                    "level_indices": value["level_indices"],
                    "times": tuple(sorted(value["times"])),
                }
                for value in by_hierarchy.values()
                if len(value["fine_indices"]) >= int(min_group_size)
            ]
            batch_result.sort(
                key=lambda group: (
                    -len(group["fine_indices"]),
                    group["level_indices"],
                    group["times"],
                )
            )
        results.append(batch_result)

    return results


def build_parent_index_maps(grid_sizes):
    """
    Build fine-to-parent index maps from the first grid to each coarser grid.

    Returns:
        List of tensors. The first tensor is identity for the fine grid. Each
        later tensor has shape [fine_num_cells] and maps a fine index to its
        parent index in that level.
    """
    parsed = [_parse_grid_size(size) for size in grid_sizes]
    fine_h, fine_w = parsed[0]
    fine_rows = torch.arange(fine_h).repeat_interleave(fine_w)
    fine_cols = torch.arange(fine_w).repeat(fine_h)

    maps = [torch.arange(fine_h * fine_w, dtype=torch.long)]
    for grid_h, grid_w in parsed[1:]:
        parent_rows = torch.div(fine_rows * grid_h, fine_h, rounding_mode="floor")
        parent_cols = torch.div(fine_cols * grid_w, fine_w, rounding_mode="floor")
        parent_rows = parent_rows.clamp(max=grid_h - 1)
        parent_cols = parent_cols.clamp(max=grid_w - 1)
        maps.append((parent_rows * grid_w + parent_cols).long())
    return maps


def fine_indices_for_parent(fine_grid_size, parent_grid_size, parent_index):
    """
    Return fine-grid indices covered by a parent cell.

    This is mostly a debugging helper for checking hierarchical alignment.
    """
    maps = build_parent_index_maps([fine_grid_size, parent_grid_size])
    parent_index = int(parent_index)
    return tuple(torch.nonzero(maps[1] == parent_index, as_tuple=False).flatten().tolist())


def _groups_for_time(
    binary_levels,
    parent_maps,
    batch_idx,
    time_idx,
    min_group_size,
    require_all_parents,
):
    fine_active = torch.nonzero(
        binary_levels[0][batch_idx, :, time_idx],
        as_tuple=False,
    ).flatten()
    groups_by_parents = {}

    for fine_index in fine_active.tolist():
        hierarchy = tuple(int(parent_map[fine_index].item()) for parent_map in parent_maps)
        parent_matches = []
        for level_idx in range(1, len(binary_levels)):
            parent_index = hierarchy[level_idx]
            is_active = bool(binary_levels[level_idx][batch_idx, parent_index, time_idx])
            parent_matches.append(is_active)

        if parent_matches:
            matched = all(parent_matches) if require_all_parents else any(parent_matches)
            if not matched:
                continue

        parent_hierarchy = hierarchy[1:]
        groups_by_parents.setdefault(parent_hierarchy, []).append(int(fine_index))

    groups = []
    for parent_hierarchy, fine_indices in groups_by_parents.items():
        if len(fine_indices) < int(min_group_size):
            continue
        first_fine_index = sorted(fine_indices)[0]
        hierarchy = tuple(int(parent_map[first_fine_index].item()) for parent_map in parent_maps)
        groups.append(
            {
                "fine_indices": tuple(sorted(fine_indices)),
                "level_indices": hierarchy,
                "times": (int(time_idx),),
            }
        )
    groups.sort(key=lambda group: (-len(group["fine_indices"]), group["level_indices"]))
    return groups


def _validate_inputs(level_spikes, grid_sizes, min_group_size, time_aggregate):
    if not isinstance(level_spikes, (list, tuple)) or len(level_spikes) < 2:
        raise ValueError("level_spikes must contain at least two levels.")
    if not isinstance(grid_sizes, (list, tuple)) or len(grid_sizes) != len(level_spikes):
        raise ValueError("grid_sizes must have the same length as level_spikes.")
    if int(min_group_size) <= 0:
        raise ValueError("min_group_size must be positive.")
    if time_aggregate not in {"any", "per_time"}:
        raise ValueError('time_aggregate must be "any" or "per_time".')

    batch_size = None
    num_steps = None
    for level_idx, (spikes, grid_size) in enumerate(zip(level_spikes, grid_sizes)):
        if not torch.is_tensor(spikes):
            raise ValueError(f"level_spikes[{level_idx}] must be a tensor.")
        if spikes.dim() != 3:
            raise ValueError(f"level_spikes[{level_idx}] must have shape [B, N, T].")

        grid_h, grid_w = _parse_grid_size(grid_size)
        if spikes.size(1) != grid_h * grid_w:
            raise ValueError(
                f"level_spikes[{level_idx}] has {spikes.size(1)} oscillators, "
                f"but grid {grid_h}x{grid_w} has {grid_h * grid_w}."
            )
        if batch_size is None:
            batch_size = spikes.size(0)
            num_steps = spikes.size(2)
        elif spikes.size(0) != batch_size or spikes.size(2) != num_steps:
            raise ValueError("All levels must share the same batch size and time length.")


def _parse_grid_size(value):
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("grid size must be positive.")
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError("grid size values must be positive.")
        return height, width
    raise ValueError("grid size must be an int or a pair of ints.")
