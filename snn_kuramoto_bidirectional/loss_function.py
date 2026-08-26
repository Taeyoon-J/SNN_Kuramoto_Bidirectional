import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hyperparameter import DEFAULT_HYPERPARAMETERS
except ImportError:
    from hyperparameter import DEFAULT_HYPERPARAMETERS


def spike_rate_loss(
    core_out,
    target_rate=0.25,
    v_th=0.5,
    temperature=0.1,
    reduction="mean",
):
    """
    Keep threshold-aware soft spiking activity near a target firing rate.

    Args:
        core_out:
            Raw membrane history shaped [B, N, T].
    """
    if core_out.dim() != 3:
        raise ValueError("core_out must have shape [B, N, T].")

    soft_spikes = torch.sigmoid(
        (core_out.float() - float(v_th)) / float(temperature)
    )
    soft_rate = soft_spikes.mean()
    loss = (soft_rate - float(target_rate)).pow(2)
    return _reduce(loss, reduction)


def dense_magnitude_loss(dense_i, target_magnitude=4.0):
    """Prevent local oscillator-dense output magnitude from collapsing."""
    magnitude = dense_i.abs().mean()
    return F.relu(float(target_magnitude) - magnitude).pow(2)


def dendritic_cancellation_loss(
    h,
    target_retained_fraction=0.5,
    branch_dim=-1,
    eps=1e-8,
):
    """Penalize excessive signed cancellation across dendritic branches.

    Completely inactive branch sets have retained fraction one via symmetric
    epsilon stabilization, so zero activity is not treated as cancellation.
    """
    total_branch_activity = h.abs().sum(dim=branch_dim)
    retained_activity = h.sum(dim=branch_dim).abs()
    retained_fraction = (
        retained_activity + float(eps)
    ) / (
        total_branch_activity + float(eps)
    )
    penalty = F.relu(float(target_retained_fraction) - retained_fraction)
    return penalty.pow(2).mean()


def spike_temporal_smoothness_loss(activity, reduction="mean"):
    """Discourage abrupt frame-to-frame changes in activity histories."""
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if activity.size(2) < 2:
        return activity.new_zeros(())

    loss = (activity[:, :, 1:] - activity[:, :, :-1]).pow(2).mean(dim=(1, 2))
    return _reduce(loss, reduction)


def spike_diversity_loss(activity, reduction="mean", eps=1e-8):
    """
    Decorrelation loss across oscillators.

    This keeps every oscillator from learning the same activity history.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")

    similarity = _pairwise_cosine(activity.float(), eps=eps)
    off_diag = _off_diagonal(similarity)
    loss = off_diag.pow(2).mean(dim=1)
    return _reduce(loss, reduction)


def structural_consistency_loss(activity, sc, reduction="mean", eps=1e-8):
    """
    Match spike-rhythm similarity to structural connectivity.

    Args:
        activity:
            Tensor shaped [B, N, T].
        sc:
            Tensor shaped [N, N] or [B, N, N].
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")

    activity_similarity = _pairwise_cosine(activity.float(), eps=eps)
    sc = _prepare_sc(
        sc,
        batch_size=activity.size(0),
        device=activity.device,
        dtype=activity.dtype,
    )
    sc = _minmax_normalize(sc, eps=eps)

    loss = (
        _off_diagonal(activity_similarity) - _off_diagonal(sc)
    ).pow(2).mean(dim=1)
    return _reduce(loss, reduction)


def sample_activity_diversity_loss(activity, reduction="mean", eps=1e-8):
    """
    Penalize different samples producing the same activity mask.

    Args:
        activity:
            Tensor shaped [B, N, T].
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if activity.size(0) < 2:
        return activity.new_zeros(())

    flat = activity.float().flatten(start_dim=1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    similarity = F.cosine_similarity(
        flat.unsqueeze(1),
        flat.unsqueeze(0),
        dim=-1,
        eps=eps,
    )
    off_diag = similarity[~torch.eye(activity.size(0), device=activity.device, dtype=torch.bool)]
    return _reduce(off_diag.pow(2), reduction)


def spatial_compactness_loss(activity, patch_grid_size, reduction="mean"):
    """
    Encourage spatially adjacent patch oscillators to form smooth components.

    This is a differentiable total-variation style term on the temporally
    averaged patch activity.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    grid_h, grid_w = _parse_grid_size(patch_grid_size)
    if activity.size(1) != grid_h * grid_w:
        raise ValueError(
            f"activity has {activity.size(1)} oscillators, but grid "
            f"{grid_h}x{grid_w} has {grid_h * grid_w}."
        )

    grid = activity.float().mean(dim=2).view(activity.size(0), grid_h, grid_w)
    vertical = (grid[:, 1:, :] - grid[:, :-1, :]).abs().mean(dim=(1, 2))
    horizontal = (grid[:, :, 1:] - grid[:, :, :-1]).abs().mean(dim=(1, 2))
    return _reduce(vertical + horizontal, reduction)


def edge_membrane_separation_loss(
    core_out,
    images,
    grid_size,
    margin=0.3,
    eps=1e-8,
):
    """Separate adjacent membrane patterns across strong RGB boundaries.

    Args:
        core_out:
            Membrane histories shaped [B, N, T].
        images:
            Original RGB images shaped [B, 3, H, W].
        grid_size:
            Row-major oscillator grid whose product equals N. Boundary weights
            compare RGB pixels directly across each shared patch boundary.
        margin:
            Centered cosine similarities above this value are penalized.

    Returns:
        Scalar RGB-weighted membrane separation loss.
    """
    if core_out.dim() != 3:
        raise ValueError("core_out must have shape [B, N, T].")
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("images must have shape [B, 3, H, W].")
    if images.size(0) != core_out.size(0):
        raise ValueError("images and core_out must have the same batch size.")

    grid_h, grid_w = _parse_grid_size(grid_size)
    if core_out.size(1) != grid_h * grid_w:
        raise ValueError(
            f"core_out has {core_out.size(1)} oscillators, but grid "
            f"{grid_h}x{grid_w} requires {grid_h * grid_w}."
        )

    image_h, image_w = images.shape[-2:]
    if grid_h > image_h or grid_w > image_w:
        raise ValueError("grid_size cannot exceed the image spatial dimensions.")

    centered = core_out - core_out.mean(dim=2, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=2, eps=float(eps))
    membrane_grid = normalized.reshape(
        core_out.size(0), grid_h, grid_w, core_out.size(2)
    )
    horizontal_similarity = (
        membrane_grid[:, :, :-1, :] * membrane_grid[:, :, 1:, :]
    ).sum(dim=-1)
    vertical_similarity = (
        membrane_grid[:, :-1, :, :] * membrane_grid[:, 1:, :, :]
    ).sum(dim=-1)

    with torch.no_grad():
        boundary_images = images.detach().to(
            device=core_out.device,
            dtype=core_out.dtype,
        )

        horizontal_distances = torch.linalg.vector_norm(
            boundary_images[:, :, :, 1:] - boundary_images[:, :, :, :-1],
            ord=2,
            dim=1,
        )
        horizontal_columns = torch.div(
            torch.arange(1, grid_w, device=core_out.device) * image_w,
            grid_w,
            rounding_mode="floor",
        )
        horizontal_boundaries = horizontal_distances[
            :, :, horizontal_columns - 1
        ]
        horizontal_weights = F.adaptive_avg_pool1d(
            horizontal_boundaries.permute(0, 2, 1).reshape(-1, 1, image_h),
            grid_h,
        ).reshape(core_out.size(0), grid_w - 1, grid_h).transpose(1, 2)

        vertical_distances = torch.linalg.vector_norm(
            boundary_images[:, :, 1:, :] - boundary_images[:, :, :-1, :],
            ord=2,
            dim=1,
        )
        vertical_rows = torch.div(
            torch.arange(1, grid_h, device=core_out.device) * image_h,
            grid_h,
            rounding_mode="floor",
        )
        vertical_boundaries = vertical_distances[:, vertical_rows - 1, :]
        vertical_weights = F.adaptive_avg_pool1d(
            vertical_boundaries.reshape(-1, 1, image_w),
            grid_w,
        ).reshape(core_out.size(0), grid_h - 1, grid_w)

    horizontal_penalty = F.relu(horizontal_similarity - float(margin))
    vertical_penalty = F.relu(vertical_similarity - float(margin))
    weighted_penalty = (
        (horizontal_weights * horizontal_penalty).sum()
        + (vertical_weights * vertical_penalty).sum()
    )
    total_boundary_weight = horizontal_weights.sum() + vertical_weights.sum()
    return weighted_penalty / (total_boundary_weight + float(eps))


def temporal_activity_balance_loss(activity, reduction="mean"):
    """
    Penalize global activity monotonically collapsing or saturating over time.

    It compares the mean activity per time step to each sample's average
    activity over the full sequence.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if activity.size(2) < 2:
        return activity.new_zeros(())

    activity_by_time = activity.float().mean(dim=1)
    target = activity_by_time.mean(dim=1, keepdim=True)
    loss = (activity_by_time - target).pow(2).mean(dim=1)
    return _reduce(loss, reduction)


def object_overlap_loss(object_groups, num_oscillators=None, reduction="mean", device=None):
    """
    Penalize one oscillator being assigned to multiple detected objects.

    Args:
        object_groups:
            List with length B. Each item is a list of tuples/lists containing
            oscillator indices for detected objects.
        num_oscillators:
            Optional total number of oscillators. If omitted, it is inferred
            from the largest oscillator index in object_groups.

    Note:
        This term is computed from discrete object groups, so it is useful as
        an objective value/selection pressure but does not provide gradients
        through the grouping operation itself.
    """
    if object_groups is None:
        raise ValueError("object_groups must not be None.")
    if not isinstance(object_groups, (list, tuple)):
        raise ValueError("object_groups must be a list with length B.")

    device = torch.device("cpu") if device is None else torch.device(device)
    if num_oscillators is None:
        max_index = -1
        for batch_groups in object_groups:
            for group in batch_groups:
                if len(group) > 0:
                    max_index = max(max_index, max(int(index) for index in group))
        num_oscillators = max_index + 1

    if int(num_oscillators) <= 0:
        losses = torch.zeros(len(object_groups), device=device)
        return _reduce(losses, reduction)

    losses = []
    for batch_groups in object_groups:
        counts = torch.zeros(int(num_oscillators), device=device)
        for group in batch_groups:
            if len(group) == 0:
                continue
            indices = torch.as_tensor(group, device=device, dtype=torch.long)
            counts.index_add_(0, indices, torch.ones_like(indices, dtype=counts.dtype))
        duplicate_counts = F.relu(counts - 1.0)
        losses.append(duplicate_counts.pow(2).mean())

    return _reduce(torch.stack(losses), reduction)


class UnsupervisedS2NetLoss(nn.Module):
    """
    Weighted unsupervised objective for the spike classifier side of S2Net.

    Expected inputs to forward:
        spikes:    [B, N, T]
        core_out:  raw membrane history [B, N, T]
        object_groups:
            list length B. Each item contains object oscillator-index groups.
        sc:        [N, N] or [B, N, N]

    Any input can be omitted; its corresponding weighted term is skipped.
    """

    def __init__(
        self,
        spike_rate_weight=DEFAULT_HYPERPARAMETERS.spike_rate_weight,
        spike_smooth_weight=DEFAULT_HYPERPARAMETERS.spike_smooth_weight,
        spike_diversity_weight=DEFAULT_HYPERPARAMETERS.spike_diversity_weight,
        structural_weight=DEFAULT_HYPERPARAMETERS.structural_weight,
        object_overlap_weight=DEFAULT_HYPERPARAMETERS.object_overlap_weight,
        sample_diversity_weight=DEFAULT_HYPERPARAMETERS.sample_diversity_weight,
        spatial_compactness_weight=DEFAULT_HYPERPARAMETERS.spatial_compactness_weight,
        temporal_balance_weight=DEFAULT_HYPERPARAMETERS.temporal_balance_weight,
        edge_membrane_weight=DEFAULT_HYPERPARAMETERS.edge_membrane_weight,
        edge_membrane_margin=DEFAULT_HYPERPARAMETERS.edge_membrane_margin,
        dense_magnitude_weight=DEFAULT_HYPERPARAMETERS.dense_magnitude_weight,
        dendritic_cancellation_weight=DEFAULT_HYPERPARAMETERS.dendritic_cancellation_weight,
        spike_target_rate=0.25,
        spike_v_th=0.5,
        spike_temperature=0.1,
        patch_grid_size=None,
    ):
        super().__init__()
        self.spike_rate_weight = float(spike_rate_weight)
        self.spike_smooth_weight = float(spike_smooth_weight)
        self.spike_diversity_weight = float(spike_diversity_weight)
        self.structural_weight = float(structural_weight)
        self.object_overlap_weight = float(object_overlap_weight)
        self.sample_diversity_weight = float(sample_diversity_weight)
        self.spatial_compactness_weight = float(spatial_compactness_weight)
        self.temporal_balance_weight = float(temporal_balance_weight)
        self.edge_membrane_weight = float(edge_membrane_weight)
        self.edge_membrane_margin = float(edge_membrane_margin)
        self.dense_magnitude_weight = float(dense_magnitude_weight)
        self.dendritic_cancellation_weight = float(dendritic_cancellation_weight)
        self.spike_target_rate = float(spike_target_rate)
        self.spike_v_th = float(spike_v_th)
        self.spike_temperature = float(spike_temperature)
        self.patch_grid_size = patch_grid_size

    def forward(
        self,
        spikes=None,
        object_groups=None,
        sc=None,
        core_out=None,
        images=None,
        dense_i=None,
        dendritic_h=None,
    ):
        # ``spikes`` is the legacy public keyword for the selected loss signal.
        # Internally it may contain spikes, membrane, or sigmoid membrane.
        activity = spikes
        device, dtype = _infer_device_dtype(
            activity, sc, core_out, dense_i, dendritic_h
        )
        total = torch.zeros((), device=device, dtype=dtype)
        parts = {}

        if core_out is not None:
            parts["spike_rate"] = spike_rate_loss(
                core_out,
                target_rate=self.spike_target_rate,
                v_th=self.spike_v_th,
                temperature=self.spike_temperature,
            )

        if activity is not None:
            parts["spike_smooth"] = spike_temporal_smoothness_loss(activity)
            parts["spike_diversity"] = spike_diversity_loss(activity)

        if activity is not None and sc is not None:
            parts["structural"] = structural_consistency_loss(activity, sc)

        if activity is not None:
            parts["sample_diversity"] = sample_activity_diversity_loss(activity)
            parts["temporal_balance"] = temporal_activity_balance_loss(activity)
            if self.patch_grid_size is not None:
                parts["spatial_compactness"] = spatial_compactness_loss(
                    activity,
                    patch_grid_size=self.patch_grid_size,
                )

        if object_groups is not None:
            parts["object_overlap"] = object_overlap_loss(
                object_groups,
                num_oscillators=(
                    activity.size(1) if activity is not None else None
                ),
                device=device,
            )

        if core_out is not None and images is not None and self.patch_grid_size is not None:
            parts["edge_membrane"] = edge_membrane_separation_loss(
                core_out=core_out,
                images=images,
                grid_size=self.patch_grid_size,
                margin=self.edge_membrane_margin,
            )

        if self.dense_magnitude_weight > 0.0 and dense_i is not None:
            parts["dense_magnitude_loss"] = dense_magnitude_loss(dense_i)

        if self.dendritic_cancellation_weight > 0.0 and dendritic_h is not None:
            parts["dendritic_cancellation_loss"] = dendritic_cancellation_loss(
                dendritic_h,
                branch_dim=-1,
            )

        weights = {
            "spike_rate": self.spike_rate_weight,
            "spike_smooth": self.spike_smooth_weight,
            "spike_diversity": self.spike_diversity_weight,
            "structural": self.structural_weight,
            "object_overlap": self.object_overlap_weight,
            "sample_diversity": self.sample_diversity_weight,
            "spatial_compactness": self.spatial_compactness_weight,
            "temporal_balance": self.temporal_balance_weight,
            "edge_membrane": self.edge_membrane_weight,
            "dense_magnitude_loss": self.dense_magnitude_weight,
            "dendritic_cancellation_loss": self.dendritic_cancellation_weight,
        }
        for name, value in parts.items():
            total = total + weights[name] * value

        parts["total"] = total
        return total, parts


def _pairwise_cosine(values, eps=1e-8):
    left = values.unsqueeze(2)
    right = values.unsqueeze(1)
    return F.cosine_similarity(left, right, dim=-1, eps=eps)


def _off_diagonal(matrix):
    if matrix.dim() != 3:
        raise ValueError("matrix must have shape [B, N, N].")

    n = matrix.size(-1)
    mask = ~torch.eye(n, device=matrix.device, dtype=torch.bool)
    return matrix[:, mask].view(matrix.size(0), n * (n - 1))


def _prepare_sc(sc, batch_size, device, dtype):
    if sc.dim() == 2:
        sc = sc.unsqueeze(0).expand(batch_size, -1, -1)
    elif sc.dim() != 3:
        raise ValueError("sc must have shape [N, N] or [B, N, N].")

    if sc.size(0) != batch_size:
        raise ValueError(f"sc batch size {sc.size(0)} does not match spikes batch size {batch_size}.")
    return sc.to(device=device, dtype=dtype)


def _minmax_normalize(values, eps=1e-8):
    flat = values.flatten(start_dim=1)
    min_value = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_value = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
    return (values - min_value) / (max_value - min_value + eps)


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


def _reduce(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError('reduction must be one of "none", "mean", or "sum".')


def _infer_device_dtype(*values):
    for value in values:
        if torch.is_tensor(value):
            return value.device, value.dtype if value.is_floating_point() else torch.float32
    return torch.device("cpu"), torch.float32
