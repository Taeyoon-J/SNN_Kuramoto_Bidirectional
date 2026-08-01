import torch
import torch.nn as nn
import torch.nn.functional as F


def spike_rate_loss(spikes, target_rate=0.1, reduction="mean"):
    """
    Keep unsupervised spiking activity near a target firing rate.

    Args:
        spikes:
            Tensor shaped [B, N, T].
    """
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")

    rate = spikes.float().mean(dim=(1, 2))
    loss = (rate - float(target_rate)).pow(2)
    return _reduce(loss, reduction)


def spike_temporal_smoothness_loss(spikes, reduction="mean"):
    """Discourage abrupt frame-to-frame changes in spike histories."""
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")
    if spikes.size(2) < 2:
        return spikes.new_zeros(())

    loss = (spikes[:, :, 1:] - spikes[:, :, :-1]).pow(2).mean(dim=(1, 2))
    return _reduce(loss, reduction)


def spike_diversity_loss(spikes, reduction="mean", eps=1e-8):
    """
    Decorrelation loss across oscillators.

    This keeps every oscillator from learning the same spike train.
    """
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")

    similarity = _pairwise_cosine(spikes.float(), eps=eps)
    off_diag = _off_diagonal(similarity)
    loss = off_diag.pow(2).mean(dim=1)
    return _reduce(loss, reduction)


def structural_consistency_loss(spikes, sc, reduction="mean", eps=1e-8):
    """
    Match spike-rhythm similarity to structural connectivity.

    Args:
        spikes:
            Tensor shaped [B, N, T].
        sc:
            Tensor shaped [N, N] or [B, N, N].
    """
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")

    spike_similarity = _pairwise_cosine(spikes.float(), eps=eps)
    sc = _prepare_sc(sc, batch_size=spikes.size(0), device=spikes.device, dtype=spikes.dtype)
    sc = _minmax_normalize(sc, eps=eps)

    loss = (_off_diagonal(spike_similarity) - _off_diagonal(sc)).pow(2).mean(dim=1)
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
        object_groups:
            list length B. Each item contains object oscillator-index groups.
        sc:        [N, N] or [B, N, N]

    Any input can be omitted; its corresponding weighted term is skipped.
    """

    def __init__(
        self,
        spike_rate_weight=1.0,
        spike_smooth_weight=0.1,
        spike_diversity_weight=0.1,
        structural_weight=0.1,
        object_overlap_weight=0.0,
        sample_diversity_weight=0.0,
        spatial_compactness_weight=0.0,
        temporal_balance_weight=0.0,
        spike_target_rate=0.1,
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
        self.spike_target_rate = float(spike_target_rate)
        self.patch_grid_size = patch_grid_size

    def forward(self, spikes=None, object_groups=None, sc=None):
        device, dtype = _infer_device_dtype(spikes, sc)
        total = torch.zeros((), device=device, dtype=dtype)
        parts = {}

        if spikes is not None:
            parts["spike_rate"] = spike_rate_loss(
                spikes,
                target_rate=self.spike_target_rate,
            )
            parts["spike_smooth"] = spike_temporal_smoothness_loss(spikes)
            parts["spike_diversity"] = spike_diversity_loss(spikes)

        if spikes is not None and sc is not None:
            parts["structural"] = structural_consistency_loss(spikes, sc)

        if spikes is not None:
            parts["sample_diversity"] = sample_activity_diversity_loss(spikes)
            parts["temporal_balance"] = temporal_activity_balance_loss(spikes)
            if self.patch_grid_size is not None:
                parts["spatial_compactness"] = spatial_compactness_loss(
                    spikes,
                    patch_grid_size=self.patch_grid_size,
                )

        if object_groups is not None:
            parts["object_overlap"] = object_overlap_loss(
                object_groups,
                num_oscillators=spikes.size(1) if spikes is not None else None,
                device=device,
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
