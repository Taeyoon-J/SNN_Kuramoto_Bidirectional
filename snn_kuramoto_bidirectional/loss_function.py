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
        object_overlap_weight=1.0,
        spike_target_rate=0.1,
    ):
        super().__init__()
        self.spike_rate_weight = float(spike_rate_weight)
        self.spike_smooth_weight = float(spike_smooth_weight)
        self.spike_diversity_weight = float(spike_diversity_weight)
        self.structural_weight = float(structural_weight)
        self.object_overlap_weight = float(object_overlap_weight)
        self.spike_target_rate = float(spike_target_rate)

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
