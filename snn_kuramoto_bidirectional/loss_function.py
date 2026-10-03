
if __package__:
    from . import error_bound
else:
    import error_bound

import math

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
    error_bound.validate_loss_function_spike_rate_loss(spikes)

    rate = spikes.float().mean(dim=(1, 2))
    loss = (rate - float(target_rate)).pow(2)
    return _reduce(loss, reduction)


def spike_temporal_smoothness_loss(spikes, reduction="mean"):
    """Discourage abrupt frame-to-frame changes in spike histories."""
    error_bound.validate_loss_function_spike_rate_loss(spikes)
    if spikes.size(2) < 2:
        return spikes.new_zeros(())

    loss = (spikes[:, :, 1:] - spikes[:, :, :-1]).pow(2).mean(dim=(1, 2))
    return _reduce(loss, reduction)


def spike_diversity_loss(spikes, reduction="mean", eps=1e-8):
    """
    Decorrelation loss across oscillators.

    This keeps every oscillator from learning the same spike train.
    """
    error_bound.validate_loss_function_spike_rate_loss(spikes)

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
    error_bound.validate_loss_function_spike_rate_loss(spikes)

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
    error_bound.validate_loss_function_sample_activity_diversity_loss(activity)
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
    error_bound.validate_loss_function_sample_activity_diversity_loss(activity)
    grid_h, grid_w = _parse_grid_size(patch_grid_size)
    error_bound.validate_loss_function_spatial_compactness_loss(grid_h, grid_w, activity)

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
    error_bound.validate_loss_function_sample_activity_diversity_loss(activity)
    if activity.size(2) < 2:
        return activity.new_zeros(())

    activity_by_time = activity.float().mean(dim=1)
    target = activity_by_time.mean(dim=1, keepdim=True)
    loss = (activity_by_time - target).pow(2).mean(dim=1)
    return _reduce(loss, reduction)


def activity_confidence_loss(activity, reduction="mean"):
    """
    Push continuous activity away from the ambiguous 0.5 region.

    This expects probability-like activity values in [0, 1], such as
    sigmoid(membrane). The loss is highest near 0.5 and lowest near 0 or 1.
    """
    error_bound.validate_loss_function_sample_activity_diversity_loss(activity)

    activity = activity.float().clamp(0.0, 1.0)
    loss = (activity * (1.0 - activity)).mean(dim=(1, 2))
    return _reduce(loss, reduction)


def activity_area_loss(activity, min_area=0.05, max_area=0.35, reduction="mean"):
    """
    Keep soft mask area inside a useful range.

    The area is the average activity per sample. This differentiable proxy
    discourages both empty masks and all-on masks before thresholding.
    """
    error_bound.validate_loss_function_activity_area_loss(activity, min_area, max_area)

    area = activity.float().clamp(0.0, 1.0).mean(dim=(1, 2))
    loss = F.relu(float(min_area) - area).pow(2) + F.relu(area - float(max_area)).pow(2)
    return _reduce(loss, reduction)


def activity_contrast_loss(activity, target_std=0.15, reduction="mean"):
    """
    Encourage visible separation between active and inactive patches.

    This prevents every oscillator from living in a narrow band around 0.5,
    which makes threshold-based masks brittle.
    """
    error_bound.validate_loss_function_activity_contrast_loss(activity, target_std)

    std = activity.float().flatten(start_dim=1).std(dim=1)
    loss = F.relu(float(target_std) - std).pow(2)
    return _reduce(loss, reduction)


def phase_locking_value(theta, settle=0):
    """
    Pairwise phase-locking value between oscillators.

    Args:
        theta: [B, T, N, D] phase history. The D components are averaged first.
        settle: number of leading steps to discard as transient.

    Returns:
        [B, N, N] in [0, 1]. PLV is 1 for a constant phase difference.

    This is the synchrony readout. Time-averaging sigmoid(membrane), the readout
    the spatial-components path uses, discards phase relationships entirely,
    which is precisely the information a binding-by-synchrony model carries.
    Computed with real matmuls rather than complex tensors so it stays cheap and
    avoids complex autograd.
    """
    error_bound.validate_loss_function_phase_locking_value(theta)
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        error_bound.validate_loss_function_phase_locking_value_2(settle, phase)
        phase = phase[:, int(settle):]

    steps = phase.size(1)
    cos_t, sin_t = torch.cos(phase), torch.sin(phase)
    real = (cos_t.transpose(1, 2) @ cos_t + sin_t.transpose(1, 2) @ sin_t) / steps
    imag = (sin_t.transpose(1, 2) @ cos_t - cos_t.transpose(1, 2) @ sin_t) / steps
    return torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12).clamp(0.0, 1.0)


def plv_bimodality_loss(plv, reduction="mean"):
    """
    Push pairwise synchrony towards locked or unlocked, away from ambiguity.

    A binding solution is a partition: two oscillators are in the same group or
    they are not. Intermediate PLV means no decision has been made.
    """
    off_diag = _off_diagonal(plv)
    loss = (off_diag * (1.0 - off_diag)).mean(dim=1)
    return _reduce(loss, reduction)


def preferred_phase(theta, settle=0, eps=1e-8):
    """Each unit's phase relative to the global mean field. -> [B, N] in (-pi, pi]."""
    error_bound.validate_loss_function_phase_locking_value(theta)
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        error_bound.validate_loss_function_phase_locking_value_2(settle, phase)
        phase = phase[:, int(settle):]
    field = torch.atan2(
        torch.sin(phase).mean(dim=2, keepdim=True),
        torch.cos(phase).mean(dim=2, keepdim=True),
    )
    relative = phase - field
    return torch.atan2(torch.sin(relative).mean(dim=1), torch.cos(relative).mean(dim=1))


def order_parameters(theta, num_slots=7.0, settle=0):
    """
    First- and k-th order Kuramoto order parameters, averaged over time.

    R1 = |<exp(i*theta)>_units|    1 when every unit shares one phase
    Rk = |<exp(i*k*theta)>_units|  1 when phases sit on a k-fold grid

    Both are reference free, which matters: measuring a phase relative to the
    mean field breaks precisely in the state we want, because k evenly spread
    phases cancel and leave the field angle undefined.

    A k-cluster state is exactly Rk high with R1 low.
    """
    error_bound.validate_loss_function_phase_locking_value(theta)
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        error_bound.validate_loss_function_phase_locking_value_2(settle, phase)
        phase = phase[:, int(settle):]

    def resultant(angle):
        return (torch.cos(angle).mean(dim=2).pow(2)
                + torch.sin(angle).mean(dim=2).pow(2)).clamp_min(1e-12).sqrt()

    return resultant(phase).mean(dim=1), resultant(float(num_slots) * phase).mean(dim=1)


def phase_quantization_loss(theta, num_slots=7.0, settle=0, reduction="mean"):
    """
    Snap phases onto num_slots evenly spaced positions on the circle.

    This states the "one object per phase window" hypothesis directly: the k-th
    order parameter is 1 exactly when every phase difference is a multiple of
    2*pi/k.

    plv_bimodality cannot express this. Pushing the alignment matrix to 0 or 1
    asks cross-group pairs to be antiphase, which more than two groups on a
    circle cannot all be: with k slots the cross-group alignment values spread
    across [0, 1] with a mean near 0.5, so that target is unreachable by
    construction.
    """
    _, r_k = order_parameters(theta, num_slots=num_slots, settle=settle)
    return _reduce(1.0 - r_k, reduction)


def phase_spread_loss(theta, settle=0, reduction="mean"):
    """
    Penalise every unit sharing a single phase.

    Quantization alone is perfectly satisfied by one occupied slot, the same
    degenerate global-synchrony solution that appeared for the PLV losses. The
    first order parameter is 1 in exactly that state and low once the phases
    occupy several slots.
    """
    r_1, _ = order_parameters(theta, settle=settle)
    return _reduce(r_1, reduction)


def phase_alignment(theta, settle=0, eps=1e-8):
    """
    Pairwise phase alignment: 1 when two oscillators sit at the same phase.

    phase_locking_value asks whether a phase difference is *constant*, so two
    units locked a quarter cycle apart score 1. That is the wrong question for a
    code where an object is the set of units active at the same moment: there
    the difference has to be near zero, not merely steady.

    Each unit's preferred phase is taken relative to the global mean field, so
    collective drift cancels and what remains is where the unit sits within the
    shared rhythm.

    Args:
        theta: [B, T, N, D] phase history.
        settle: leading steps to discard as transient.

    Returns:
        [B, N, N] in [0, 1], (1 + cos(phi_i - phi_j)) / 2.
    """
    error_bound.validate_loss_function_phase_locking_value(theta)
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        error_bound.validate_loss_function_phase_locking_value_2(settle, phase)
        phase = phase[:, int(settle):]

    field = torch.atan2(
        torch.sin(phase).mean(dim=2, keepdim=True),
        torch.cos(phase).mean(dim=2, keepdim=True),
    )
    relative = phase - field
    cos_p = torch.cos(relative).mean(dim=1)
    sin_p = torch.sin(relative).mean(dim=1)
    scale = (cos_p.pow(2) + sin_p.pow(2)).sqrt().clamp_min(eps)
    cos_p, sin_p = cos_p / scale, sin_p / scale

    align = cos_p.unsqueeze(2) * cos_p.unsqueeze(1) + sin_p.unsqueeze(2) * sin_p.unsqueeze(1)
    return ((1.0 + align) / 2.0).clamp(0.0, 1.0)


def signal_synchrony(signal, settle=0, eps=1e-8):
    """
    Pairwise synchrony between real-valued unit traces, e.g. membrane or spikes.

    Args:
        signal: [B, N, T].
        settle: leading steps to discard as transient.

    Returns:
        [B, N, N] in [0, 1], the absolute centred correlation between traces.

    phase_locking_value reads the Kuramoto phases directly, which bypasses the
    dendritic and membrane layers entirely. This reads the same synchrony off
    whatever the spiking side produces, so the SNN sits inside the measured
    path. Centring matters: without it, all-positive near-constant traces give a
    cosine of about 1 for every pair, which is why spike_diversity_loss sat
    pinned near its maximum for a whole training run.
    """
    error_bound.validate_loss_function_signal_synchrony(signal)
    trace = signal.float()
    if int(settle) > 0:
        error_bound.validate_loss_function_signal_synchrony_2(settle, trace)
        trace = trace[:, :, int(settle):]

    trace = trace - trace.mean(dim=2, keepdim=True)
    trace = trace / trace.norm(dim=2, keepdim=True).clamp_min(eps)
    return torch.bmm(trace, trace.transpose(1, 2)).abs().clamp(0.0, 1.0)


def plv_collapse_loss(plv, eps=1e-4, reduction="mean"):
    """
    Barrier against a uniform synchrony matrix.

    plv_bimodality_loss is zero at PLV == 1 everywhere just as much as at a real
    partition, so global synchrony is one of its minima, and it is the one the
    optimiser reaches: measured per-element gradients at PLV == 1 are 1/M for
    bimodality against 0.27/M for a 0.867 density target and 0.49/M for a group
    count target, so those constraints lose by 2x to 4x and the run collapses
    (within 0.9993, between 0.9990). A 0.25 density target wins at 1.5/M, which
    is the only reason that configuration escaped.

    This removes the minimum instead of out-weighting it: the loss diverges as
    the variance of the off-diagonal entries goes to zero, so a constant matrix
    is not a solution at any weight. For a 0/1 partition with density d the
    variance is d(1 - d), giving about 2.2 at d = 0.867 against 9.2 at collapse.
    """
    off_diag = _off_diagonal(plv)
    variance = off_diag.var(dim=1, unbiased=False)
    return _reduce(-torch.log(variance + float(eps)), reduction)


def plv_group_count_loss(plv, target_groups=7.0, reduction="mean"):
    """
    Target the effective number of synchronised groups.

    For a symmetric matrix with a unit diagonal the participation ratio

        PR = N^2 / ||PLV||_F^2

    counts groups directly: 1 when everything is locked into one group, N when
    nothing is locked, and k for k equal blocks. It needs no eigendecomposition.

    This replaces targeting a mean synchrony level, which turned out to be
    ill-posed: with CLEVR ground truth the ideal mean PLV is 0.867 if the
    background counts as one group and 0.003 if it stays incoherent, so any
    intermediate target pulls the solution away from both.
    """
    num_nodes = plv.size(-1)
    frob_sq = plv.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)
    participation = (float(num_nodes) ** 2) / frob_sq
    target = float(target_groups)
    loss = ((participation - target) / target).pow(2)
    return _reduce(loss, reduction)


def plv_group_balance_loss(plv, target_density=0.25, reduction="mean"):
    """
    Keep the mean synchrony near a target.

    Without this the bimodality term is minimised by locking everything (PLV 1
    everywhere, one global group) or nothing (PLV 0, no groups). Both were
    observed: dense coupling drove global synchrony at high K.
    """
    density = _off_diagonal(plv).mean(dim=1)
    loss = (density - float(target_density)).pow(2)
    return _reduce(loss, reduction)


def plv_spatial_coherence_loss(plv, patch_grid_size, reduction="mean"):
    """
    Prefer synchronised groups that are spatially contiguous.

    Objects are connected regions, so a group scattered across the grid is not
    an object. This penalises synchrony that varies sharply between neighbouring
    patches, applied to each oscillator's mean synchrony with the rest.
    """
    grid_h, grid_w = _parse_grid_size(patch_grid_size)
    batch_size, num_nodes, _ = plv.shape
    error_bound.validate_loss_function_plv_spatial_coherence_loss(num_nodes, grid_h, grid_w)
    field = _off_diagonal(plv).view(batch_size, num_nodes, num_nodes - 1).mean(dim=2)
    field = field.view(batch_size, grid_h, grid_w)
    d_h = (field[:, 1:, :] - field[:, :-1, :]).abs().mean(dim=(1, 2))
    d_w = (field[:, :, 1:] - field[:, :, :-1]).abs().mean(dim=(1, 2))
    return _reduce(d_h + d_w, reduction)


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
    error_bound.validate_loss_function_object_overlap_loss(object_groups)

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
        activity_confidence_weight=0.0,
        activity_area_weight=0.0,
        activity_contrast_weight=0.0,
        spike_target_rate=0.1,
        patch_grid_size=None,
        activity_min_area=0.05,
        activity_max_area=0.35,
        activity_target_std=0.15,
        plv_bimodality_weight=0.0,
        plv_balance_weight=0.0,
        plv_coherence_weight=0.0,
        plv_group_count_weight=0.0,
        plv_collapse_weight=0.0,
        phase_quantization_weight=0.0,
        phase_spread_weight=0.0,
        phase_num_slots=7.0,
        plv_target_density=0.25,
        plv_target_groups=7.0,
    ):
        super().__init__()
        self.plv_bimodality_weight = float(plv_bimodality_weight)
        self.plv_balance_weight = float(plv_balance_weight)
        self.plv_coherence_weight = float(plv_coherence_weight)
        self.plv_group_count_weight = float(plv_group_count_weight)
        self.plv_collapse_weight = float(plv_collapse_weight)
        self.phase_quantization_weight = float(phase_quantization_weight)
        self.phase_spread_weight = float(phase_spread_weight)
        self.phase_num_slots = float(phase_num_slots)
        self.plv_target_density = float(plv_target_density)
        self.plv_target_groups = float(plv_target_groups)
        self.spike_rate_weight = float(spike_rate_weight)
        self.spike_smooth_weight = float(spike_smooth_weight)
        self.spike_diversity_weight = float(spike_diversity_weight)
        self.structural_weight = float(structural_weight)
        self.object_overlap_weight = float(object_overlap_weight)
        self.sample_diversity_weight = float(sample_diversity_weight)
        self.spatial_compactness_weight = float(spatial_compactness_weight)
        self.temporal_balance_weight = float(temporal_balance_weight)
        self.activity_confidence_weight = float(activity_confidence_weight)
        self.activity_area_weight = float(activity_area_weight)
        self.activity_contrast_weight = float(activity_contrast_weight)
        self.spike_target_rate = float(spike_target_rate)
        self.patch_grid_size = patch_grid_size
        self.activity_min_area = float(activity_min_area)
        self.activity_max_area = float(activity_max_area)
        self.activity_target_std = float(activity_target_std)

    def forward(self, spikes=None, object_groups=None, sc=None, plv=None, theta=None,
                plv_settle=0):
        device, dtype = _infer_device_dtype(spikes, sc, plv)
        total = torch.zeros((), device=device, dtype=dtype)
        parts = {}

        if theta is not None:
            parts["phase_quantization"] = phase_quantization_loss(
                theta, num_slots=self.phase_num_slots, settle=plv_settle
            )
            parts["phase_spread"] = phase_spread_loss(theta, settle=plv_settle)

        if plv is not None:
            parts["plv_bimodality"] = plv_bimodality_loss(plv)
            parts["plv_balance"] = plv_group_balance_loss(
                plv,
                target_density=self.plv_target_density,
            )
            parts["plv_group_count"] = plv_group_count_loss(
                plv,
                target_groups=self.plv_target_groups,
            )
            parts["plv_collapse"] = plv_collapse_loss(plv)
            if self.patch_grid_size is not None:
                parts["plv_coherence"] = plv_spatial_coherence_loss(
                    plv,
                    patch_grid_size=self.patch_grid_size,
                )

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
            parts["activity_confidence"] = activity_confidence_loss(spikes)
            parts["activity_area"] = activity_area_loss(
                spikes,
                min_area=self.activity_min_area,
                max_area=self.activity_max_area,
            )
            parts["activity_contrast"] = activity_contrast_loss(
                spikes,
                target_std=self.activity_target_std,
            )
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
            "activity_confidence": self.activity_confidence_weight,
            "activity_area": self.activity_area_weight,
            "activity_contrast": self.activity_contrast_weight,
            "plv_bimodality": self.plv_bimodality_weight,
            "plv_balance": self.plv_balance_weight,
            "plv_coherence": self.plv_coherence_weight,
            "plv_group_count": self.plv_group_count_weight,
            "plv_collapse": self.plv_collapse_weight,
            "phase_quantization": self.phase_quantization_weight,
            "phase_spread": self.phase_spread_weight,
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
    error_bound.validate_loss_function_off_diagonal(matrix)

    n = matrix.size(-1)
    mask = ~torch.eye(n, device=matrix.device, dtype=torch.bool)
    return matrix[:, mask].view(matrix.size(0), n * (n - 1))


def _prepare_sc(sc, batch_size, device, dtype):
    if sc.dim() == 2:
        sc = sc.unsqueeze(0).expand(batch_size, -1, -1)
    elif sc.dim() != 3:
        raise ValueError("sc must have shape [N, N] or [B, N, N].")

    error_bound.validate_loss_function_prepare_sc(batch_size, sc)
    return sc.to(device=device, dtype=dtype)


def _minmax_normalize(values, eps=1e-8):
    flat = values.flatten(start_dim=1)
    min_value = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_value = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
    return (values - min_value) / (max_value - min_value + eps)


def _parse_grid_size(value):
    if isinstance(value, int):
        error_bound.validate_loss_function_parse_grid_size(value)
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        error_bound.validate_loss_function_parse_grid_size_2(height, width)
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
