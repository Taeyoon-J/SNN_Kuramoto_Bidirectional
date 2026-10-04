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
    if spikes.dim() != 3:
        raise ValueError("spikes must have shape [B, N, T].")

    rate = spikes.float().mean(dim=(1, 2))
    loss = (rate - float(target_rate)).pow(2)
    return _reduce(loss, reduction)


def edge_membrane_separation_loss(membrane, images, grid_size, margin=0.3, eps=1e-8):
    """Penalize synchronized neighboring membranes across strong RGB edges.

    Adapted from patch_sw's same-named loss. RGB only supplies non-trainable
    boundary weights; the gradient flows through the membrane histories.
    """
    if membrane.ndim != 3:
        raise ValueError("membrane must have shape [B,N,T]")
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] != membrane.shape[0]:
        raise ValueError("images must have shape [B,3,H,W] matching membrane batch")
    grid_h, grid_w = _parse_grid_size(grid_size)
    if membrane.shape[1] != grid_h * grid_w:
        raise ValueError("membrane oscillator count does not match grid")
    height, width = images.shape[-2:]
    if grid_h > height or grid_w > width:
        raise ValueError("grid cannot exceed RGB image dimensions")
    centered = membrane - membrane.mean(dim=2, keepdim=True)
    pattern = F.normalize(centered, p=2, dim=2, eps=eps).reshape(
        membrane.shape[0], grid_h, grid_w, membrane.shape[2]
    )
    horizontal_similarity = (pattern[:, :, :-1] * pattern[:, :, 1:]).sum(dim=-1)
    vertical_similarity = (pattern[:, :-1] * pattern[:, 1:]).sum(dim=-1)
    with torch.no_grad():
        rgb = images.detach().to(device=membrane.device, dtype=membrane.dtype)
        horizontal_distances = torch.linalg.vector_norm(
            rgb[:, :, :, 1:] - rgb[:, :, :, :-1], ord=2, dim=1
        )
        columns = torch.div(torch.arange(1, grid_w, device=membrane.device) * width,
                            grid_w, rounding_mode="floor")
        horizontal_boundaries = horizontal_distances[:, :, columns - 1]
        horizontal_weights = F.adaptive_avg_pool1d(
            horizontal_boundaries.permute(0, 2, 1).reshape(-1, 1, height), grid_h
        ).reshape(membrane.shape[0], grid_w - 1, grid_h).transpose(1, 2)
        vertical_distances = torch.linalg.vector_norm(
            rgb[:, :, 1:, :] - rgb[:, :, :-1, :], ord=2, dim=1
        )
        rows = torch.div(torch.arange(1, grid_h, device=membrane.device) * height,
                         grid_h, rounding_mode="floor")
        vertical_boundaries = vertical_distances[:, rows - 1, :]
        vertical_weights = F.adaptive_avg_pool1d(
            vertical_boundaries.reshape(-1, 1, width), grid_w
        ).reshape(membrane.shape[0], grid_h - 1, grid_w)
    numerator = (horizontal_weights * F.relu(horizontal_similarity - float(margin))).sum()
    numerator = numerator + (vertical_weights * F.relu(vertical_similarity - float(margin))).sum()
    return numerator / (horizontal_weights.sum() + vertical_weights.sum() + eps)


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


def activity_confidence_loss(activity, reduction="mean"):
    """
    Push continuous activity away from the ambiguous 0.5 region.

    This expects probability-like activity values in [0, 1], such as
    sigmoid(membrane). The loss is highest near 0.5 and lowest near 0 or 1.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")

    activity = activity.float().clamp(0.0, 1.0)
    loss = (activity * (1.0 - activity)).mean(dim=(1, 2))
    return _reduce(loss, reduction)


def activity_area_loss(activity, min_area=0.05, max_area=0.35, reduction="mean"):
    """
    Keep soft mask area inside a useful range.

    The area is the average activity per sample. This differentiable proxy
    discourages both empty masks and all-on masks before thresholding.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if min_area < 0.0 or max_area > 1.0 or min_area > max_area:
        raise ValueError("min_area and max_area must satisfy 0 <= min <= max <= 1.")

    area = activity.float().clamp(0.0, 1.0).mean(dim=(1, 2))
    loss = F.relu(float(min_area) - area).pow(2) + F.relu(area - float(max_area)).pow(2)
    return _reduce(loss, reduction)


def activity_contrast_loss(activity, target_std=0.15, reduction="mean"):
    """
    Encourage visible separation between active and inactive patches.

    This prevents every oscillator from living in a narrow band around 0.5,
    which makes threshold-based masks brittle.
    """
    if activity.dim() != 3:
        raise ValueError("activity must have shape [B, N, T].")
    if target_std < 0.0:
        raise ValueError("target_std must be non-negative.")

    std = activity.float().flatten(start_dim=1).std(dim=1)
    loss = F.relu(float(target_std) - std).pow(2)
    return _reduce(loss, reduction)


def phase_locking_value(theta, settle=0, combine="mean"):
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

    combine says what to do with the osc_dim components. "mean" averages the
    phases first, which is what this has always done and what gate_mode
    "phase_mean" does before the SNN, so a patch's group membership ends up
    carried by a single number. With 6.5 objects per image that number has to
    hold seven distinguishable bands. Measured on a trained checkpoint,
    computing synchrony per component and combining afterwards raises foreground
    ARI from 0.365 to 0.470, which is the first readout here to beat clustering
    the input features directly. "product" requires every component to agree,
    "min" is its softer form.
    """
    if combine not in {"mean", "product", "min", "component_mean"}:
        raise ValueError('combine must be "mean", "product", "min" or "component_mean".')
    if combine != "mean":
        if theta.dim() != 4:
            raise ValueError("theta must have shape [B, T, N, D].")
        per = torch.stack([
            phase_locking_value(theta[..., d:d + 1], settle=settle)
            for d in range(theta.size(-1))
        ])
        if combine == "product":
            return per.prod(dim=0)
        if combine == "min":
            return per.min(dim=0).values
        return per.mean(dim=0)
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        if int(settle) >= phase.size(1):
            raise ValueError("settle must be smaller than the number of steps.")
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
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        if int(settle) >= phase.size(1):
            raise ValueError("settle must be smaller than the number of steps.")
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
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        if int(settle) >= phase.size(1):
            raise ValueError("settle must be smaller than the number of steps.")
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
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        if int(settle) >= phase.size(1):
            raise ValueError("settle must be smaller than the number of steps.")
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
    if signal.dim() != 3:
        raise ValueError("signal must have shape [B, N, T].")
    trace = signal.float()
    if int(settle) > 0:
        if int(settle) >= trace.size(2):
            raise ValueError("settle must be smaller than the number of steps.")
        trace = trace[:, :, int(settle):]

    trace = trace - trace.mean(dim=2, keepdim=True)
    trace = trace / trace.norm(dim=2, keepdim=True).clamp_min(eps)
    return torch.bmm(trace, trace.transpose(1, 2)).abs().clamp(0.0, 1.0)


def graph_teacher_synchrony_loss(membrane, graph, settle=0, temperature=0.1):
    """Distill an image-conditioned graph into membrane-trace synchrony.

    The graph is a detached, unsupervised teacher. For every oscillator, its
    normalized outgoing edge weights define a neighbor distribution. The
    membrane trace yields a centered-synchrony distribution over the same
    neighbors. KL divergence trains the SNN path to preserve graph structure
    without using CLEVR instance masks or changing the core dynamics.
    """
    if membrane.ndim != 3 or graph.ndim != 3:
        raise ValueError("membrane and graph must have shapes [B,N,T] and [B,N,N].")
    batch, nodes, _ = membrane.shape
    if nodes < 2 or graph.shape != (batch, nodes, nodes):
        raise ValueError("graph must match the membrane batch and oscillator dimensions.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    synchrony = signal_synchrony(membrane, settle=settle)
    diagonal = torch.eye(nodes, dtype=torch.bool, device=membrane.device).unsqueeze(0)
    teacher = graph.detach().to(device=membrane.device, dtype=synchrony.dtype)
    teacher = teacher.clamp_min(0.0).masked_fill(diagonal, 0.0)
    row_mass = teacher.sum(dim=-1, keepdim=True)
    if bool((row_mass <= 0).any()):
        raise ValueError("Every graph row needs at least one non-self neighbor.")
    teacher = teacher / row_mass
    logits = (synchrony / float(temperature)).masked_fill(diagonal, -1e4)
    return F.kl_div(F.log_softmax(logits, dim=-1), teacher, reduction="none").sum(dim=-1).mean()


def patch_pool_rgb(images, grid_size):
    """[B, 3, H, W] images -> [B, grid*grid, 3] mean RGB per patch."""
    grid_h, grid_w = _parse_grid_size(grid_size)
    pooled = F.adaptive_avg_pool2d(images, (grid_h, grid_w))
    return pooled.flatten(2).transpose(1, 2)


def slot_reconstruction_loss(theta, target, num_slots=7, settle=0, temperature=0.3,
                            reduction="mean", eps=1e-8):
    """
    Require the phase grouping to explain the image.

    Every other loss here is a generic structure prior: make synchrony bimodal,
    do not collapse, keep groups contiguous. None of them knows what an object
    is, which is why targets like a fixed group count or a k-fold phase grid can
    be optimised perfectly while the task score falls to chance. The model can
    satisfy them with structure unrelated to objects.

    Reconstruction removes that freedom. Oscillators are softly assigned to slots
    by their phase, each slot takes the mean of the target over the patches
    assigned to it, and every patch is then rebuilt from its slot:

        phi   = preferred phase of each unit
        w     = softmax_k cos(phi_i - psi_k) / temperature      [B, N, K]
        slot  = sum_i w_ik * target_i / sum_i w_ik              [B, K, C]
        recon = sum_k w_ik * slot_k                             [B, N, C]

    A grouping that cuts across objects makes each slot mean a blur of unlike
    content, so the reconstruction is poor. A grouping that follows objects makes
    slots homogeneous and the reconstruction cheap. This is the signal
    slot-based methods use, and it is what the phase structure has been missing.

    Everything is differentiable, so it works despite the eventual readout
    (spectral clustering) being outside the graph.

    Args:
        theta:  [B, T, N, D] phase history.
        target: [B, N, C] what each patch should be rebuilt as. Prefer something
            external to the model, such as pooled RGB; reconstructing the
            model's own features invites a collapse to uniform features.
    """
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    if target.dim() != 3:
        raise ValueError("target must have shape [B, N, C].")

    phi = preferred_phase(theta, settle=settle)                      # [B, N]
    slots = torch.arange(int(num_slots), device=phi.device, dtype=phi.dtype)
    centres = 2.0 * math.pi * slots / float(num_slots)
    weights = (torch.cos(phi.unsqueeze(-1) - centres) / float(temperature)).softmax(dim=-1)

    mass = weights.sum(dim=1).clamp_min(eps)                         # [B, K]
    slot_value = torch.einsum("bnk,bnc->bkc", weights, target) / mass.unsqueeze(-1)
    reconstruction = torch.einsum("bnk,bkc->bnc", weights, slot_value)

    loss = (reconstruction - target).pow(2).mean(dim=(1, 2))
    scale = target.var(dim=(1, 2)).clamp_min(eps)                    # scale free
    return _reduce(loss / scale, reduction)


def oscillator_state_features(theta, settle=0, eps=1e-8):
    """
    Per-unit summary of the dynamics, as input to a clustering head. [B, N, 4]

    cos and sin of the preferred phase, the mean phase advance (the frequency
    the image wrote into the unit), and the resultant length, which says how
    reliably the unit holds a phase at all. Grouping here is by frequency and
    phase similarity, so these are the quantities the decision rests on.
    """
    if theta.dim() != 4:
        raise ValueError("theta must have shape [B, T, N, D].")
    phase = theta.mean(dim=-1)
    if int(settle) > 0:
        phase = phase[:, int(settle):]
    field = torch.atan2(
        torch.sin(phase).mean(dim=2, keepdim=True),
        torch.cos(phase).mean(dim=2, keepdim=True),
    )
    relative = phase - field
    cos_p = torch.cos(relative).mean(dim=1)
    sin_p = torch.sin(relative).mean(dim=1)
    resultant = (cos_p.pow(2) + sin_p.pow(2)).clamp_min(eps).sqrt()
    advance = (phase[:, 1:] - phase[:, :-1]).mean(dim=1)
    return torch.stack([cos_p / resultant, sin_p / resultant, advance, resultant], dim=-1)


def occupancy_floor_loss(assignment, floor=0.01, reduction="mean"):
    """
    Penalise a slot holding less than `floor` of the units, and nothing else.

    This guards the same failure the orthogonality term guards -- everything in
    one cluster -- without asking for clusters of equal size. Measured on CLEVR
    patches, where background is 94% of the grid, the balance term ranks the
    partitions backwards: the ground truth scores 0.897 against 0.003 for a
    random balanced partition, and no weight on it recovers the right order. The
    floor scores the ground truth at 0.052 and the single-cluster solution at
    0.689, so at a weight of 0.5 the ground truth becomes the objective's
    optimum on the spike synchrony matrix.
    """
    if assignment.dim() != 3:
        raise ValueError("assignment must have shape [B, N, K].")
    mass = assignment.sum(dim=1) / assignment.size(1)
    return _reduce((1.0 - mass / float(floor)).clamp_min(0.0).mean(dim=-1), reduction)


def mincut_loss(affinity, assignment, eps=1e-8, reduction="mean",
                ortho_weight=1.0, floor=0.0, floor_weight=0.0,
                entropy_weight=0.0):
    """
    Relaxed normalized cut on a soft assignment, from MinCutPool.

        L_cut   = -trace(Y^T A Y) / trace(Y^T D Y)
        L_ortho = || Y^T Y / ||Y^T Y||_F - I_K / sqrt(K) ||_F

    Why this rather than another hand-written term. Spectral clustering sits
    outside the loss and is not differentiable, so training shapes the pairwise
    synchrony while the metric scores a partition; the two are different objects.
    This puts the partition itself in the loss, and the head that produces the
    assignment becomes the readout.

    It is taken from the literature rather than invented here on purpose. Every
    objective this project wrote from scratch to encode object-ness -- a target
    group count, a k-fold phase grid, a reconstruction from phase slots -- was
    optimised perfectly and scored at chance, because a scalar summary of the
    affinity says nothing about *which* units belong together. The cut term is
    per-pair, and the orthogonality term is what stops the single-cluster
    solution that the earlier attempts all fell into.

    The balance term is optional because it is wrong for this data. Pass
    ortho_weight=0 with floor_weight>0 to swap it for an occupancy floor, which
    is what the measurement supports; see occupancy_floor_loss.

    Dropping the balance term opens a hole the floor does not cover. The cut is
    -1.0 for the uniform soft assignment, spreading every unit evenly over every
    slot, and that assignment gives each slot 1/K of the mass, so the floor is
    satisfied and the uniform solution becomes the optimum -- measured, it is
    exactly where training went. The floor guards empty slots; nothing guards
    diffuse ones. entropy_weight penalises the row entropy of the assignment,
    which is zero for any hard partition and maximal for the uniform one, so it
    removes the degenerate solution without changing how hard partitions rank
    against each other.

    Args:
        affinity:   [B, N, N] non-negative, e.g. a PLV or spike synchrony matrix.
        assignment: [B, N, K] rows summing to one.
    """
    if affinity.dim() != 3 or assignment.dim() != 3:
        raise ValueError("affinity must be [B, N, N] and assignment [B, N, K].")
    num_slots = assignment.size(-1)

    degree = affinity.sum(dim=-1)
    numerator = torch.einsum("bnk,bnm,bmk->bk", assignment, affinity, assignment).sum(-1)
    denominator = torch.einsum("bnk,bn,bnk->bk", assignment, degree, assignment).sum(-1)
    cut = -(numerator / denominator.clamp_min(eps))

    gram = torch.einsum("bnk,bnl->bkl", assignment, assignment)
    gram = gram / gram.norm(dim=(1, 2), keepdim=True).clamp_min(eps)
    target = torch.eye(num_slots, device=gram.device, dtype=gram.dtype) / (num_slots ** 0.5)
    ortho = (gram - target).norm(dim=(1, 2))

    total = cut + float(ortho_weight) * ortho
    if float(floor_weight) != 0.0:
        mass = assignment.sum(dim=1) / assignment.size(1)
        total = total + float(floor_weight) * (
            1.0 - mass / float(floor)
        ).clamp_min(0.0).mean(dim=-1)
    if float(entropy_weight) != 0.0:
        row_entropy = -(assignment.clamp_min(eps).log() * assignment).sum(dim=-1)
        total = total + float(entropy_weight) * (
            row_entropy.mean(dim=-1) / math.log(num_slots)
        )
    return _reduce(total, reduction)


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
    if grid_h * grid_w != num_nodes:
        raise ValueError(
            f"patch grid {grid_h}x{grid_w} does not match {num_nodes} oscillators."
        )
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
        slot_reconstruction_weight=0.0,
        slot_num_slots=7,
        slot_temperature=0.3,
        phase_quantization_weight=0.0,
        phase_spread_weight=0.0,
        phase_num_slots=7.0,
        plv_target_density=0.25,
        plv_target_groups=7.0,
        mincut_weight=0.0,
        mincut_ortho_weight=0.0,
        mincut_floor=0.01,
        mincut_floor_weight=0.5,
        mincut_entropy_weight=0.5,
    ):
        super().__init__()
        self.mincut_weight = float(mincut_weight)
        self.mincut_ortho_weight = float(mincut_ortho_weight)
        self.mincut_floor = float(mincut_floor)
        self.mincut_floor_weight = float(mincut_floor_weight)
        self.mincut_entropy_weight = float(mincut_entropy_weight)
        self.plv_bimodality_weight = float(plv_bimodality_weight)
        self.plv_balance_weight = float(plv_balance_weight)
        self.plv_coherence_weight = float(plv_coherence_weight)
        self.plv_group_count_weight = float(plv_group_count_weight)
        self.plv_collapse_weight = float(plv_collapse_weight)
        self.slot_reconstruction_weight = float(slot_reconstruction_weight)
        self.slot_num_slots = int(slot_num_slots)
        self.slot_temperature = float(slot_temperature)
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
                plv_settle=0, recon_target=None, assignment=None):
        device, dtype = _infer_device_dtype(spikes, sc, plv)
        total = torch.zeros((), device=device, dtype=dtype)
        parts = {}

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
            "slot_reconstruction": self.slot_reconstruction_weight,
            "phase_quantization": self.phase_quantization_weight,
            "phase_spread": self.phase_spread_weight,
            "mincut": self.mincut_weight,
        }

        def add(name, available, term):
            """
            Evaluate a term only when it is switched on.

            Every term used to be computed and then multiplied by its weight, so
            a term at weight 0 still ran. Two of them build [B, N, N, T]
            tensors, which is 12 GB at a 32x32 grid, and that is what made a
            1024-oscillator run go out of memory even with every spiking term
            disabled. It also explains plv_group_count printing 3e21 in runs
            that never used it.
            """
            if available and float(weights[name]) != 0.0:
                parts[name] = term()

        add("phase_quantization", theta is not None,
            lambda: phase_quantization_loss(theta, num_slots=self.phase_num_slots,
                                            settle=plv_settle))
        add("phase_spread", theta is not None,
            lambda: phase_spread_loss(theta, settle=plv_settle))
        add("slot_reconstruction", theta is not None and recon_target is not None,
            lambda: slot_reconstruction_loss(theta, recon_target,
                                             num_slots=self.slot_num_slots,
                                             settle=plv_settle,
                                             temperature=self.slot_temperature))

        add("plv_bimodality", plv is not None, lambda: plv_bimodality_loss(plv))
        add("plv_balance", plv is not None,
            lambda: plv_group_balance_loss(plv, target_density=self.plv_target_density))
        add("plv_group_count", plv is not None,
            lambda: plv_group_count_loss(plv, target_groups=self.plv_target_groups))
        add("plv_collapse", plv is not None, lambda: plv_collapse_loss(plv))
        add("plv_coherence", plv is not None and self.patch_grid_size is not None,
            lambda: plv_spatial_coherence_loss(plv, patch_grid_size=self.patch_grid_size))

        add("spike_rate", spikes is not None,
            lambda: spike_rate_loss(spikes, target_rate=self.spike_target_rate))
        add("spike_smooth", spikes is not None,
            lambda: spike_temporal_smoothness_loss(spikes))
        add("spike_diversity", spikes is not None, lambda: spike_diversity_loss(spikes))
        add("structural", spikes is not None and sc is not None,
            lambda: structural_consistency_loss(spikes, sc))
        add("sample_diversity", spikes is not None,
            lambda: sample_activity_diversity_loss(spikes))
        add("temporal_balance", spikes is not None,
            lambda: temporal_activity_balance_loss(spikes))
        add("activity_confidence", spikes is not None,
            lambda: activity_confidence_loss(spikes))
        add("activity_area", spikes is not None,
            lambda: activity_area_loss(spikes, min_area=self.activity_min_area,
                                       max_area=self.activity_max_area))
        add("activity_contrast", spikes is not None,
            lambda: activity_contrast_loss(spikes, target_std=self.activity_target_std))
        add("spatial_compactness", spikes is not None and self.patch_grid_size is not None,
            lambda: spatial_compactness_loss(spikes, patch_grid_size=self.patch_grid_size))

        add("mincut", assignment is not None and plv is not None,
            lambda: mincut_loss(plv, assignment,
                                ortho_weight=self.mincut_ortho_weight,
                                floor=self.mincut_floor,
                                floor_weight=self.mincut_floor_weight,
                                entropy_weight=self.mincut_entropy_weight))
        add("object_overlap", object_groups is not None,
            lambda: object_overlap_loss(object_groups,
                                        num_oscillators=spikes.size(1) if spikes is not None else None,
                                        device=device))

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
