"""
Image-conditioned coupling graph for Kuramoto oscillators -- a drop-in
replacement for a fixed structural-connectivity matrix.

WHY
    A static SC cannot express object membership. Which patches belong together
    is a property of the current image, not of the dataset, so one fixed [N, N]
    matrix cannot represent it. Measured on CLEVR at a 16x16 patch grid, against
    colour-derived object masks:

        static Pearson SC ..................... contributed nothing over K = 0
        spatial-only graph .................... ARI -0.001, no better than chance
        image-conditioned sparse graph ........ the only variant that helped

WHAT MATTERS (all measured, not guessed)
    image conditioning  required; a fixed graph scores the same as no coupling
    sparsity            required; dense graphs collapse into global synchrony as
                        coupling grows (ARI -0.0002 at K = 256)
    no lower bound      a 0.5 floor, which keeps dissimilar patches half-coupled,
                        cost ARI 0.027 -> -0.0002
    feature-only        mixing spatial proximity into the weights hurt
                        (+0.091 vs +0.105); as a *prior on the logits* it helps
    learned             0.293 learned vs 0.056 for a hand-written kernel

PREREQUISITE -- READ THIS
    The graph does nothing on its own. With the usual sensory drive
    kappa*sin(gamma - theta) the system settles to a fixed point and globally
    synchronises (measured |dtheta| 0.006 by step 31, PLV 1.000, order parameter
    0.977). Every oscillator then carries the same phase and no group exists, so
    no coupling design can help.

    Letting the image set oscillator FREQUENCIES is what makes grouping
    possible, because two oscillators lock when |domega| < K_eff:

        omega_eff = omega + freq_gain * gamma        freq_gain ~ 2.0

    kuramoto_step below includes this. Without it the graph is inert.

USAGE
    graph = ImageConditionedGraph(in_channels=8, grid_size=16,
                                  spatial_decay=0.861, top_k=8)
    A = graph(gamma)                      # gamma [B, C, N] -> A [B, N, N]
    theta = kuramoto_step(theta, drive, A, omega, kappa, freq_gain=2.0)

    Set the Kuramoto K to N so that the effective coupling per oscillator equals
    the graph's coupling_gain; graph.effective_coupling(K, N) reports it.

Run this file directly for a self-test.
"""

if __package__:
    from . import error_bound
else:
    import error_bound


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageConditionedGraph(nn.Module):
    """
    Learned per-image coupling graph for the Kuramoto layer.

    A fixed structural-connectivity matrix cannot express object membership,
    because which patches belong together is a property of the current image,
    not of the dataset. This produces one graph per sample instead:

        gamma [B, C, N] -> projection -> cosine similarity -> top-k -> A [B, N, N]

    Measured on CLEVR at a 16x16 patch grid against colour-derived object masks
    (ARI of spectral clusters, and mean PLV within an object minus between):

    - image conditioning is required: a purely spatial graph scored ARI -0.001,
      no better than chance;
    - sparsity is required: dense graphs collapse into global synchrony as
      coupling grows (ARI -0.0002 at K=256);
    - no lower bound on the weights: a 0.5 floor, which keeps dissimilar patches
      half-coupled, cost ARI 0.027 -> -0.0002 at K=256;
    - learning beats a hand-written kernel: 0.293 against 0.056.

    Two optional structural additions, both off by default so an existing
    checkpoint still loads and behaves identically:

    ``spatial_decay`` adds a distance prior to the logits. Objects are connected
    regions, which the features alone do not know. A hand-written spatial x
    similarity kernel scored 0.056 against 0.038 for this module at random
    initialisation, so it is a better starting point than noise.

    ``feedback_strength`` lets the graph track the synchrony it is producing.
    Without it the graph is computed once and held for the whole rollout, and
    the resulting PLV matrix stays nearly uniform (mean 0.763): a static graph
    has no way to sharpen a cluster once it starts to form. With it, oscillators
    that have stayed in phase couple more strongly, which is the positive
    feedback that lets groups crystallise.
    """

    def __init__(
        self,
        in_channels,
        hidden_dim=16,
        top_k=8,
        coupling_gain=8.0,
        learn_gain=True,
        temperature=0.1,
        grid_size=None,
        spatial_decay=None,
        feedback_strength=0.0,
        feedback_momentum=0.9,
    ):
        super().__init__()
        error_bound.validate_graph_generator_image_conditioned_graph_init(top_k)
        self.top_k = int(top_k)
        self.feedback_momentum = float(feedback_momentum)
        self.projection = nn.Linear(int(in_channels), int(hidden_dim), bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())
        gain = torch.tensor(float(coupling_gain)).log()
        if learn_gain:
            self.log_coupling_gain = nn.Parameter(gain)
        else:
            self.register_buffer("log_coupling_gain", gain)

        self.spatial_rate = None
        if spatial_decay is not None:
            error_bound.validate_graph_generator_image_conditioned_graph_init_2(grid_size, spatial_decay)
            grid_h, grid_w = _pair(grid_size)
            index = torch.arange(grid_h * grid_w)
            rows = (index // grid_w).float()
            cols = (index % grid_w).float()
            distance = ((rows[:, None] - rows[None, :]) ** 2
                        + (cols[:, None] - cols[None, :]) ** 2).sqrt()
            self.register_buffer("grid_distance", distance)
            # logits contribute -rate * distance, so rate = -log(decay) > 0.
            rate = -math.log(float(spatial_decay))
            self.spatial_rate = nn.Parameter(torch.tensor(_inverse_softplus(rate)))

        self.feedback_strength = None
        if float(feedback_strength) != 0.0:
            self.feedback_strength = nn.Parameter(torch.tensor(float(feedback_strength)))

    @property
    def uses_feedback(self):
        return self.feedback_strength is not None

    def forward(self, gamma, alignment=None):
        """
        gamma: [B, C, N] channel-major patch features, or [B, N, C] with C last.
        alignment: optional [B, N, N] in [-1, 1], sustained phase alignment.

        Returns A [B, N, N], symmetric and non-negative, rows summing to the
        coupling gain before symmetrization.
        """
        error_bound.validate_graph_generator_image_conditioned_graph_forward(gamma)
        if (gamma.size(1) == self.projection.in_features
                and gamma.size(2) != self.projection.in_features):
            gamma = gamma.transpose(1, 2)  # [B, N, C]

        z = F.normalize(self.projection(gamma), dim=-1)
        logits = torch.bmm(z, z.transpose(1, 2)) / self.log_temperature.exp().clamp_min(1e-3)

        if self.spatial_rate is not None:
            rate = F.softplus(self.spatial_rate)
            logits = logits - rate * self.grid_distance.unsqueeze(0)

        if self.feedback_strength is not None and alignment is not None:
            logits = logits + self.feedback_strength * alignment

        num_nodes = logits.size(-1)
        values, indices = logits.topk(min(self.top_k, num_nodes), dim=-1)
        weights = values.softmax(dim=-1) * self.log_coupling_gain.exp()
        adjacency = torch.zeros_like(logits).scatter_(-1, indices, weights)
        return 0.5 * (adjacency + adjacency.transpose(1, 2))

    def initial_alignment(self, batch_size, num_nodes, device):
        return torch.zeros(batch_size, num_nodes, num_nodes, device=device)

    def update_alignment(self, alignment, theta):
        """
        Track sustained phase alignment as an exponential moving average of
        cos(theta_i - theta_j).

        Detached on purpose: the feedback shapes the dynamics, but gradients
        reach the graph through its feature and spatial terms instead of through
        a 64-step chain of adjacency matrices.
        """
        phase = theta.mean(dim=-1)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        current = (cos_p.unsqueeze(2) * cos_p.unsqueeze(1)
                   + sin_p.unsqueeze(2) * sin_p.unsqueeze(1))
        momentum = self.feedback_momentum
        return (momentum * alignment + (1.0 - momentum) * current).detach()

    def effective_coupling(self, k, num_regions):
        """Coupling per oscillator once the Kuramoto layer applies its K / N."""
        return float(k) / float(num_regions) * float(self.log_coupling_gain.exp())


def _inverse_softplus(value):
    return math.log(math.expm1(float(value)))


def _pair(value):
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("grid_size must be an int or a pair of ints.")


# ---------------------------------------------------------------------------
# Minimal Kuramoto step, included so the graph can be run without the rest of
# the project. Mirrors the update the measurements above were taken with.
# ---------------------------------------------------------------------------

def kuramoto_step(theta, drive, adjacency, omega, kappa, freq_gain=2.0, k=None, dt=0.1):
    """
    theta     [B, N, D]  oscillator phases
    drive     [B, N, D]  sensory drive, already mapped onto a phase range
    adjacency [B, N, N]  from ImageConditionedGraph
    omega     [N, D]     intrinsic frequency
    kappa     [N, D]     drive stiffness
    freq_gain scalar     image -> frequency. At 0 the model globally
                         synchronises and no grouping is possible.
    """
    num_nodes = theta.size(1)
    k = float(num_nodes) if k is None else float(k)
    a_sym = torch.relu(0.5 * (adjacency + adjacency.transpose(1, 2))) + 1e-6

    phase_diff = theta.unsqueeze(1) - theta.unsqueeze(2)
    coupling = (k / num_nodes) * torch.sum(a_sym.unsqueeze(-1) * torch.sin(phase_diff), dim=2)

    omega_eff = omega + freq_gain * drive
    return theta + dt * (omega_eff + coupling + kappa * torch.sin(drive - theta))


def phase_locking_value(theta_hist, settle=0):
    """[B, T, N, D] -> [B, N, N] pairwise PLV. This is the readout that works."""
    phase = theta_hist.mean(dim=-1)
    if settle:
        phase = phase[:, settle:]
    steps = phase.size(1)
    cos_t, sin_t = torch.cos(phase), torch.sin(phase)
    real = (cos_t.transpose(1, 2) @ cos_t + sin_t.transpose(1, 2) @ sin_t) / steps
    imag = (sin_t.transpose(1, 2) @ cos_t - cos_t.transpose(1, 2) @ sin_t) / steps
    return torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Diagnostics
#
# Every number below comes with the reading that makes it actionable. Most need
# no ground truth: the failure modes this project hit -- a frozen fixed point,
# global synchrony, a dense graph collapsing under coupling -- are all visible
# without labels, and each one makes object grouping impossible regardless of
# what the loss is doing.
# ---------------------------------------------------------------------------

def order_parameters(theta_hist, num_slots=7, settle=0):
    """
    R1 = |<exp(i*theta)>_units|      1 when every unit shares one phase
    Rk = |<exp(i*k*theta)>_units|    1 when phases sit on a k-fold grid

    Reference free on purpose. Measuring a phase against the mean field breaks
    in exactly the state you want, because k evenly spread phases cancel and
    leave the field angle undefined.
    """
    phase = theta_hist.mean(dim=-1)[:, settle:]
    resultant = lambda a: (torch.cos(a).mean(2).pow(2) + torch.sin(a).mean(2).pow(2)).clamp_min(1e-12).sqrt()
    return float(resultant(phase).mean()), float(resultant(num_slots * phase).mean())


def effective_group_count(plv):
    """N^2 / ||PLV||_F^2 counts groups: 1 for global synchrony, N for none."""
    return float((plv.size(-1) ** 2) / plv.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)).__float__() \
        if plv.dim() == 2 else float(((plv.size(-1) ** 2) / plv.pow(2).sum(dim=(1, 2)).clamp_min(1e-8)).mean())


def morans_i(field, grid_size):
    """Spatial autocorrelation of a per-unit value. 1 smooth blobs, 0 noise."""
    grid_h, grid_w = _pair(grid_size)
    x = field.view(-1, grid_h, grid_w)
    c = x - x.mean(dim=(1, 2), keepdim=True)
    num = (c[:, :, :-1] * c[:, :, 1:]).sum((1, 2)) + (c[:, :-1, :] * c[:, 1:, :]).sum((1, 2))
    pairs = 2 * grid_h * (grid_w - 1) + 2 * grid_w * (grid_h - 1)
    return float((((grid_h * grid_w) / pairs) * num / c.pow(2).sum((1, 2)).clamp_min(1e-12)).mean())


def _kmeans(x, k, iters=50, restarts=5, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best_labels, best = None, float("inf")
    for _ in range(restarts):
        centres = x[torch.randint(x.size(0), (1,), generator=gen)]
        for _ in range(k - 1):
            d = torch.cdist(x, centres).min(dim=1).values ** 2
            centres = torch.cat([centres, x[torch.multinomial(d / d.sum().clamp_min(1e-12), 1, generator=gen)]])
        labels = torch.zeros(x.size(0), dtype=torch.long)
        for _ in range(iters):
            new = torch.cdist(x, centres).argmin(dim=1)
            if torch.equal(new, labels):
                break
            labels = new
            for c in range(k):
                if (labels == c).any():
                    centres[c] = x[labels == c].mean(dim=0)
        inertia = float(torch.cdist(x, centres).min(dim=1).values.pow(2).sum())
        if inertia < best:
            best, best_labels = inertia, labels.clone()
    return best_labels


def spectral_cluster(affinity, k):
    """Normalized spectral clustering on a dense affinity matrix -> [N] labels."""
    a = affinity.detach().double().clone()
    a.fill_diagonal_(0.0)
    a = a.clamp_min(0.0)
    d = a.sum(1).clamp_min(1e-9).pow(-0.5)
    m = d.unsqueeze(1) * a * d.unsqueeze(0)
    m = 0.5 * (m + m.T) + 1e-6 * torch.eye(m.size(0), dtype=m.dtype, device=m.device)
    try:
        _, vectors = torch.linalg.eigh(m)
    except Exception:
        return torch.zeros(m.size(0), dtype=torch.long)
    emb = vectors[:, -k:]
    return _kmeans((emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-9)).float(), k)


def adjusted_rand_index(a, b):
    """Permutation invariant agreement between two partitions. 0 = chance."""
    _, ia = torch.unique(a, return_inverse=True)
    _, ib = torch.unique(b, return_inverse=True)
    table = torch.zeros(int(ia.max()) + 1, int(ib.max()) + 1)
    for i, j in zip(ia.tolist(), ib.tolist()):
        table[i, j] += 1
    comb = lambda n: n * (n - 1) / 2
    sij, sa, sb = comb(table).sum(), comb(table.sum(1)).sum(), comb(table.sum(0)).sum()
    total = comb(torch.tensor(float(len(a))))
    expected = sa * sb / total.clamp_min(1e-9)
    return float((sij - expected) / (0.5 * (sa + sb) - expected).clamp_min(1e-9))


def diagnose(theta_hist, adjacency=None, labels=None, grid_size=None,
             settle=None, num_slots=7, dt=0.1, coupling_k=None):
    """
    Print a readable report on whether this configuration can bind at all.

    theta_hist [B, T, N, D]   phase history
    adjacency  [B, N, N]      optional, the coupling graph
    labels     [B, N]         optional ground truth, -1 for background
    grid_size                 optional, enables the spatial checks
    """
    steps, num_nodes = theta_hist.size(1), theta_hist.size(2)
    settle = steps // 2 if settle is None else settle
    phase = theta_hist.mean(dim=-1)
    report = {}

    print("=" * 68)
    print("DYNAMICS -- is anything oscillating?")
    print("=" * 68)
    advance = (phase[:, 1:] - phase[:, :-1])
    last = float(advance[:, -1].abs().mean())
    per_unit = advance.mean(dim=1)
    report["step_change_final"] = last
    # the signed mean cancels when frequencies straddle zero, so use the typical
    # rotation speed instead
    report["period_steps"] = float(2 * math.pi / max(float(per_unit.abs().median()), 1e-9))
    report["frequency_spread"] = float(per_unit.std())
    print("  |dtheta| at the last step      %.5f rad" % last)
    if last < 0.01:
        print("    -> FROZEN. The phases have settled to a fixed point. Binding by")
        print("       synchrony needs sustained phase relations, so nothing can work")
        print("       here. Raise freq_gain, or lower kappa.")
    else:
        print("    -> oscillating")
    print("  typical period                 %.0f steps  (median unit)" % report["period_steps"])
    print("  spread of unit frequencies     %.4f rad/step" % report["frequency_spread"])
    if report["frequency_spread"] > 1e-6:
        sep = 2 * math.pi / (report["frequency_spread"] * math.sqrt(2))
        report["steps_to_separate"] = sep
        print("    -> two typical units drift a full cycle apart in ~%.0f steps," % sep)
        print("       and the rollout is %d steps, so ~%.2f of a cycle." % (steps, steps / sep))
        if steps / sep < 0.25:
            print("       TOO SLOW: they barely separate. Raise freq_gain or run longer.")
        elif steps / sep > 8:
            print("       TOO FAST: even similar units separate. Lower freq_gain.")

    print()
    print("=" * 68)
    print("SYNCHRONY -- are there groups, one group, or none?")
    print("=" * 68)
    plv = phase_locking_value(theta_hist, settle=settle)
    off = ~torch.eye(num_nodes, dtype=torch.bool, device=plv.device)
    mean_plv = float(plv[:, off].mean())
    r1, rk = order_parameters(theta_hist, num_slots=num_slots, settle=settle)
    groups = effective_group_count(plv)
    report.update(mean_plv=mean_plv, R1=r1, Rk=rk, effective_groups=groups)
    print("  mean pairwise PLV              %.3f" % mean_plv)
    if mean_plv > 0.95:
        print("    -> GLOBAL SYNCHRONY. Everything is locked to everything, so no")
        print("       group is distinguishable. This is the dominant failure mode.")
    elif mean_plv < 0.05:
        print("    -> INCOHERENT. Nothing is locked to anything; there are no groups")
        print("       to read out. Lower freq_gain or raise the coupling.")
    else:
        print("    -> in the useful range")
    print("  R1 (one shared phase)          %.3f   %s" % (r1, "<- 1.0 means total collapse" if r1 > 0.9 else ""))
    print("  R%-2d (k-fold phase grid)        %.3f" % (num_slots, rk))
    print("  effective number of groups     %.1f   (1 = all one group, %d = none)" % (groups, num_nodes))

    if adjacency is not None:
        print()
        print("=" * 68)
        print("GRAPH -- sparse, image conditioned, and not saturating?")
        print("=" * 68)
        nonzero = float((adjacency > 1e-8).float().sum(-1).mean())
        rowsum = float(adjacency.sum(-1).mean())
        symmetric = bool(torch.allclose(adjacency, adjacency.transpose(1, 2), atol=1e-5))
        varies = (float((adjacency[0] - adjacency[1]).abs().mean()) if adjacency.size(0) > 1 else float("nan"))
        report.update(nonzeros_per_row=nonzero, row_sum=rowsum, image_conditioned=varies)
        print("  nonzeros per row               %.1f of %d" % (nonzero, num_nodes))
        if nonzero > num_nodes / 4:
            print("    -> DENSE. Dense graphs collapse into global synchrony as coupling")
            print("       grows; sparsify with top_k.")
        print("  row sum (coupling gain)        %.2f" % rowsum)
        print("  symmetric                      %s" % symmetric)
        if adjacency.size(0) > 1:
            print("  differs between images         %.4f   %s"
                  % (varies, "<- 0 means a fixed graph, which cannot express objects" if varies < 1e-6 else ""))
        if coupling_k is not None:
            k_eff = float(coupling_k) / num_nodes * rowsum
            report["k_eff"] = k_eff
            print("  effective coupling K_eff       %.2f" % k_eff)
            if report["frequency_spread"] > 1e-6:
                ratio = (report["frequency_spread"] / dt) / max(k_eff, 1e-9)
                print("  sigma_omega / K_eff            %.2f   %s" % (
                    ratio, "(<0.5 tends to global sync, >5 tends to incoherence)"))

    if grid_size is not None:
        print()
        print("=" * 68)
        print("SPATIAL -- are the synchronised sets contiguous?")
        print("=" * 68)
        field = plv.mean(dim=2)
        moran = morans_i(field, grid_size)
        report["morans_i"] = moran
        print("  Moran's I of the PLV field     %.3f   (1 smooth blobs, 0 noise)" % moran)
        if moran < 0.2:
            print("    -> weak: synchronised units are scattered rather than forming regions")

    if labels is not None:
        print()
        print("=" * 68)
        print("GROUND TRUTH")
        print("=" * 68)
        within, between, aris = [], [], []
        for i in range(labels.size(0)):
            lab = labels[i]
            if int(lab.max()) < 1:
                continue
            eye = torch.eye(num_nodes, dtype=torch.bool, device=lab.device)
            same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0) & ~eye
            diff = (lab[:, None] != lab[None, :]) & (lab[:, None] >= 0) & (lab[None, :] >= 0)
            if same.any():
                within.append(float(plv[i][same].mean()))
            if diff.any():
                between.append(float(plv[i][diff].mean()))
            g = lab.clone()
            g[g < 0] = int(g.max()) + 1
            g = torch.unique(g, return_inverse=True)[1]
            aris.append(adjusted_rand_index(spectral_cluster(plv[i].cpu(), int(g.max()) + 1), g.cpu()))
        mean = lambda x: sum(x) / len(x) if x else float("nan")
        report.update(binding_within=mean(within), binding_between=mean(between), ari=mean(aris))
        print("  PLV within an object           %.3f" % mean(within))
        print("  PLV between objects            %.3f" % mean(between))
        print("  binding gap                    %+.4f" % (mean(within) - mean(between)))
        print("  ARI of spectral clusters       %.4f   (0 = chance)" % mean(aris))
        print()
        print("  Note: the gap failed to predict ARI four times in this project.")
        print("  Trust ARI; treat the gap as descriptive only.")

    print()
    return report


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, channels, grid = 4, 8, 16
    num_nodes, osc_dim, steps = grid * grid, 4, 128

    gamma = torch.randn(batch, channels, num_nodes)
    graph = ImageConditionedGraph(in_channels=channels, grid_size=grid,
                                  spatial_decay=0.861, top_k=8)
    adjacency = graph(gamma)

    drive = math.pi * torch.tanh(
        F.normalize(gamma.transpose(1, 2), dim=-1)[..., :osc_dim] * 3.0)
    omega = torch.randn(num_nodes, osc_dim) * 0.1
    kappa = torch.ones(num_nodes, osc_dim)

    for freq_gain in (0.0, 2.0):
        print()
        print("#" * 68)
        print("# freq_gain = %.1f" % freq_gain)
        print("#" * 68)
        theta, hist = drive.clone(), []
        for _ in range(steps):
            theta = kuramoto_step(theta, drive, adjacency, omega, kappa,
                                  freq_gain=freq_gain, k=num_nodes)
            hist.append(theta)
        diagnose(torch.stack(hist, dim=1), adjacency=adjacency,
                 grid_size=grid, coupling_k=num_nodes)
