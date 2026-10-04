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
        geodesic_steps=0,
        geodesic_radius=1.5,
        geodesic_contrast=2.0,
        geodesic_temperature=0.5,
        geodesic_cap=16.0,
        feedback_strength=0.0,
        feedback_momentum=0.9,
    ):
        super().__init__()
        if int(top_k) < 1:
            raise ValueError("top_k must be at least 1.")
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
            if grid_size is None:
                raise ValueError("grid_size is required when spatial_decay is set.")
            if not 0.0 < float(spatial_decay) < 1.0:
                raise ValueError("spatial_decay must lie in (0, 1).")
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

        self.geodesic_steps = int(geodesic_steps)
        self.geodesic_radius = float(geodesic_radius)
        self.geodesic_temperature = float(geodesic_temperature)
        self.geodesic_cap = float(geodesic_cap)
        self.geodesic_contrast = nn.Parameter(
            torch.tensor(_inverse_softplus(float(geodesic_contrast)))
        ) if int(geodesic_steps) > 0 else None

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
        if gamma.dim() != 3:
            raise ValueError("gamma must have shape [B, C, N] or [B, N, C].")
        if (gamma.size(1) == self.projection.in_features
                and gamma.size(2) != self.projection.in_features):
            gamma = gamma.transpose(1, 2)  # [B, N, C]

        z = F.normalize(self.projection(gamma), dim=-1)
        logits = torch.bmm(z, z.transpose(1, 2)) / self.log_temperature.exp().clamp_min(1e-3)

        if self.spatial_rate is not None:
            rate = F.softplus(self.spatial_rate)
            distance = self.grid_distance.unsqueeze(0)
            if self.geodesic_steps > 0:
                distance = self._geodesic_distance(z, distance)
            logits = logits - rate * distance

        if self.feedback_strength is not None and alignment is not None:
            logits = logits + self.feedback_strength * alignment

        num_nodes = logits.size(-1)
        values, indices = logits.topk(min(self.top_k, num_nodes), dim=-1)
        weights = values.softmax(dim=-1) * self.log_coupling_gain.exp()
        adjacency = torch.zeros_like(logits).scatter_(-1, indices, weights)
        return 0.5 * (adjacency + adjacency.transpose(1, 2))

    def _geodesic_distance(self, z, euclidean, eps=1e-6):
        """
        Patch distance measured along the image rather than through it.

        Euclidean decay cannot separate two objects of the same colour: they look
        alike to the feature term and sit a few patches apart, so they stay
        linked. Measured on a trained checkpoint, the graph wired two
        same-coloured objects together at 0.418 mean edge weight against 0.749
        inside an object, and 83% of CLEVR scenes contain a repeated colour.

        Walking between them crosses background, so the route is what tells them
        apart. Each step costs Euclidean distance times a dissimilarity factor,
        so a step onto a patch that looks different is expensive; the shortest
        route is then found by repeated min-plus relaxation, which is the
        Floyd-Warshall update restricted to a few rounds and kept differentiable
        by a soft minimum.
        """
        similarity = torch.bmm(z, z.transpose(1, 2)).clamp(-1.0, 1.0)
        # a step is cheap between patches that look alike, expensive otherwise
        step = euclidean * (1.0 + F.softplus(self.geodesic_contrast) * (1.0 - similarity))
        # only immediate neighbours are steps; the rest must be reached by walking.
        # The barrier is a large finite number rather than inf: inf makes the
        # softened minimum produce NaN gradients, which arrive at the projection
        # as exactly zero and silently detach the graph from the loss.
        step = torch.where(euclidean <= self.geodesic_radius, step,
                           torch.full_like(step, float(self.geodesic_cap)))
        distance = step
        for _ in range(int(self.geodesic_steps)):
            # min over intermediate nodes, softened so the gradient survives
            through = distance.unsqueeze(2) + distance.unsqueeze(1)
            relaxed = -self.geodesic_temperature * torch.logsumexp(
                -through / self.geodesic_temperature, dim=-1)
            distance = torch.minimum(distance, relaxed)
        return distance.clamp(max=float(self.geodesic_cap))

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
