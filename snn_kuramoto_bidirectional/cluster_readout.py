"""
A differentiable readout that turns oscillator dynamics into a soft partition.

Until now the partition came from spectral clustering run after training, which
is not differentiable, so the objective shaped the pairwise synchrony while the
metric scored a grouping. Nothing connected the two: a phase field could be
pushed into any structure that scored well on an aggregate statistic and still
produce a poor partition, which is exactly what happened to every hand-written
"object-ness" term this project tried.

This module produces the grouping inside the graph, so the loss can score the
partition itself and the gradient reaches the dynamics that produced it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_function import oscillator_state_features


class ClusterReadout(nn.Module):
    """
    Soft k-means over per-oscillator dynamics features. [B, N, num_slots]

    The centroids are re-estimated per image rather than learned as fixed
    prototypes. A shared MLP would have to cut the phase circle at image
    independent boundaries, which splits any object that happens to straddle a
    boundary; letting the centroids move to wherever the image put its groups
    avoids that. Learned queries only supply the starting point, and the
    embedding the distances are measured in is learned.

    Args:
        num_slots: partition size. Slots the image does not need stay empty,
            though the orthogonality term in the mincut loss pushes against
            that, so it is worth keeping near the typical object count.
        num_iters: soft k-means refinement rounds.
    """

    def __init__(self, num_slots=7, embed_dim=16, num_iters=3, temperature=0.5,
                 feature_source="phase", signal_dim=None):
        super().__init__()
        if int(num_slots) < 2:
            raise ValueError("num_slots must be at least 2.")
        if int(num_iters) < 1:
            raise ValueError("num_iters must be at least 1.")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive.")
        if feature_source not in {"phase", "signal"}:
            raise ValueError('feature_source must be "phase" or "signal".')
        if feature_source == "signal" and signal_dim is None:
            raise ValueError("signal_dim is required when feature_source is signal.")
        self.feature_source = feature_source
        self.num_slots = int(num_slots)
        self.num_iters = int(num_iters)
        self.temperature = float(temperature)

        # cos(phase), sin(phase), phase advance, resultant length. They are
        # standardized across units first: the phase terms span [-1, 1] while a
        # phase advance is a fraction of a radian per step, so without it the
        # frequency the image wrote into each unit -- the quantity groups
        # actually differ by once freq_gain is on -- is swamped in the
        # embedding by the two phase channels.
        # In "signal" mode the head reads each unit's own trace instead, which is
        # what the architecture actually calls for: the readout is meant to be
        # "units that spike together are one object", and a shared linear map
        # preserves the inner products that co-firing is measured by, so an
        # untrained head already groups by spike similarity and training only
        # sharpens it.
        in_dim = 4 if feature_source == "phase" else int(signal_dim)
        self.feature_norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Centroids are seeded from the data, farthest-point style, not from
        # learned queries. Learned queries were measured to collapse: with a
        # diffuse first assignment the centroid update averages every slot onto
        # the data mean, after which the assignment is exactly uniform, and
        # uniform is a stationary point of the objective, so training sits there
        # at row entropy 1.000 and never leaves. Seeding from actual units
        # cannot produce identical centroids.

    def forward(self, theta=None, settle=0, features=None, signal=None):
        """
        Args:
            theta: [B, T, N, D] phase history.
            features: [B, N, 4] precomputed oscillator_state_features, for when
                the dynamics are frozen and only the head is being trained.
            signal: [B, N, T] per-unit trace, in "signal" mode.
        Returns:
            [B, N, num_slots] soft assignment, rows summing to one.
        """
        if self.feature_source == "signal":
            if signal is None:
                raise ValueError('signal is required when feature_source is "signal".')
            feats = signal.float()[:, :, int(settle):]
        else:
            if (theta is None) == (features is None):
                raise ValueError("pass exactly one of theta or features.")
            feats = (
                oscillator_state_features(theta, settle=settle)
                if features is None else features
            )
        feats = self.feature_norm(
            (feats - feats.mean(dim=1, keepdim=True))
            / feats.std(dim=1, keepdim=True).clamp_min(1e-6)
        )
        z = F.normalize(self.embed(feats), dim=-1)                    # [B, N, E]

        centroids = self._seed_centroids(z)                           # [B, K, E]
        for step in range(self.num_iters):
            logits = torch.einsum("bne,bke->bnk", z, centroids) / self.temperature
            assignment = logits.softmax(dim=-1)
            if step == self.num_iters - 1:
                return assignment
            mass = assignment.sum(dim=1).clamp_min(1e-8).unsqueeze(-1)
            centroids = F.normalize(
                torch.einsum("bnk,bne->bke", assignment, z) / mass, dim=-1
            )

    def _seed_centroids(self, z):
        """Farthest-point seeds: start away from the mean, then keep the point
        least similar to everything already chosen."""
        mean = F.normalize(z.mean(dim=1, keepdim=True), dim=-1)
        similarity = torch.einsum("bne,bme->bnm", z, mean).squeeze(-1)
        picked = [similarity.argmin(dim=1)]
        chosen = torch.einsum("bne,bn->be", z, F.one_hot(picked[0], z.size(1)).to(z.dtype))
        best = torch.einsum("bne,be->bn", z, F.normalize(chosen, dim=-1))
        for _ in range(self.num_slots - 1):
            index = best.argmin(dim=1)
            picked.append(index)
            hot = F.one_hot(index, z.size(1)).to(z.dtype)
            best = torch.maximum(best, torch.einsum("bne,be->bn", z, torch.einsum("bne,bn->be", z, hot)))
        index = torch.stack(picked, dim=1)                             # [B, K]
        return torch.gather(z, 1, index.unsqueeze(-1).expand(-1, -1, z.size(-1)))

    @torch.no_grad()
    def labels(self, theta=None, settle=0, features=None, signal=None):
        """Hard labels [B, N] for scoring. This replaces spectral clustering."""
        return self.forward(
            theta=theta, settle=settle, features=features, signal=signal
        ).argmax(dim=-1)
