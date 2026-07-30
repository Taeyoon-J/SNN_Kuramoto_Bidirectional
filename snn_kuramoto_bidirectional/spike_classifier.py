"""Differentiable clustering of oscillator membrane-potential histories."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SoftMembraneClassifier(nn.Module):
    """Cluster complete membrane histories into soft oscillator groups.

    A shared MLP embeds every oscillator's complete temporal history. Cluster
    centers are initialized from oscillator embeddings in the current input
    and iteratively updated using differentiable soft assignments.

    Args:
        history_length:
            Number of membrane-history time steps ``T``.
        embedding_dim:
            Dimension ``D`` of each encoded oscillator history.
        num_iterations:
            Number of differentiable center-update iterations.
        temperature:
            Softmax temperature for center membership.
        eps:
            Numerical stability value used during normalization.
    """

    def __init__(
        self,
        history_length: int,
        embedding_dim: int,
        num_iterations: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.embedding_dim = int(embedding_dim)
        self.num_iterations = int(num_iterations)
        self.temperature = float(temperature)
        self.eps = float(eps)

        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.oscillator_encoder = nn.Sequential(
            nn.Linear(self.history_length, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(
        self,
        membrane_history: Tensor,
        num_centers: int,
    ) -> Tensor:
        """Return soft memberships shaped ``[B, num_centers, N]``.

        Args:
            membrane_history:
                Complete oscillator histories shaped ``[B, N, T]``.
            num_centers:
                Number ``K`` of input-conditioned clustering centers.

        Returns:
            Soft oscillator memberships shaped ``[B, K, N]``. Memberships
            sum to one across the center dimension for every oscillator.
        """

        oscillator_embeddings = self.oscillator_encoder(membrane_history)
        centers = self._initialize_centers(
            oscillator_embeddings,
            num_centers=int(num_centers),
        )

        for _ in range(self.num_iterations):
            membership = self._calculate_membership(
                oscillator_embeddings,
                centers,
            )
            center_mass = membership.sum(dim=2, keepdim=True)
            centers = (
                membership @ oscillator_embeddings
            ) / center_mass.clamp_min(self.eps)

        return self._calculate_membership(
            oscillator_embeddings,
            centers,
        )

    def _initialize_centers(
        self,
        oscillator_embeddings: Tensor,
        num_centers: int,
    ) -> Tensor:
        batch_size, num_oscillators, embedding_dim = (
            oscillator_embeddings.shape
        )
        random_scores = torch.rand(
            batch_size,
            num_oscillators,
            device=oscillator_embeddings.device,
        )
        center_indices = random_scores.argsort(dim=1)[:, :num_centers]
        gather_indices = center_indices.unsqueeze(-1).expand(
            -1,
            -1,
            embedding_dim,
        )
        return torch.gather(
            oscillator_embeddings,
            dim=1,
            index=gather_indices,
        )

    def _calculate_membership(
        self,
        oscillator_embeddings: Tensor,
        centers: Tensor,
    ) -> Tensor:
        normalized_embeddings = F.normalize(
            oscillator_embeddings,
            p=2,
            dim=2,
            eps=self.eps,
        )
        normalized_centers = F.normalize(
            centers,
            p=2,
            dim=2,
            eps=self.eps,
        )
        similarity_logits = (
            normalized_centers
            @ normalized_embeddings.transpose(1, 2)
        )
        return torch.softmax(
            similarity_logits / self.temperature,
            dim=1,
        )
