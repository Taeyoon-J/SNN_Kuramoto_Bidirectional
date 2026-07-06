import itertools

import torch
import torch.nn.functional as F


def brain_similarity(gamma_t, gamma_next, eps=1e-8):
    """
    BrainSim(gamma_t, gamma_next): cosine similarity between adjacent gammas.

    Args:
        gamma_t:
            Tensor shaped [..., D].
        gamma_next:
            Tensor shaped [..., D].

    Returns:
        Tensor shaped [...].
    """
    return F.cosine_similarity(gamma_t, gamma_next, dim=-1, eps=eps)


def gamma_ordering_loss(ordered_gammas, lambda_smooth=1.0, mu_similarity=1.0):
    """
    Compute:
        L(pi) = lambda * sum_t ||gamma_{t+2} - 2 gamma_{t+1} + gamma_t||^2
                - mu * sum_t BrainSim(gamma_t, gamma_{t+1})

    Args:
        ordered_gammas:
            Tensor shaped [B, T, D].

    Returns:
        Tensor shaped [B].
    """
    if ordered_gammas.dim() != 3:
        raise ValueError("ordered_gammas must have shape [B, T, D]. Use B=1 for one sample.")

    num_steps = ordered_gammas.size(1)
    device = ordered_gammas.device
    batch_size = ordered_gammas.size(0)
    loss = torch.zeros(batch_size, device=device, dtype=ordered_gammas.dtype)

    if num_steps >= 3:
        second_diff = ordered_gammas[:, 2:] - 2 * ordered_gammas[:, 1:-1] + ordered_gammas[:, :-2]
        loss = loss + float(lambda_smooth) * second_diff.pow(2).sum(dim=(1, 2))

    if num_steps >= 2:
        adjacent_sim = brain_similarity(ordered_gammas[:, :-1], ordered_gammas[:, 1:])
        loss = loss - float(mu_similarity) * adjacent_sim.sum(dim=1)

    return loss


@torch.no_grad()
def order_gammas(
    gammas,
    lambda_smooth=1.0,
    mu_similarity=1.0,
    method="auto",
    exact_max_steps=8,
    local_search_passes=5,
):
    """
    Reorder gamma vectors to minimize gamma_ordering_loss.

    Args:
        gammas:
            Tensor shaped [B, T, D].
        method:
            "auto", "exact", or "local_search".
            "exact" checks every permutation and is only practical for small T.
            "local_search" starts from a greedy cosine-similarity path, then
            improves it by pairwise swaps.

    Returns:
        ordered_gammas, order_indices, loss

        ordered_gammas: [B, T, D]
        order_indices: [B, T]
        loss: [B]
    """
    if gammas.dim() != 3:
        raise ValueError("gammas must have shape [B, T, D]. Use B=1 for one sample.")

    ordered = []
    indices = []
    losses = []
    for batch_gammas in gammas:
        ordered_gammas, order_indices, loss = _order_single(
            batch_gammas,
            lambda_smooth=lambda_smooth,
            mu_similarity=mu_similarity,
            method=method,
            exact_max_steps=exact_max_steps,
            local_search_passes=local_search_passes,
        )
        ordered.append(ordered_gammas)
        indices.append(order_indices)
        losses.append(loss)
    return torch.stack(ordered), torch.stack(indices), torch.stack(losses)


def _order_single(
    gammas,
    lambda_smooth,
    mu_similarity,
    method,
    exact_max_steps,
    local_search_passes,
):
    if method not in {"auto", "exact", "local_search"}:
        raise ValueError('method must be one of "auto", "exact", or "local_search".')

    num_steps = gammas.size(0)
    if num_steps <= 1:
        indices = torch.arange(num_steps, device=gammas.device)
        loss = gamma_ordering_loss(gammas.unsqueeze(0), lambda_smooth, mu_similarity).squeeze(0)
        return gammas, indices, loss

    if method == "auto":
        method = "exact" if num_steps <= int(exact_max_steps) else "local_search"

    if method == "exact":
        if num_steps > int(exact_max_steps):
            raise ValueError(
                f"exact ordering for T={num_steps} is too expensive. "
                "Use method='local_search' or increase exact_max_steps explicitly."
            )
        return _order_exact(gammas, lambda_smooth, mu_similarity)

    return _order_local_search(
        gammas,
        lambda_smooth=lambda_smooth,
        mu_similarity=mu_similarity,
        passes=local_search_passes,
    )


def _order_exact(gammas, lambda_smooth, mu_similarity):
    best_indices = None
    best_loss = None

    for permutation in itertools.permutations(range(gammas.size(0))):
        indices = torch.tensor(permutation, device=gammas.device)
        ordered = gammas.index_select(0, indices)
        loss = gamma_ordering_loss(ordered.unsqueeze(0), lambda_smooth, mu_similarity).squeeze(0)
        if best_loss is None or loss.item() < best_loss.item():
            best_indices = indices
            best_loss = loss

    return gammas.index_select(0, best_indices), best_indices, best_loss


def _order_local_search(gammas, lambda_smooth, mu_similarity, passes):
    best_indices = _greedy_similarity_order(gammas)
    best_loss = gamma_ordering_loss(
        gammas.index_select(0, best_indices).unsqueeze(0),
        lambda_smooth,
        mu_similarity,
    ).squeeze(0)

    num_steps = gammas.size(0)
    for _ in range(int(passes)):
        improved = False
        for i in range(num_steps - 1):
            for j in range(i + 1, num_steps):
                candidate_indices = best_indices.clone()
                candidate_indices[i], candidate_indices[j] = (
                    candidate_indices[j].clone(),
                    candidate_indices[i].clone(),
                )
                candidate_loss = gamma_ordering_loss(
                    gammas.index_select(0, candidate_indices).unsqueeze(0),
                    lambda_smooth,
                    mu_similarity,
                ).squeeze(0)
                if candidate_loss.item() < best_loss.item():
                    best_indices = candidate_indices
                    best_loss = candidate_loss
                    improved = True
        if not improved:
            break

    return gammas.index_select(0, best_indices), best_indices, best_loss


def _greedy_similarity_order(gammas):
    num_steps = gammas.size(0)
    similarity = F.cosine_similarity(
        gammas.unsqueeze(1),
        gammas.unsqueeze(0),
        dim=-1,
    )
    similarity.fill_diagonal_(-float("inf"))

    row_scores = similarity.sum(dim=1)
    current = int(torch.argmax(row_scores).item())
    order = [current]
    remaining = set(range(num_steps))
    remaining.remove(current)

    while remaining:
        candidates = torch.tensor(sorted(remaining), device=gammas.device)
        candidate_scores = similarity[current].index_select(0, candidates)
        next_idx = int(candidates[torch.argmax(candidate_scores)].item())
        order.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    return torch.tensor(order, device=gammas.device)
