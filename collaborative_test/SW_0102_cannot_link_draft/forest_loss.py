"""SW0102 maximin cannot-link draft, awaiting integration review."""

import statistics

import numpy as np
import torch
import torch.nn.functional as F


def maximin_forest_edge_map(q_detached, edge_threshold=0.40,
                            symmetry_tolerance=1e-6):
    """Build detached forest maps on CPU for q[B,N,N].

    Edge weights are s=max(q,q.T); off-diagonal edges strictly above the
    threshold enter Kruskal. Each forest merge labels every cross-component
    pair in both directions with the flat index of that merge edge. This is a
    widest-path bottleneck edge map, not a differentiable topology operation.
    """
    if q_detached.dim() != 3 or q_detached.size(1) != q_detached.size(2):
        raise ValueError("q must have shape [B,N,N]")
    q = q_detached.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(q).all():
        raise FloatingPointError("q contains nonfinite values")
    if not torch.allclose(q, q.transpose(1, 2), atol=symmetry_tolerance, rtol=0):
        raise ValueError("classifier q is not symmetric within tolerance")
    symmetric = np.maximum(q.numpy(), q.transpose(1, 2).numpy())
    batch, nodes, _ = symmetric.shape
    maps = np.full((batch, nodes, nodes), -1, dtype=np.int64)
    upper_i, upper_j = np.triu_indices(nodes, k=1)
    for b in range(batch):
        weights = symmetric[b]
        upper_w = weights[upper_i, upper_j]
        keep = upper_w > float(edge_threshold)
        edge_i, edge_j, edge_w = upper_i[keep], upper_j[keep], upper_w[keep]
        order = np.lexsort((edge_j, edge_i, -edge_w))
        parent = list(range(nodes))
        members = [[node] for node in range(nodes)]

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for edge_number in order:
            u, v = int(edge_i[edge_number]), int(edge_j[edge_number])
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            left, right = members[ru], members[rv]
            edge_index = u * nodes + v
            maps[b][np.ix_(left, right)] = edge_index
            maps[b][np.ix_(right, left)] = edge_index
            if len(left) < len(right):
                ru, rv = rv, ru
                left, right = right, left
            parent[rv] = ru
            members[ru] = left + right
            members[rv] = []
    return torch.from_numpy(maps)


def same_component_pairs(q_detached, pair_mask, threshold=0.50,
                         symmetry_tolerance=1e-6):
    """Boolean same-CC map at classifier affinity >= threshold (diagnostic)."""
    if q_detached.shape != pair_mask.shape:
        raise ValueError("pair mask must match q")
    if q_detached.dim() != 3 or q_detached.size(1) != q_detached.size(2):
        raise ValueError("q must have shape [B,N,N]")
    q = q_detached.detach().to(device="cpu", dtype=torch.float64)
    mask = pair_mask.detach().to(device="cpu", dtype=torch.bool)
    if not torch.isfinite(q).all():
        raise FloatingPointError("q contains nonfinite values")
    if not torch.allclose(q, q.transpose(1, 2), atol=symmetry_tolerance, rtol=0):
        raise ValueError("classifier q is not symmetric within tolerance")
    s = np.maximum(q.numpy(), q.transpose(1, 2).numpy())
    mask_np = mask.numpy()
    batch, nodes, _ = s.shape
    result = np.zeros((batch, nodes, nodes), dtype=np.bool_)
    upper_i, upper_j = np.triu_indices(nodes, k=1)
    for b in range(batch):
        parent = list(range(nodes))

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        upper_w = s[b, upper_i, upper_j]
        keep = upper_w >= float(threshold)
        edge_i, edge_j, edge_w = upper_i[keep], upper_j[keep], upper_w[keep]
        order = np.lexsort((edge_j, edge_i, -edge_w))
        for edge_number in order:
            i, j = int(edge_i[edge_number]), int(edge_j[edge_number])
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        root_ids = np.fromiter((find(node) for node in range(nodes)),
                               dtype=np.int64, count=nodes)
        result[b] = (root_ids[:, None] == root_ids[None, :]) & mask_np[b]
        np.fill_diagonal(result[b], False)
    return torch.from_numpy(result).to(device=q_detached.device)


def cannot_link_hinge(q, negative_masks_by_bin, edge_threshold=0.40,
                      hinge_margin=0.40, symmetry_tolerance=1e-6):
    """Balanced live-edge hinge over directed SW0101 negatives.

    `negative_masks_by_bin` is an ordered sequence of bool [B,N,N] masks. The
    caller must construct masks with the exact SW0101 bin and eligible-anchor
    rule. Positives only affect that eligibility; this function never uses
    positive pairs as supervision. Per-anchor negatives include disconnected
    (zero-loss) decisions, then aggregate anchors, nonempty bins/image, and
    nonempty images/batch as fixed by the provisional final specification.
    """
    if not negative_masks_by_bin:
        raise ValueError("at least one spatial-bin mask is required")
    if not 0.0 <= float(hinge_margin) <= float(edge_threshold):
        raise ValueError("hinge margin must be between zero and edge threshold")
    batch, nodes, nodes2 = q.shape
    if nodes != nodes2:
        raise ValueError("q must be square by image")
    s = torch.maximum(q, q.transpose(1, 2))
    maps = maximin_forest_edge_map(q.detach(), edge_threshold,
                                   symmetry_tolerance).to(q.device)
    bin_image_losses = [[] for _ in range(batch)]
    counts = {"selected_directed_negatives": 0,
              "connected_directed_negatives": 0,
              "eligible_anchors": 0,
              "nonempty_bins": 0,
              "nonempty_images": 0}
    for negative in negative_masks_by_bin:
        if negative.shape != q.shape or negative.dtype != torch.bool:
            raise ValueError("each negative bin mask must be bool [B,N,N]")
        for b in range(batch):
            mask = negative[b]
            anchor_counts = mask.sum(dim=-1)
            anchors = anchor_counts > 0
            if not anchors.any():
                continue
            edge_index = maps[b]
            connected = (edge_index >= 0) & mask
            counts["selected_directed_negatives"] += int(mask.sum())
            counts["connected_directed_negatives"] += int(connected.sum())
            flat_index = edge_index.clamp_min(0).reshape(-1)
            live_edges = s[b].reshape(-1).index_select(0, flat_index).reshape(nodes, nodes)
            c = torch.where(connected, live_edges,
                            torch.full_like(live_edges, float(hinge_margin)))
            pair_loss = F.relu(c - float(hinge_margin)).square() * mask.to(q.dtype)
            per_anchor = pair_loss.sum(dim=-1) / anchor_counts.clamp_min(1).to(q.dtype)
            bin_loss = per_anchor[anchors].mean()
            bin_image_losses[b].append(bin_loss)
            counts["eligible_anchors"] += int(anchors.sum())
            counts["nonempty_bins"] += 1
    image_losses = [torch.stack(items).mean() for items in bin_image_losses if items]
    counts["nonempty_images"] = len(image_losses)
    if not image_losses:
        # Differentiable exact zero for the fully empty batch.
        return s.sum() * 0.0, counts
    return torch.stack(image_losses).mean(), counts


def calibration_lambda(old_weighted_spike_grad_norms, cut_grad_norms,
                       multiplier=0.25):
    """Return immutable shared median lambda from valid calibration batches."""
    if len(old_weighted_spike_grad_norms) != len(cut_grad_norms):
        raise ValueError("gradient norm lists must have equal length")
    ratios = []
    for old_norm, cut_norm in zip(old_weighted_spike_grad_norms,
                                  cut_grad_norms):
        old_norm, cut_norm = float(old_norm), float(cut_norm)
        if (not torch.isfinite(torch.tensor(old_norm))
                or not torch.isfinite(torch.tensor(cut_norm))
                or old_norm <= 0 or cut_norm <= 0):
            raise ValueError("calibration requires finite positive gradient norms")
        ratios.append(float(multiplier) * old_norm / cut_norm)
    if not ratios:
        raise ValueError("at least one valid calibration batch is required")
    return float(statistics.median(ratios))
