"""Ground-truth-free spectral readouts constrained by a frozen spike CC mask."""
import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from snn_kuramoto_bidirectional.training.evaluate_binding import kmeans


def gamma_local_rows(row_count, global_start, count):
    """Resolve either an aligned count-row tensor or a global-row tensor."""
    if count < 1 or global_start < 0:
        raise ValueError("global_start must be non-negative and count positive")
    if row_count == count:
        return list(range(count))
    if row_count >= global_start + count:
        return list(range(global_start, global_start + count))
    # An aligned slice starts at row 0 even when preflighting only its prefix.
    if global_start > 0 and row_count >= count:
        return list(range(count))
    raise ValueError(
        f"Gamma has {row_count} rows; expected exactly {count} aligned rows or "
        f"at least {global_start + count} full-grid rows"
    )


def generic_spectral_partition(affinity, cluster_count):
    """Return a positive 1D spectral partition for an arbitrary node count.

    Unlike the historical 16x16 readout, this routine does not designate one
    cluster as background: every supplied node is already inside the frozen
    foreground mask.
    """
    affinity = torch.as_tensor(affinity, dtype=torch.float32)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    n = affinity.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    k = min(max(1, int(cluster_count)), n)
    affinity = affinity + 1e-6 * torch.eye(n, dtype=affinity.dtype)
    inv_degree = affinity.sum(dim=1).clamp_min(1e-8).rsqrt()
    normalized = inv_degree[:, None] * affinity * inv_degree[None, :]
    try:
        _, eigenvectors = torch.linalg.eigh(normalized)
    except RuntimeError:
        _, eigenvectors = torch.linalg.eigh(normalized.double())
        eigenvectors = eigenvectors.float()
    labels = kmeans(eigenvectors[:, -k:], k).to(torch.int64).cpu().numpy() + 1
    return labels


def compact_labels(labels):
    """Renumber positive labels in first-seen order; preserve zero background."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int64)
    for new_id, old_id in enumerate((x for x in np.unique(labels) if x != 0), 1):
        out[labels == old_id] = new_id
    return out


def preserve_foreground(labels, foreground):
    """Mask a spectral partition by a frozen foreground mask, then compact IDs."""
    labels = np.asarray(labels)
    foreground = np.asarray(foreground, dtype=bool)
    if labels.shape != foreground.shape:
        raise ValueError("labels and foreground must have identical shapes")
    result = np.where(foreground, labels, 0)
    # A positive spectral label is guaranteed on foreground. Fail rather than
    # silently changing the frozen mask if a caller passes malformed labels.
    if np.any(foreground & (result == 0)):
        raise ValueError("spectral labels must be positive on foreground nodes")
    return compact_labels(result)


def restricted_spectral_labels(affinity, foreground, cluster_count, spectral_labels_fn):
    """Cluster only foreground nodes, returning a full-grid label vector."""
    affinity = np.asarray(affinity)
    foreground = np.asarray(foreground, dtype=bool)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    if foreground.shape != (affinity.shape[0],):
        raise ValueError("foreground must match affinity node count")
    indices = np.flatnonzero(foreground)
    if not len(indices):
        return np.zeros(foreground.shape, dtype=np.int64)
    k = min(max(2, int(cluster_count)), len(indices))
    labels = np.zeros(foreground.shape, dtype=np.int64)
    if len(indices) == 1:
        labels[indices[0]] = 1
        return labels
    sub = affinity[np.ix_(indices, indices)]
    labels[indices] = compact_labels(spectral_labels_fn(sub, k))
    if np.any(labels[indices] == 0):
        raise ValueError("restricted spectral readout left foreground unassigned")
    return labels
