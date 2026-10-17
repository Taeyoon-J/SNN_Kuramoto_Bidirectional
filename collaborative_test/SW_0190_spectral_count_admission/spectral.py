"""Deterministic global partition of the unchanged spike-derived affinity."""
import numpy as np


def cluster(affinity, count):
    a = np.asarray(affinity, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all()
            or np.any(a < 0) or type(count) is not int or count < 1):
        raise ValueError('finite nonnegative square affinity and positive integer count required')
    if not np.allclose(a, a.T, atol=1e-6, rtol=0):
        raise ValueError('symmetric affinity required')
    n = len(a)
    if n == 0: return np.empty(0, dtype=np.int32)
    k = min(count, n)
    a = (a + a.T) * .5
    np.fill_diagonal(a, 0.)
    scale = 1. / np.sqrt(np.maximum(a.sum(1), 1e-12))
    normalized = a * scale[:, None] * scale[None, :]
    _, vectors = np.linalg.eigh(normalized)
    embedding = vectors[:, -k:]
    embedding = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    # Farthest-point initialization and 50 Lloyd steps, no selected restarts.
    first = int(np.argmax(np.sum((embedding - embedding.mean(0))**2, axis=1)))
    chosen = [first]
    distance = np.sum((embedding - embedding[first])**2, axis=1)
    for _ in range(1, k):
        distance[chosen] = -1.
        index = int(np.argmax(distance)); chosen.append(index)
        distance = np.minimum(distance, np.sum((embedding - embedding[index])**2, axis=1))
    centers = embedding[chosen].copy()
    previous = None
    for _ in range(50):
        distances = np.sum((embedding[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = distances.argmin(1)
        if previous is not None and np.array_equal(previous, labels): break
        previous = labels.copy()
        residual = distances[np.arange(n), labels]
        for index in range(k):
            selected = labels == index
            centers[index] = embedding[selected].mean(0) if selected.any() else embedding[int(residual.argmax())]
    return labels.astype(np.int32)


def background_from_model(clusters, baseline):
    labels = np.asarray(clusters)
    prior = np.asarray(baseline)
    if labels.shape != prior.shape or labels.size == 0:
        raise ValueError('matching nonempty model partitions required')
    candidates = np.unique(labels)
    # No GT masks here. Max overlap with the model's existing background,
    # then largest cluster, then smallest cluster ID.
    bg = max(candidates.tolist(), key=lambda c: (int(np.sum((labels == c) & (prior == 0))),
                                                 int(np.sum(labels == c)), -int(c)))
    out = np.zeros(labels.shape, dtype=np.int32)
    for index, value in enumerate(c for c in candidates if c != bg):
        out[labels == value] = index + 1
    return out


def transform(affinity, baseline, oracle_count):
    baseline = np.asarray(baseline)
    if baseline.ndim != 2 or baseline.dtype.kind not in 'iu' or np.any(baseline < 0):
        raise ValueError('nonnegative native patch mask required')
    q = np.asarray(affinity)
    if q.shape != (baseline.size, baseline.size):
        raise ValueError('patch mask and affinity geometry differ')
    support = baseline.ravel() > 0
    retained = np.zeros(baseline.size, dtype=np.int32)
    if support.any():
        retained[support] = cluster(q[np.ix_(support, support)], oracle_count) + 1
    all_labels = cluster(q, oracle_count + 1).reshape(baseline.shape)
    return {'spectral_existing_support': retained.reshape(baseline.shape),
            'spectral_all_patch': background_from_model(all_labels, baseline)}
