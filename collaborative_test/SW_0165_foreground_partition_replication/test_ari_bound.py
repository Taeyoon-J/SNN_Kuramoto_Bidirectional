import itertools

import numpy as np

from collaborative_test.SW_0165_foreground_partition_replication.ari_bound import (
    adjusted_rand_from_labels,
    fixed_support_ari_upper_bound,
)


def _partitions(n):
    """Restricted-growth strings: every set partition exactly once."""
    if n == 0:
        yield ()
        return
    def visit(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for value in range(max(prefix) + 2):
            yield from visit(prefix + [value])
    yield from visit([0])


def test_fixed_support_bound_exhaustive_through_six_patches():
    for n in range(2, 7):
        for target in _partitions(n):
            target = np.asarray(target, dtype=np.int64) + 1
            for support_tuple in itertools.product((False, True), repeat=n):
                support = np.asarray(support_tuple, dtype=bool)
                bound = fixed_support_ari_upper_bound(target, support)
                covered = np.flatnonzero(support)
                for positive_partition in _partitions(covered.size):
                    prediction = np.zeros(n, dtype=np.int64)
                    if covered.size:
                        prediction[covered] = np.asarray(positive_partition) + 1
                    score = adjusted_rand_from_labels(target, prediction)
                    assert score <= bound + 1e-12


def test_formula_spotchecks_sklearn_and_degenerate_convention():
    from sklearn.metrics import adjusted_rand_score

    cases = [
        ([1, 1, 2, 2], [0, 1, 2, 2]),
        ([1, 1, 1, 1], [0, 0, 1, 1]),
        ([1, 2, 3, 4], [0, 1, 2, 3]),
    ]
    for target, prediction in cases:
        assert np.isclose(
            adjusted_rand_from_labels(target, prediction),
            adjusted_rand_score(target, prediction),
        )
    assert adjusted_rand_from_labels([1, 1], [2, 2]) == 1.0
    assert adjusted_rand_from_labels([1, 2], [3, 4]) == 1.0
    assert np.isnan(fixed_support_ari_upper_bound(np.array([0]), np.array([False])))
