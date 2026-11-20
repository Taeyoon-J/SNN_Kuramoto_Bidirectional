"""Run the peer's own test for the merge against our copy, plus checks a port needs.

The first test is theirs verbatim from patch_final tests/test_sw240.py. If our copy does
not pass the test written for the original, it is not the same function.
"""
import sys
sys.path.insert(0, "/Users/taeyoonkim/Desktop/SNN_Kuramoto_Bidirectional/collaborative_test/ported")
import numpy as np
from partition import foreground_induced_merge
from signed_energy import signed_pair_energy

def test_peer_verbatim():
    support = np.array([[1, 1], [0, 2]], np.int64)
    q = np.full((4, 4), .1); np.fill_diagonal(q, 1.); q[0, 1] = q[1, 0] = .9
    result = foreground_induced_merge(q, np.eye(4), 8., support, grid_shape=(2, 2))['labels']
    assert np.array_equal(result > 0, support > 0)
    assert result[0, 0] == result[0, 1] and result[1, 1] != result[0, 1]
    return "peer's own test passes"

def test_foreground_support_preserved():
    rng = np.random.default_rng(0)
    g = 8; n = g * g
    sup = (rng.random((g, g)) < .3).astype(np.int64)
    q = rng.random((n, n)); q = (q + q.T) / 2; np.fill_diagonal(q, 1.0)
    a = np.zeros((n, n))
    out = foreground_induced_merge(q, a, 8., sup, grid_shape=(g, g))['labels']
    assert np.array_equal(out > 0, sup > 0), "foreground support must not change"
    return "foreground support preserved on a random 8x8"

def test_never_merges_across_a_gap():
    # two foreground blobs with no four-connected path between them must stay separate
    sup = np.array([[1, 0, 1], [1, 0, 1], [0, 0, 0]], np.int64)
    n = 9
    q = np.full((n, n), 0.99); np.fill_diagonal(q, 1.0)
    out = foreground_induced_merge(q, np.zeros((n, n)), 8., sup, grid_shape=(3, 3))['labels']
    left = {out[0, 0], out[1, 0]}; right = {out[0, 2], out[1, 2]}
    assert len(left) == 1 and len(right) == 1, "each blob should be one group"
    assert left.isdisjoint(right), "disconnected blobs merged across a gap"
    return "four-connectivity respected even at Q=0.99 everywhere"

def test_low_q_does_not_merge():
    sup = np.array([[1, 1]], np.int64)
    q = np.full((2, 2), 0.2); np.fill_diagonal(q, 1.0)
    out = foreground_induced_merge(q, np.zeros((2, 2)), 8., sup, grid_shape=(1, 2))['labels']
    assert out[0, 0] != out[0, 1], "Q below 0.5 gives negative logit and must not merge"
    return "Q < 0.5 leaves the pair split"

def test_native_nodes_changes_magnitude_not_sign():
    rng = np.random.default_rng(1)
    n = 16
    q = rng.random((n, n)); q = (q + q.T) / 2; np.fill_diagonal(q, 1.0)
    a = rng.random((n, n)) * 0.01
    e1 = signed_pair_energy(q, a, 8.0, native_nodes=1024)
    e2 = signed_pair_energy(q, a, 8.0, native_nodes=4096)
    assert np.array_equal(np.sign(e1), np.sign(e2)), "the node count must not flip any sign"
    assert not np.allclose(e1, e2), "it should still change the magnitudes"
    return "native_nodes scales magnitude, never flips a merge decision"

ok = 0
for fn in (test_peer_verbatim, test_foreground_support_preserved, test_never_merges_across_a_gap,
           test_low_q_does_not_merge, test_native_nodes_changes_magnitude_not_sign):
    try:
        print("  PASS  %s" % fn()); ok += 1
    except AssertionError as e:
        print("  FAIL  %s: %s" % (fn.__name__, e))
    except Exception as e:
        print("  ERROR %s: %s: %s" % (fn.__name__, type(e).__name__, e))
print("%d/5" % ok)
