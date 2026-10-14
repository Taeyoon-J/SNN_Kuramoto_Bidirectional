"""Focused CPU checks for the SW0162 signed connected-union kernel."""
import unittest

import numpy as np

from collaborative_test.SW_0162_connected_signed_partition import connected_signed as cs


def q_for_energy(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-matrix))


class ConnectedSignedTests(unittest.TestCase):
    def test_signed_formula_uses_both_directed_source_edges_and_clamps(self):
        q = np.full((3, 3), .5, dtype=np.float64)
        q[0, 1] = q[1, 0] = .8
        a = np.zeros((3, 3), dtype=np.float64)
        a[0, 1], a[1, 0] = .2, .6
        got = cs.signed_pair_energy(q, a, 2.0)
        expected = (1 + 1024 * (.2 + .6) / 4) * np.log(.8 / .2)
        self.assertAlmostEqual(got[0, 1], expected, places=12)
        self.assertEqual(got[0, 0], 0.0)
        self.assertTrue(np.array_equal(got, got.T))
        edge = cs.signed_pair_energy(np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2)), 1)
        self.assertTrue(np.isfinite(edge).all())

    def test_q_is_symmetrized_before_endpoint_clamping(self):
        q = np.full((3, 3), .4, dtype=np.float64)
        q[0, 1], q[1, 0] = .5e-6, 1.5e-6
        a = np.zeros_like(q)
        got = cs.signed_pair_energy(q, a, 1.0)
        q_sym = (q + q.T) / 2
        q_bar = np.clip(q_sym, 1e-6, 1 - 1e-6)
        expected = np.log(q_bar / (1 - q_bar))
        np.fill_diagonal(expected, 0)
        np.testing.assert_array_equal(got, expected)

    def test_merge_gain_sums_positive_and_negative_cross_evidence(self):
        # The strongest initial edge merges 0--1. Although 1--2 alone is
        # attractive, the merged union's complete cross-energy is 3 + (-10).
        e = np.zeros((3, 3), dtype=np.float64)
        e[0, 1] = e[1, 0] = 10.0
        e[1, 2] = e[2, 1] = 3.0
        e[0, 2] = e[2, 0] = -10.0
        result = cs.connected_signed_merge(q_for_energy(e), np.zeros_like(e), 1.0,
                                           grid_shape=(1, 3))
        self.assertEqual(result["merge_count"], 1)
        self.assertEqual(result["labels"].tolist(), [[0, 0, 2]])
        self.assertLess(result["pair_energy"][0, 2] + result["pair_energy"][1, 2], 0)

    def test_non_neighbor_positive_edge_cannot_merge_and_ties_are_repeatable(self):
        e = np.full((4, 4), -4.0, dtype=np.float64)
        np.fill_diagonal(e, 0)
        e[0, 3] = e[3, 0] = 12.0  # diagonal on a 2x2 grid, not four-neighbour adjacent
        q = q_for_energy(e)
        no_merge = cs.connected_signed_merge(q, np.zeros_like(e), 1.0, grid_shape=(2, 2))
        self.assertEqual(no_merge["merge_count"], 0)

        tied = np.zeros((4, 4), dtype=np.float64)
        for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
            tied[i, j] = tied[j, i] = 2.0
        first = cs.connected_signed_merge(q_for_energy(tied), np.zeros_like(tied), 1.0,
                                          grid_shape=(2, 2))
        second = cs.connected_signed_merge(q_for_energy(tied), np.zeros_like(tied), 1.0,
                                           grid_shape=(2, 2))
        np.testing.assert_array_equal(first["labels"], second["labels"])
        self.assertEqual(first["clusters"], second["clusters"])
        self.assertEqual(first["clusters"][0]["representative"], 0)

    def test_background_ties_use_first_patch_and_small_groups_fold_into_background(self):
        labels = np.array([[7, 7, 9, 9], [7, 7, 9, 9],
                           [7, 7, 9, 9], [4, 4, 4, 4]], dtype=np.int64)
        got = cs.canonical_labels(labels, min_size=5)
        # Labels 7 and 9 tie as largest; 7 owns the earliest patch. Label 4
        # is below min_size and folds into BG; label 9 becomes FG ID 1.
        self.assertEqual(got[0, 0], 0)
        self.assertEqual(got[0, 2], 1)
        self.assertTrue(np.all(got[3, :] == 0))

    def test_invalid_source_contract_rejects_nonfinite_asymmetry_and_gain(self):
        q = np.full((4, 4), .5, dtype=np.float64)
        a = np.zeros_like(q)
        for bad_q, bad_a, gain in ((np.full((4, 4), np.nan), a, 1),
                                   (q + np.triu(np.ones_like(q), 1) * .1, a, 1),
                                   (q, a, 0)):
            with self.assertRaises(ValueError):
                cs.connected_signed_merge(bad_q, bad_a, gain, grid_shape=(2, 2))
        with self.assertRaises(ValueError):
            cs.connected_signed_merge(q, a, 1, grid_shape=(1, 3))


if __name__ == "__main__":
    unittest.main()
