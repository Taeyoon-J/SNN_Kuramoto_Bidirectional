from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0150_spike_spatial_mixture.seed_selection import (
    NODES, select_seed_ids,
)


def qcc(rows):
    labels = torch.zeros((1, 32, 32), dtype=torch.int64)
    flat = labels.reshape(-1)
    for patch, label in rows.items():
        flat[patch] = label
    return labels


def symmetric_q():
    return torch.zeros((1, NODES, NODES), dtype=torch.float32)


def set_edge(q, left, right, value):
    q[0, left, right] = value
    q[0, right, left] = value


class SeedSelectionTests(unittest.TestCase):
    def test_background_zero_uses_q_medoid_then_farthest_first_and_next(self):
        labels = qcc({0: 0, 1: 0, 2: 0, 3: 5, 4: 7})
        q = symmetric_q()
        set_edge(q, 1, 0, 0.4)
        set_edge(q, 1, 2, 0.8)
        set_edge(q, 1, 3, 0.1)  # first foreground seed is farthest from medoid 1
        set_edge(q, 1, 4, 0.9)
        set_edge(q, 3, 4, 0.2)
        self.assertEqual(select_seed_ids(q, labels), [[1, 3, 4]])

    def test_missing_zero_falls_back_to_largest_region_and_ties_by_first_patch(self):
        labels = torch.empty((1, 32, 32), dtype=torch.int64)
        flat = labels.reshape(-1)
        flat[:512] = 9
        flat[512:] = 4
        # Regions 4 and 9 tie in size; label 9 starts at the smaller patch ID.
        self.assertEqual(select_seed_ids(symmetric_q(), labels), [[0, 512]])

    def test_seed_count_tracks_distinct_foreground_regions_and_caps_at_ten(self):
        q = symmetric_q()
        few = qcc({0: 0, 1: 0, 2: 0, 3: 8, 4: 8, 5: 8, 6: 8})
        # One predicted foreground region yields one foreground seed, not ten.
        self.assertEqual(select_seed_ids(q, few), [[0, 3]])

        many = torch.zeros((1, 32, 32), dtype=torch.int64)
        flat = many.reshape(-1)
        flat[:12] = torch.arange(1, 13)
        flat[12:] = 0
        selected = select_seed_ids(q, many)[0]
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected[0], 12)
        self.assertEqual(selected[1:], list(range(10)))

    def test_background_patch_exclusion_and_one_region_k1(self):
        labels = qcc({0: 0, 1: 0, 2: 3, 3: 3})
        self.assertEqual(select_seed_ids(symmetric_q(), labels), [[0, 2]])
        all_background = torch.zeros((1, 32, 32), dtype=torch.int64)
        self.assertEqual(select_seed_ids(symmetric_q(), all_background), [[0]])

    def test_selection_is_deterministic_under_exact_farthest_point_ties(self):
        labels = qcc({0: 0, 1: 0, 9: 2, 10: 3, 11: 4})
        q = symmetric_q()
        expected = [[0, 9, 10, 11]]
        self.assertEqual(select_seed_ids(q, labels), expected)
        self.assertEqual(select_seed_ids(q, labels), expected)

    def test_rejects_invalid_q_label_shape_or_label_values(self):
        q = symmetric_q()
        labels = qcc({0: 0, 1: 1})
        with self.assertRaisesRegex(ValueError, "symmetric"):
            bad = q.clone()
            bad[0, 0, 1] = 0.3
            select_seed_ids(bad, labels)
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = q.clone()
            bad[0, 0, 1] = float("nan")
            bad[0, 1, 0] = float("nan")
            select_seed_ids(bad, labels)
        with self.assertRaisesRegex(ValueError, "shape"):
            select_seed_ids(q, torch.zeros((1, 31, 32), dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            bad_labels = labels.clone()
            bad_labels.reshape(-1)[3] = -1
            select_seed_ids(q, bad_labels)
        with self.assertRaises(TypeError):
            select_seed_ids(q, labels.float())


if __name__ == "__main__":
    unittest.main()
