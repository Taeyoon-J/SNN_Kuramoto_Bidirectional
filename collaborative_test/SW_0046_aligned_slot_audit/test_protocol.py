"""Pure NumPy tests for Slot Attention mask protocol helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import perimeter_background, remap_foreground, validate_slice


class ProtocolTest(unittest.TestCase):
    def test_slice_validation_returns_half_open_bounds(self):
        self.assertEqual(validate_slice(1320, 320, 1640), (1320, 1640))

    def test_slice_validation_rejects_out_of_bounds_and_empty(self):
        for args in ((-1, 1, 10), (9, 2, 10), (0, 0, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_slice(*args)

    def test_perimeter_counts_corners_once_and_uses_smallest_id_for_ties(self):
        mask = np.array([
            [1, 1, 1, 0],
            [2, 3, 3, 0],
            [2, 3, 3, 0],
            [1, 1, 0, 0],
        ])
        # Perimeter sequence is top, bottom, left interior, right interior.
        # IDs 0 and 1 tie at five pixels; argmax chooses the lower ID.
        self.assertEqual(perimeter_background(mask, num_slots=4), 0)

    def test_foreground_remap_reserves_zero_for_background(self):
        slots = np.array([[0, 1], [2, 1]], dtype=np.int64)
        labels = remap_foreground(slots, background_slot=1, num_slots=3)
        np.testing.assert_array_equal(labels, [[1, 0], [3, 0]])

    def test_mask_helpers_reject_invalid_slot_ids(self):
        with self.assertRaises(ValueError):
            perimeter_background(np.array([[0, 1], [2, 4]]), num_slots=4)
        with self.assertRaises(ValueError):
            remap_foreground(np.array([[0, 1]]), background_slot=2, num_slots=2)


if __name__ == "__main__":
    unittest.main()
