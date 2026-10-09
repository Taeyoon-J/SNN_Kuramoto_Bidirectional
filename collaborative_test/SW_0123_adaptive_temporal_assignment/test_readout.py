import numpy as np
import torch
import unittest

from readout import assignment_to_labels


def probabilities(labels):
    p = torch.zeros(1, 256, 11)
    p[0, torch.arange(256), torch.as_tensor(labels)] = 1
    return p


class FixedAssignmentReadoutTests(unittest.TestCase):
    def test_largest_slot_is_background_and_first_occupied_breaks_ties(self):
        labels = [2] * 20 + [4] * 20 + [1] * 216
        result = assignment_to_labels(probabilities(labels)).flatten()
        self.assertEqual(int((result == 0).sum()), 216)
        self.assertEqual(int((result == 3).sum()), 20)
        self.assertEqual(int((result == 5).sum()), 20)

        tied = [4] * 100 + [2] * 100 + [7] * 56
        tied_result = assignment_to_labels(probabilities(tied)).flatten()
        self.assertEqual(int((tied_result == 0).sum()), 100)
        self.assertEqual(int((tied_result == 3).sum()), 100)

    def test_foreground_slots_smaller_than_two_are_background(self):
        labels = [0] * 252 + [3] * 3 + [6]
        result = assignment_to_labels(probabilities(labels)).flatten()
        self.assertEqual(int((result == 0).sum()), 253)
        self.assertEqual(int((result == 4).sum()), 3)

    def test_rejects_nonfinite_and_wrong_patch_shape(self):
        with self.assertRaises(ValueError):
            assignment_to_labels(torch.rand(1, 255, 11))
        p = probabilities([0] * 256)
        p[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            assignment_to_labels(p)


if __name__ == "__main__":
    unittest.main()
