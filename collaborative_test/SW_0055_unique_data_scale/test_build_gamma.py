import unittest

from build_gamma import training_ids


class TrainingIdsTest(unittest.TestCase):
    def test_segments_and_heldout_exclusion(self):
        ids = training_ids(2500)
        self.assertEqual(len(ids), 2500)
        self.assertEqual(ids[:2], [0, 1])
        self.assertEqual(ids[998:1002], [998, 999, 1640, 1641])
        self.assertEqual(ids[-1], 3139)
        self.assertTrue(set(ids).isdisjoint(range(1000, 1640)))

    def test_rejects_dropping_original_rows(self):
        with self.assertRaises(ValueError):
            training_ids(999)


if __name__ == "__main__":
    unittest.main()
