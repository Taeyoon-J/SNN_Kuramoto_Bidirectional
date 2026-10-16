"""Pure CPU tests for SW0179 paired-image, three-seed scoring math."""
import unittest

import numpy as np

from collaborative_test.SW_0179_qce_seed_replication import score


class SW0179ScoreMathContracts(unittest.TestCase):
    def test_candidate_seed_mean_precedes_bootstrap_over_shared_images(self):
        vectors = {
            0: np.full(score.COUNT, 0.1),
            1: np.full(score.COUNT, 0.2),
            2: np.full(score.COUNT, 0.6),
        }
        averaged = score._average_seed_metric_vectors(vectors)
        self.assertTrue(np.array_equal(averaged, np.full(score.COUNT, 0.3)))
        baseline = np.full(score.COUNT, 0.1)
        rng = np.random.RandomState(score.BOOTSTRAP_SEED)
        indices = rng.randint(0, score.COUNT,
            size=(score.BOOTSTRAP_REPLICATES, score.COUNT))
        actual = score._bootstrap_rows(averaged, baseline, indices)
        delta = averaged - baseline
        expected = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
        self.assertEqual(actual["mean"], float(delta.mean()))
        self.assertEqual(actual["ci95"], [float(expected[0]), float(expected[1])])
        self.assertEqual(actual["resampling_unit"], "paired_image_after_three_seed_average")

    def test_primary_gate_requires_both_fg_baselines_and_both_confidence_bounds(self):
        def group(fg, fiou, oiou):
            return {"fg_ari": {"mean": fg}, "foreground_iou": {"mean": fiou},
                    "matched_object_iou": {"mean": oiou}}
        scores = {
            "candidate_standalone": group(.82, .25, .24),
            "SW0165_foreground_induced_standalone": group(score.SW165_FG_BASELINE, .6, .65),
            "Slot70k_standalone": group(score.SLOT_FG_BASELINE,
                score.SLOT_FOREGROUND_IOU_BASELINE, score.SLOT_MATCHED_IOU_BASELINE),
        }
        positive = {"ci95": [0.001, 0.02]}
        paired = {"standalone_candidate_minus_SW0165_fg_ari": positive,
                  "standalone_candidate_minus_Slot70k_fg_ari": positive}
        gate = score._gates(scores, paired)
        self.assertTrue(gate["primary_gate_passed"])
        self.assertFalse(gate["fg_ari_at_least_0_87_goal"])
        paired["standalone_candidate_minus_Slot70k_fg_ari"] = {"ci95": [-0.001, 0.02]}
        self.assertFalse(score._gates(scores, paired)["primary_gate_passed"])

    def test_invalid_seed_vector_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "three finite metric vectors"):
            score._average_seed_metric_vectors({0: np.zeros(score.COUNT), 1: np.zeros(score.COUNT)})
        vectors = {seed: np.zeros(score.COUNT) for seed in score.SEEDS}
        vectors[2][0] = np.nan
        with self.assertRaisesRegex(ValueError, "three finite metric vectors"):
            score._average_seed_metric_vectors(vectors)


if __name__ == "__main__":
    unittest.main()
