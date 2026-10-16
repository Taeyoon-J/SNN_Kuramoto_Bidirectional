"""CPU checks for SW0181 paired fixed320 aggregation and registered gates."""
from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0181_actual_descent_qce import score


class ScoreMathTests(unittest.TestCase):
    def test_score_cli_help_is_wired(self):
        with self.assertRaises(SystemExit) as help_exit:
            score.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)

    def test_frozen_prediction_check_binds_both_archive_and_protocol(self):
        frozen = {"prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64}
        score._assert_prediction_unchanged(1, dict(frozen), frozen)
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            score._assert_prediction_unchanged(1,
                {"prediction_sha256": "a" * 64, "protocol_sha256": "c" * 64}, frozen)

    def test_seed_average_is_paired_by_image_and_requires_all_three_seeds(self):
        vectors = {0: np.arange(score.COUNT, dtype=np.float64),
            1: np.arange(score.COUNT, dtype=np.float64) + 2,
            2: np.arange(score.COUNT, dtype=np.float64) + 4}
        result = score._average_seed(vectors)
        np.testing.assert_array_equal(result, np.arange(score.COUNT, dtype=np.float64) + 2)
        with self.assertRaises(ValueError): score._average_seed({0: vectors[0], 1: vectors[1]})

    def test_bootstrap_is_fixed_rng_and_resamples_shared_images(self):
        candidate = np.linspace(0.2, 0.8, score.COUNT)
        baseline = candidate - 0.1
        indices = np.random.RandomState(score.BOOTSTRAP_SEED).randint(
            0, score.COUNT, size=(score.BOOTSTRAP_REPLICATES, score.COUNT))
        left = score._bootstrap(candidate, baseline, indices)
        right = score._bootstrap(candidate, baseline, indices)
        self.assertEqual(left, right)
        self.assertEqual(left["resampling_unit"], "paired_image_after_three_seed_average")
        self.assertEqual(left["seed"], 178)

    def test_primary_gate_requires_ci_and_both_slot_iou_gates(self):
        scores = {"candidate_standalone": {"fg_ari": {"mean": 0.9},
            "foreground_iou": {"mean": 0.4}, "matched_object_iou": {"mean": 0.4}},
            "SW0165_foreground_induced_standalone": {"fg_ari": {"mean": score.SW165_FG_BASELINE}},
            "Slot70k_standalone": {"fg_ari": {"mean": score.SLOT_FG_BASELINE},
                "foreground_iou": {"mean": score.SLOT_FOREGROUND_IOU_BASELINE},
                "matched_object_iou": {"mean": score.SLOT_MATCHED_IOU_BASELINE}}}
        paired = {"standalone_candidate_minus_SW0165_fg_ari": {"ci95": [0.01, 0.2]},
            "standalone_candidate_minus_Slot70k_fg_ari": {"ci95": [0.02, 0.3]}}
        self.assertTrue(score._gate(scores, paired)["primary_gate_passed"])
        scores["candidate_standalone"]["matched_object_iou"]["mean"] = 0.1
        self.assertFalse(score._gate(scores, paired)["primary_gate_passed"])


if __name__ == "__main__": unittest.main()
