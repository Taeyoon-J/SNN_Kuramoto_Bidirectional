"""Statistical and report-contract checks for the read-only SW0125 summary."""
import unittest
from pathlib import Path

import numpy as np

from collaborative_test.SW_0125_late_rollout_credit import summarize


def _score(fg):
    fg = np.asarray(fg, dtype=np.float64)
    return {"metrics": {"fg_ari": float(fg.mean()), "foreground_iou": 0.3,
                         "matched_object_iou": 0.4},
            "valid_count": {metric: 320 for metric in summarize.METRICS},
            "per_image": {"fg_ari": fg, "foreground_iou": np.full(320, 0.3),
                          "matched_object_iou": np.full(320, 0.4)}}


class SummaryTests(unittest.TestCase):
    def test_seed0_native_source_path_and_training_binding_are_distinct(self):
        checkpoint = Path("seed0_positive_frozen") / "core.pt"
        self.assertEqual(summarize._native_source_evaluation_path(checkpoint),
                         checkpoint.parent / "evaluation.json")

        historical_sha = "a" * 64
        endpoint_sha = "b" * 64
        selected_native, training_binding = summarize._seed0_source_hashes(
            {"arms": {"source": {"evaluation_sha256": historical_sha}}}, endpoint_sha)
        self.assertEqual(selected_native, historical_sha)
        self.assertEqual(training_binding, endpoint_sha)
        self.assertNotEqual(selected_native, training_binding)

    def test_bootstrap_resamples_same_validation_images_across_seeds(self):
        candidate, reference = {}, {}
        image = np.arange(320, dtype=np.float64) / 320.0
        for seed in summarize.SEEDS:
            candidate[seed] = _score(image + 0.1 + seed * 0.01)
            reference[seed] = _score(image * 0.25)
        actual = summarize.paired_bootstrap(candidate, reference, draws=73, seed=125)
        mean_delta_by_image = np.stack([
            candidate[s]["per_image"]["fg_ari"] - reference[s]["per_image"]["fg_ari"]
            for s in summarize.SEEDS]).mean(axis=0)
        indices = np.random.default_rng(125).integers(0, 320, size=(73, 320))
        expected = np.quantile(mean_delta_by_image[indices].mean(axis=1), [0.025, 0.975])
        np.testing.assert_array_equal(actual["ci95"], expected)
        self.assertEqual(actual["draws"], 73)
        self.assertEqual(actual["unit"],
                         "paired image; identical sampled indices applied to all three seeds")

    def test_metric_contract_rejects_nonfinite_or_wrong_count(self):
        payload = _score(np.full(320, 0.5))
        self.assertEqual(summarize._score_payload(payload, "fixture")["metrics"]["fg_ari"], 0.5)
        payload["per_image"]["fg_ari"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid fg_ari"):
            summarize._score_payload(payload, "fixture")
        payload = _score(np.full(320, 0.5))
        payload["valid_count"]["foreground_iou"] = 319
        with self.assertRaisesRegex(ValueError, "invalid foreground_iou"):
            summarize._score_payload(payload, "fixture")


if __name__ == "__main__":
    unittest.main()
