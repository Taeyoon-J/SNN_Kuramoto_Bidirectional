import unittest
from unittest import mock

import numpy as np

from collaborative_test.SW_0122_joint_rgb_seed_replication import summarize


class SW0122SummaryTests(unittest.TestCase):
    def test_bootstrap_averages_paired_seed_differences_by_shared_image_index(self):
        self.assertEqual(summarize.BOOTSTRAP_SEED, 122)
        candidate, control = {}, {}
        for seed in (0, 1, 2):
            values = np.arange(320, dtype=np.float64) / 320.0
            candidate[seed] = {"per_image": {"fg_ari": values + (seed + 1) / 100.0}}
            control[seed] = {"per_image": {"fg_ari": values}}
        result = summarize._bootstrap_pair(candidate, control)
        self.assertAlmostEqual(result["mean_delta"], 0.02, places=14)
        rng = np.random.default_rng(summarize.BOOTSTRAP_SEED)
        sampled = rng.integers(0, 320, size=(summarize.DRAWS, 320))
        expected = np.full(summarize.DRAWS, 0.02)
        np.testing.assert_allclose(result["ci95"], np.quantile(expected, [0.025, 0.975]))
        self.assertEqual(sampled.shape, (10_000, 320))
        self.assertEqual(result["unit"], "paired image; each draw index is applied to all three seeds")

    def test_slot_reference_is_pinned_and_self_contained(self):
        ref = summarize.json.loads(summarize.SLOT_REFERENCE.read_text(encoding="utf-8"))
        self.assertEqual(summarize._sha(summarize.SLOT_REFERENCE),
                         summarize.EXPECTED_SLOT_REFERENCE_SHA256)
        self.assertEqual(ref["source_summary_sha256"],
                         "85f5f36c9900e31841909d7d7e50855f5bb24a3f2d567e05846b67266923056e")
        self.assertEqual(ref["training_pool"]["unique_images"], 70000)
        self.assertEqual(ref["evaluation"]["ids_inclusive"], [1320, 1639])
        self.assertEqual(ref["evaluation"]["count"], 320)

    def test_training_source_report_sha_must_match_frozen_source_reference(self):
        fields = ("source_core_sha256", "source_manifest_sha256",
                  "source_evaluation_sha256", "training_ids_sha256")
        artifact = {name: f"sha-{name}" for name in fields}
        source = dict(artifact)
        summarize._verify_training_source_binding(artifact, source)
        source["source_evaluation_sha256"] = "different-report"
        with self.assertRaisesRegex(AssertionError, "source_evaluation_sha256"):
            summarize._verify_training_source_binding(artifact, source)

    def test_score_rejects_nan_and_recomputed_mean_mismatch(self):
        vals = [0.5] * 320
        report = {"ids": [1320, 1639], "images": 320,
                  "ground_truth_used_for_prediction": False,
                  "sweep": [{"scored_targets": {"our_hdf5": {
                      "metrics": {m: 0.5 for m in summarize.METRICS},
                      "valid_count": {m: 320 for m in summarize.METRICS},
                      "per_image": {m: list(vals) for m in summarize.METRICS}}}}]}
        report["_path"] = __file__
        self.assertEqual(summarize._score(report, seed=1, arm=None)["metrics"]["fg_ari"], 0.5)
        report["sweep"][0]["scored_targets"]["our_hdf5"]["per_image"]["fg_ari"][17] = float("nan")
        with self.assertRaises(AssertionError):
            summarize._score(report, seed=1, arm=None)

    def test_all_registered_stages_must_validate_before_summary(self):
        tasks = [{"task_id": f"t{i}"} for i in range(13)]
        with mock.patch.object(summarize.coordinator, "task_plan", return_value=tasks), \
                mock.patch.object(summarize.coordinator, "valid_result", side_effect=[True] * 12 + [False]):
            with self.assertRaisesRegex(RuntimeError, "t12"):
                summarize._assert_all_tasks_valid()


if __name__ == "__main__":
    unittest.main()
