import unittest

from collaborative_test.SW_0058_component_spike_only_loss.summarize import (
    canonical_argv, fixed_spike_metrics, fixed_multireadout_metrics,
)


class SummaryContractTests(unittest.TestCase):
    def test_canonical_argv_ignores_only_seed_and_save_path(self):
        a = ["--seed", "0", "--save-path", "/tmp/a.pt", "--lr", "0.0003", "--primary-loss-weight", "0"]
        b = ["--seed", "2", "--save-path", "/tmp/b.pt", "--lr", "0.0003", "--primary-loss-weight", "0"]
        self.assertEqual(canonical_argv(a), canonical_argv(b))
        self.assertNotEqual(canonical_argv(a), canonical_argv(a[:-1] + ["1"]))

    def test_primary_metric_extraction_requires_fixed_threshold_and_split(self):
        metrics = {"fg_ari": .4, "foreground_iou": .3, "matched_object_iou": .2}
        report = {
            "ground_truth_used_for_prediction": False,
            "ids": [1320, 1639], "images": 320,
            "inference": {"steps": 1024, "settle": 512},
            "sweep": [{"affinity_mode": "spike", "synchrony_threshold": .50,
                       "scored_targets": {"our_hdf5": {"metrics": metrics}}}],
        }
        self.assertEqual(fixed_spike_metrics(report), metrics)
        report["sweep"][0]["synchrony_threshold"] = .35
        with self.assertRaises(ValueError):
            fixed_spike_metrics(report)

    def test_secondary_readout_set_is_fixed(self):
        report = {
            "experiment": "SW0057 fixed multi-readout", "seed": 1,
            "ids": [1320, 1639], "images": 320,
            "inference": {"steps": 1024, "settle": 512},
            "readout_contract": {"ground_truth_used_for_prediction": False},
            "rows": [
                {"readout": "spike_cc_threshold_0p50", "metrics": {
                    "fg_ari": .3, "foreground_iou": .4, "matched_object_iou": .2}},
                {"readout": "membrane_spatial_sigma1p5_k10", "metrics": {
                    "fg_ari": .5, "foreground_iou": .2, "matched_object_iou": .4}},
            ],
        }
        result = fixed_multireadout_metrics(report, 1)
        self.assertEqual(result["membrane_spatial_sigma1p5_k10"]["fg_ari"], .5)
        report["rows"].pop()
        with self.assertRaises(ValueError):
            fixed_multireadout_metrics(report, 1)


if __name__ == "__main__":
    unittest.main()
