import unittest

from collaborative_test.SW_0086_sw0072_on_peer_contract import summarize


class SummaryTest(unittest.TestCase):
    def test_three_seed_mean_and_slot_gate(self):
        reports = [{
            "seed": seed, "peer_rows": [6000, 6299], "images": 300,
            "ground_truth_used_for_prediction": False,
            "metrics": {key: value for key in summarize.METRICS},
        } for seed, value in enumerate((.7, .8, .9))]
        result = summarize.summarize_reports(reports)
        self.assertEqual(tuple(reports[0]["metrics"]),
                         ("fg_ari", "foreground_iou", "matched_object_iou"))
        self.assertAlmostEqual(result["means"]["fg_ari"], .8)
        self.assertTrue(result["exceeds_peer_slot_all_three"])

    def test_rejects_prefixed_metric_schema(self):
        reports = [{
            "seed": seed, "peer_rows": [6000, 6299], "images": 300,
            "ground_truth_used_for_prediction": False,
            "metrics": {f"patch_{key}": .8 for key in summarize.METRICS},
        } for seed in range(3)]
        with self.assertRaises(KeyError):
            summarize.summarize_reports(reports)


if __name__ == "__main__":
    unittest.main()

