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
        self.assertAlmostEqual(result["means"]["patch_fg_ari"], .8)
        self.assertTrue(result["exceeds_peer_slot_all_three"])


if __name__ == "__main__":
    unittest.main()

