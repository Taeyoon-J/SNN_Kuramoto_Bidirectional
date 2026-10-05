import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import EXPECTED, summarize_reports


def make_report(vth, improvement=0.0, event_rate=1.0):
    return {
        "ids": [1320, 1351], "images": 32,
        "inference": {"steps": 256, "settle": 64, "membrane_vth": vth,
                      "synchrony_thresholds": [.35], "affinity_modes": ["spike"]},
        "event_diagnostics": {"binary_event_rate": event_rate,
                              "binary_always_on_fraction": event_rate,
                              "binary_constant_history_fraction": event_rate,
                              "binary_temporal_std_mean": 0.0 if event_rate == 1 else .2,
                              "shape_after_settle": [32, 4, 256, 192],
                              "binary_values_only": True},
        "sweep": [{"affinity_mode": "spike", "synchrony_threshold": .35,
                   "scored_targets": {"our_hdf5": {"metrics": {
                       "fg_ari": .5 + improvement, "foreground_iou": .4 + improvement,
                       "matched_object_iou": .3 + improvement}}}}],
    }


class SummaryTests(unittest.TestCase):
    def test_fixed_sweep_and_diagnostics(self):
        reports = {tag: make_report(vth, .02 if tag == "v20" else 0,
                                    .3 if tag == "v20" else 1.0)
                   for tag, vth in EXPECTED.items()}
        result = summarize_reports(reports)
        self.assertTrue(result["thresholds"]["v20"]["all_three_metrics_improve_vs_v006"])
        self.assertAlmostEqual(result["thresholds"]["v20"]["binary_event_rate"], .3)
        self.assertFalse(result["thresholds"]["v05"]["all_three_metrics_improve_vs_v006"])

    def test_rejects_missing_or_protocol_mismatch(self):
        with self.assertRaises(ValueError):
            summarize_reports({"v006": make_report(.06)})
        reports = {tag: make_report(vth) for tag, vth in EXPECTED.items()}
        reports["v10"]["inference"]["steps"] = 64
        with self.assertRaises(ValueError):
            summarize_reports(reports)


if __name__ == "__main__":
    unittest.main()
