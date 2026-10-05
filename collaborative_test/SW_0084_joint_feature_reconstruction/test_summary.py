import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import ARMS, EPOCHS, METRICS, summarize


class PilotSummaryTests(unittest.TestCase):
    def test_all_arm_epochs_compare_fixed_three_metrics(self):
        baseline = {"fg_ari": .5, "foreground_iou": .4, "matched_object_iou": .3}
        reports = {"baseline_epoch10": baseline}
        for arm_index, arm in enumerate(ARMS):
            for epoch in EPOCHS:
                delta = .01 if arm == "A" and epoch == "10" else -.01
                reports[f"{arm}_epoch{epoch}"] = {
                    key: value + delta for key, value in baseline.items()}
        result = summarize(reports)
        self.assertTrue(result["arms"]["A"]["10"]["all_three_strictly_improved"])
        self.assertFalse(result["arms"]["B"]["10"]["all_three_strictly_improved"])
        self.assertEqual(set(result["arms"]), set(ARMS))
        self.assertEqual(set(METRICS), set(result["arms"]["A"]["10"]["metrics"]))

    def test_incomplete_arm_matrix_rejected(self):
        with self.assertRaises(ValueError):
            summarize({"baseline_epoch10": {}})


if __name__ == "__main__":
    unittest.main()
