import json
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import summarize


def report(values):
    return {"ids": [1320, 1351], "images": 32,
            "sweep": [{"synchrony_threshold": .35,
                       "scored_targets": {"our_hdf5": {"metrics": values}}}]}


class TrajectorySummaryTests(unittest.TestCase):
    def test_fixed_five_epoch_delta_gate(self):
        baseline = {"fg_ari": .5, "foreground_iou": .4, "matched_object_iou": .3}
        reference = {key: value + .01 for key, value in baseline.items()}
        payloads = {"baseline": baseline, "sw0081": reference}
        for epoch in range(1, 6):
            payloads[f"e{epoch}"] = {
                key: value + (-.01 if epoch == 3 and key == "fg_ari" else .01)
                for key, value in baseline.items()}
        with patch("summarize.read_metrics", side_effect=lambda path: payloads[str(path)]):
            result = summarize("baseline", {e: f"e{e}" for e in range(1, 6)}, "sw0081")
        self.assertFalse(result["epochs"]["3"]["advance"])
        self.assertTrue(result["epochs"]["2"]["advance"])

    def test_requires_all_epochs(self):
        with self.assertRaises(ValueError):
            summarize("missing", {1: "only-one"}, "missing-reference")


if __name__ == "__main__":
    unittest.main()
