"""Synthetic tests for seed-complete, validation-only spatial sweep summary."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("summarize_three_seed.py")
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("summarize_three_seed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_report(seed):
    rows = []
    for mode, sigma in (("spike", None), ("spike_spatial", 1.5), ("spatial_only", "inf")):
        value = 0.4 + 0.1 * seed
        rows.append({
            "affinity_mode": mode, "spatial_sigma": sigma,
            "synchrony_threshold": 0.2,
            "predicted_foreground_fraction": value,
            "predicted_object_count": {"mean": 3 + seed, "empty_image_count": seed},
            "scored_targets": {"our_hdf5": {"metrics": {
                "fg_ari": value, "foreground_iou": value + .1,
                "matched_object_iou": value + .2,
            }}},
        })
    return json.dumps({
        "split": "validation", "ground_truth_used_for_prediction": False,
        "target_sources": {"our_hdf5": {"path": "synthetic"}},
        "sweep": rows,
    })


class ThreeSeedSummaryTest(unittest.TestCase):
    def test_common_configs_report_three_seed_mean_and_sample_std(self):
        def fake_report_text(path, encoding=None):
            seed = int(path.parent.name[-1])
            return make_report(seed)

        with mock.patch.object(Path, "is_file", return_value=True), \
                mock.patch.object(Path, "read_text", fake_report_text):
            report = MODULE.summarize_window(Path("synthetic-root"), "short")
        self.assertEqual(report["validation_only"], True)
        self.assertEqual(report["selection_performed"], False)
        self.assertEqual(report["common_configuration_count"], 3)
        self.assertEqual(report["sw0042_spike_control_common_count"], 1)
        spike = next(row for row in report["summary_rows"] if row["affinity_mode"] == "spike")
        self.assertAlmostEqual(spike["metrics"]["fg_ari"]["mean"], .5)
        self.assertAlmostEqual(spike["metrics"]["fg_ari"]["std_sample"], .1)
        for actual, expected in zip(spike["metrics"]["fg_ari"]["per_seed"], [.4, .5, .6]):
            self.assertAlmostEqual(actual, expected)

    def test_missing_seed_json_fails_clearly(self):
        def file_exists(path):
            return not (path.parent.name.endswith("_s2")
                        and path.name == "spatial_affinity_long.json")

        with mock.patch.object(Path, "is_file", file_exists), \
                mock.patch.object(Path, "read_text", lambda path, encoding=None: make_report(
                    int(path.parent.name[-1]))):
            with self.assertRaisesRegex(FileNotFoundError, "seed2 long validation JSON"):
                MODULE.summarize_window(Path("synthetic-root"), "long")

    def test_legacy_sw0042_row_defaults_to_spike_only(self):
        self.assertEqual(MODULE._row_key({"synchrony_threshold": .2}),
                         ("spike", None, .2))


if __name__ == "__main__":
    unittest.main()
