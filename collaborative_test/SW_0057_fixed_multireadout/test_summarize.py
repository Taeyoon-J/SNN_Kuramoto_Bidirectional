import json
import unittest
from pathlib import Path
from unittest.mock import patch

from summarize import summarize
from gamma_contract import aligned_gamma_prefix
from target_contract import require_complete_predictions
import numpy as np


def fake_report(seed):
    rows = []
    for name in ("spike_cc_threshold_0p50", "membrane_spatial_sigma1p5_k10"):
        rows.append({"readout": name, "metrics": {"fg_ari": seed / 10, "foreground_iou": .2, "matched_object_iou": .3},
                     "predicted_object_count_mean": 5., "predicted_foreground_fraction": .4})
    return {"schema_version": 1, "experiment": "SW0057 fixed multi-readout", "seed": seed,
            "split": "fixed_hdf5_aligned_validation", "ids": [1320, 1639], "images": 320,
            "checkpoint": {"sha256": "a" * 64}, "evaluator": {"sha256": "b" * 64, "dependency_code_sha256": "c" * 64},
            "gamma": {"sha256": "d" * 64, "manifest_sha256": "e" * 64},
            "inference": {"steps": 1024, "settle": 512},
            "readout_contract": {"ground_truth_used_for_prediction": False}, "rows": rows}


class SummaryContractTest(unittest.TestCase):
    def test_targets_require_both_completed_prediction_sets(self):
        with self.assertRaises(ValueError):
            require_complete_predictions({}, 4)
        with self.assertRaises(ValueError):
            require_complete_predictions({"spike_cc_threshold_0p50": [np.zeros((4, 16, 16))],
                                         "membrane_spatial_sigma1p5_k10": []}, 4)
        with self.assertRaises(ValueError):
            require_complete_predictions({"spike_cc_threshold_0p50": [np.zeros((4, 16, 16))],
                                         "membrane_spatial_sigma1p5_k10": [np.zeros((1, 16, 16))]}, 4)
        require_complete_predictions({"spike_cc_threshold_0p50": [np.zeros((3, 16, 16)), np.zeros((1, 16, 16))],
                                      "membrane_spatial_sigma1p5_k10": [np.zeros((2, 16, 16)), np.zeros((2, 16, 16))]}, 4)

    def test_gamma_preflight_prefix_and_invalid_count(self):
        digest = "f" * 64
        manifest = {"image_ids": [1320, 1639], "shape": [320, 8, 256], "dtype": "torch.float32",
                    "patch_grid_size": 16, "gamma_sha256": digest}
        gamma = np.arange(320 * 8 * 256, dtype=np.float32).reshape(320, 8, 256)
        prefix = aligned_gamma_prefix(gamma, manifest, digest, 1320, 4, 1024, 512)
        self.assertEqual(prefix.shape, (4, 8, 256))
        self.assertTrue(np.array_equal(prefix, gamma[:4]))
        self.assertEqual(aligned_gamma_prefix(gamma, manifest, digest, 1320, 320, 1024, 512).shape, (320, 8, 256))
        for count in (1, 5, 64):
            with self.assertRaises(ValueError):
                aligned_gamma_prefix(gamma, manifest, digest, 1320, count, 1024, 512)
        with self.assertRaises(ValueError):
            aligned_gamma_prefix(gamma[:4], manifest, digest, 1320, 4, 1024, 512)
        for key, value in (("image_ids", [1320, 1638]), ("shape", [4, 8, 256]),
                           ("gamma_sha256", "0" * 64), ("patch_grid_size", 8),
                           ("dtype", "torch.float64")):
            bad = dict(manifest); bad[key] = value
            with self.assertRaises(ValueError):
                aligned_gamma_prefix(gamma, bad, digest, 1320, 4, 1024, 512)
        with self.assertRaises(ValueError):
            aligned_gamma_prefix(gamma, manifest, digest, 1000, 4, 1024, 512)
        with self.assertRaises(ValueError):
            aligned_gamma_prefix(gamma, manifest, digest, 1320, 4, 256, 64)

    def test_three_seed_means_and_missing_seed(self):
        blobs = {seed: fake_report(seed) for seed in (0, 1, 2)}

        def exists(path):
            seed = int(path.parent.name.split("_s")[1].split("_")[0])
            return seed in blobs

        def read_text(path, encoding=None):
            seed = int(path.parent.name.split("_s")[1].split("_")[0])
            if seed not in blobs:
                raise FileNotFoundError(path)
            return json.dumps(blobs[seed])

        with patch.object(Path, "is_file", exists), patch.object(Path, "read_text", read_text):
            result = summarize(Path("/fake/model_root"))
            self.assertAlmostEqual(result["readouts"]["spike_cc_threshold_0p50"]["fg_ari"]["mean"], .1)
            del blobs[2]
            with self.assertRaises(FileNotFoundError):
                summarize(Path("/fake/model_root"))


if __name__ == "__main__":
    unittest.main()
