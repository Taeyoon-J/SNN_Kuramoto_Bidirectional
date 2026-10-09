import numpy as np  # Import NumPy before torch on Windows.
import unittest
from pathlib import Path
from unittest import mock

import collaborative_test.SW_0118_allowed_patch_labels.run as run_module
from collaborative_test.SW_0118_allowed_patch_labels.run import (
    METRICS,
    _assert_matches_reference,
    _assert_matches_slot_means,
    _ids,
    _tensor_labels,
    _validate_released_slot_protocol,
    OURS_PREDICTION_SHA,
    validate_registered_hashes,
    validate_sha256_literal,
)


def metric_rows(values, mean=None):
    values = [float(x) for x in values]
    return {metric: {"per_image": values.copy(), "mean": (sum(values) / len(values) if mean is None else mean),
                     "valid_count": len(values)} for metric in METRICS}


class FrozenEvaluationContractTests(unittest.TestCase):
    def test_frozen_prediction_requires_exact_grid_and_integer_labels(self):
        labels = np.zeros((320, 16, 16), dtype=np.int32)
        self.assertEqual(tuple(_tensor_labels(labels, "fixture").shape), (320, 16, 16))
        with self.assertRaises(ValueError):
            _tensor_labels(np.zeros((319, 16, 16), dtype=np.int64), "fixture")
        with self.assertRaises(ValueError):
            _tensor_labels(np.zeros((320, 16, 16), dtype=np.float32), "fixture")

    def test_image_order_must_be_the_registered_contiguous_slice(self):
        self.assertEqual(_ids(list(range(1320, 1640)), "ids"), list(range(1320, 1640)))
        with self.assertRaises(ValueError):
            _ids(list(range(1321, 1641)), "ids")
        with self.assertRaises(ValueError):
            _ids(list(reversed(range(1320, 1640))), "ids")

    def test_source_reference_requires_exact_three_metrics_and_counts(self):
        expected = {"metrics": {}, "valid_count": {}, "per_image": {}}
        for metric in METRICS:
            expected["metrics"][metric] = 0.5
            expected["valid_count"][metric] = 320
            expected["per_image"][metric] = [0.5] * 320
        actual = metric_rows([0.5] * 320)
        _assert_matches_reference(actual, expected, "source fixture")
        bad = metric_rows([0.5] * 320)
        bad["fg_ari"]["per_image"][17] = 0.6
        with self.assertRaises(ValueError):
            _assert_matches_reference(bad, expected, "source fixture")

    def test_slot_mean_check_rejects_invalid_count_or_drift(self):
        expected = {"mean": {m: 0.5 for m in METRICS},
                    "valid_count": {m: 320 for m in METRICS}}
        _assert_matches_slot_means(metric_rows([0.5] * 320), expected, "slot fixture")
        wrong = dict(expected)
        wrong["mean"] = dict(expected["mean"], fg_ari=0.5001)
        with self.assertRaises(ValueError):
            _assert_matches_slot_means(metric_rows([0.5] * 320), wrong, "slot fixture")
        wrong_count = {"mean": expected["mean"], "valid_count": dict(expected["valid_count"], fg_ari=319)}
        with self.assertRaises(ValueError):
            _assert_matches_slot_means(metric_rows([0.5] * 320), wrong_count, "slot fixture")

    def test_released_slot_protocol_matches_registered_absolute_checkpoint(self):
        protocol = {
            "image_ids": [1320, 1639], "count": 320, "seed": 0, "batch_size": 1,
            "num_slots": 11, "iterations": 3,
            "checkpoint_source": "gs://gresearch/slot-attention/object-discovery/ckpt-500",
            "checkpoint_prefix": "/Data0/kevinswk/patch_v2/trained_models/slot_attention_official_patch_eval_20261002/checkpoint/ckpt-500",
            "ground_truth_used_for_prediction": False,
            "model_sha256": "96c2b12d8b28c22fd2605eccf9027bda304f38bb9332f8c1620898f471c67ec4",
        }
        _validate_released_slot_protocol(protocol)
        wrong = dict(protocol, checkpoint_prefix="peerassets/checkpoint/ckpt-500")
        with self.assertRaises(ValueError):
            _validate_released_slot_protocol(wrong)

    def test_frozen_ours_prediction_hashes_are_bound_for_all_seeds(self):
        self.assertIsNone(OURS_PREDICTION_SHA[0])  # Seed0 hash comes from the bound SW0114 report.
        self.assertEqual(OURS_PREDICTION_SHA[1], "2eb0407714549a7d09d6bb55925095a8385537da8de9a3352be55316ce409225")
        self.assertEqual(OURS_PREDICTION_SHA[2], "da82e728097eef6266bada9d37dd4aa4454479c6d658f35a703d3ad4bfebcb20")

    def test_every_registered_hash_is_well_formed_before_any_provider_load(self):
        validate_registered_hashes()
        self.assertEqual(len(validate_sha256_literal("a" * 64, "fixture")), 64)
        with self.assertRaises(ValueError):
            validate_sha256_literal("f5c80fceb2506d5356aa23ac39556448ebcb4f9454eeebabdc2ef148b1fcbd", "bad fixture")
        with mock.patch.dict(run_module.SLOT_PREDICTION_SHA, {1: "f5c80fceb2506d5356aa23ac39556448ebcb4f9454eeebabdc2ef148b1fcbd"}), \
             mock.patch.object(run_module, "_load_ours", side_effect=AssertionError("provider must not load")):
            with self.assertRaisesRegex(ValueError, "Registered matched Slot seed1 prediction SHA"):
                run_module.run(Path("unused-output"), run_module.DATASET)


if __name__ == "__main__":
    unittest.main()
