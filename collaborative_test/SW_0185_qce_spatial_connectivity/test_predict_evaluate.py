"""CPU contracts for SW0185 prediction bindings and endpoint support."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0185_qce_spatial_connectivity import predict_evaluate as pe


class PredictionAdapterTests(unittest.TestCase):
    def test_new_objective_loads_original_SourceT_factory_before_trained_state(self):
        # A loss name is not a supported source-factory arm. Preserve the
        # original source architecture and load the new trained tensors later.
        prepared={"manifest":{},"artifact":Path("unused"),"training_inputs":object()}
        with mock.patch.object(pe.train.base,"_make_fresh",side_effect=LookupError("factory reached")) as factory:
            with self.assertRaisesRegex(LookupError,"factory reached"):
                pe._load_model(2,prepared,"cpu")
        self.assertEqual(factory.call_args.args[:2],("product_q_ce",2))

    def test_prediction_cli_help_is_wired(self):
        with self.assertRaises(SystemExit) as help_exit:
            pe.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)

    def test_prediction_roots_and_bindings_are_seed_specific_and_manifest_bound(self):
        self.assertNotEqual(pe._out(0), pe._out(2))
        self.assertNotEqual(pe._work(0), pe._work(2))
        base = {"manifest_sha256": "a" * 64, "training_state_sha256": "b" * 64,
            "queue_fingerprint_sha256": "c" * 64, "model_state_sha256": "d" * 64,
            "manifest": {"bindings": {"source_binding": {"seed": 2}}},
            "references": {"source_T": {"protocol_sha256": "e" * 64}},
            "training_inputs": {"dataset_manifest_sha256": "f" * 64,
                "dataset_arrays_sha256": "1" * 64}, "seed": 2}
        bindings = pe._bindings(base)
        self.assertEqual(bindings["seed"], 2)
        self.assertEqual(bindings["training_manifest_sha256"], "a" * 64)
        self.assertEqual(bindings["source_binding"]["seed"], 2)

    def test_conditional_and_standalone_masks_keep_distinct_foreground_contracts(self):
        count, grid = pe.COUNT, pe.GRID
        source = np.zeros((count, grid, grid), dtype=bool)
        source[:, :8, :8] = True
        support = np.zeros((count, grid, grid), dtype=np.int32)
        support[:, 16:, 16:] = 2
        conditional = np.zeros((count, grid, grid), dtype=np.int32)
        conditional[:, :8, :8] = 1
        standalone = support.copy()
        checked, checked_support = pe.native_eval.validate_prediction_arrays(
            {"conditional": conditional, "standalone": standalone}, support, source)
        self.assertTrue(np.array_equal(checked["conditional"] > 0, source))
        self.assertTrue(np.array_equal(checked["standalone"] > 0, checked_support > 0))
        self.assertFalse(np.array_equal(checked["standalone"] > 0, source))
        bad = standalone.copy(); bad[:, 0, 0] = 1
        with self.assertRaises(ValueError):
            pe.native_eval.validate_prediction_arrays(
                {"conditional": conditional, "standalone": bad}, support, source)

    def test_prediction_bindings_reject_cross_seed_prepared_input(self):
        prepared = {"seed": 0}
        self.assertNotEqual(prepared["seed"], 1)
        # The public prediction entrypoint performs this identity check before CUDA.
        with self.assertRaisesRegex(ValueError, "another seed"):
            pe.predict_seed(1, prepared=prepared)


if __name__ == "__main__":
    unittest.main()
