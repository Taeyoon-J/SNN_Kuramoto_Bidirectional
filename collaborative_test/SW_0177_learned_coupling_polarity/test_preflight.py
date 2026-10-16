"""CPU contracts for SW0177 one-image source preflight preparation."""
from __future__ import annotations

import json
from unittest import mock
import unittest

import numpy as np  # Initialize NumPy/OpenMP before Torch.
import torch

from collaborative_test.SW_0177_learned_coupling_polarity import preflight
from collaborative_test.SW_0175_critical_component_binding import train_v2 as train


class PreflightTests(unittest.TestCase):
    @staticmethod
    def arrays():
        return {
            "image_ids": np.arange(512, dtype=np.int64),
            "rgb": np.broadcast_to(np.zeros((1, 128, 128, 3), dtype=np.uint8),
                                   (512, 128, 128, 3)),
            "source_support": np.broadcast_to(np.zeros((1, 32, 32), dtype=np.int32),
                                              (512, 32, 32)),
            "feasible_targets": np.broadcast_to(np.zeros((1, 32, 32), dtype=np.int32),
                                                (512, 32, 32)),
        }

    def test_prepare_inputs_uses_actual_sw0175_cpu_producer_contract(self):
        arrays = self.arrays()
        proof = {"source_cache_state_sha256": "a" * 64,
                 "ordered_train_ids": list(range(512)),
                 "validation_data_read": False}
        admitted = {"prepared": {"arrays": arrays}, "proof": proof,
                    "admission_report": {"context_zero_init_sha256": "b" * 64}}
        with mock.patch.object(train, "_load_admitted_inputs", return_value=admitted):
            prepared = preflight.prepare_inputs_cpu()
        self.assertEqual(prepared["selected_position"], 0)
        self.assertEqual(prepared["selected_image_id"], 0)
        self.assertIs(prepared["arrays"], arrays)
        self.assertEqual(prepared["source_proof"], proof)
        self.assertFalse(prepared["validation_data_read"])
        self.assertEqual(prepared["optimizer_updates"], 0)
        json.dumps(prepared["source_proof"], allow_nan=False)
        self.assertIn("collaborative_test/SW_0177_learned_coupling_polarity/resource_queue.py",
                      prepared["implementation_fingerprint"])

    def test_prepare_inputs_rejects_wrong_real_cache_geometry(self):
        arrays = self.arrays()
        arrays["rgb"] = arrays["rgb"][:, :, :, :2]
        admitted = {"prepared": {"arrays": arrays}, "proof": {}, "admission_report": {}}
        with mock.patch.object(train, "_load_admitted_inputs", return_value=admitted):
            with self.assertRaisesRegex(ValueError, "identity/RGB schema"):
                preflight.prepare_inputs_cpu()

    def test_actual_source_factory_is_public_and_requires_cuda_hidden(self):
        prepared = {"factory_fixture": True}
        with mock.patch.object(preflight.torch.cuda, "is_initialized", return_value=False), \
             mock.patch.object(preflight, "_build_source", return_value=(1, 2, 3, 4)) as factory:
            self.assertEqual(preflight.build_source(prepared, "cpu"), (1, 2, 3, 4))
            factory.assert_called_once_with(prepared, "cpu")
        with mock.patch.object(preflight.torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "precede CUDA"):
                preflight.build_source(prepared, "cpu")

    def test_context_initialization_digest_requires_actual_report_field_level(self):
        report = {"context_zero_init_sha256": "c" * 64,
                  "initialization": {"context_zero_init_sha256": "d" * 64}}
        self.assertEqual(preflight._context_sha(report), "c" * 64)
        with self.assertRaisesRegex(ValueError, "zero-context"):
            preflight._context_sha({"initialization": {"context_zero_init_sha256": "c" * 64}})


if __name__ == "__main__":
    unittest.main()
