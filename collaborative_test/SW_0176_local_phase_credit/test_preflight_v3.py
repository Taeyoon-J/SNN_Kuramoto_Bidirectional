"""CPU contract tests for SW0176's public cached-source preparation path."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np  # Keep NumPy/OpenMP initialization before torch.
import torch

from collaborative_test.SW_0176_local_phase_credit import preflight_v3 as preflight
from collaborative_test.SW_0175_critical_component_binding import train_v2 as train


class PreflightContractTests(unittest.TestCase):
    def test_public_run_preflight_selects_cuda_device_before_peak_stats(self):
        events = []
        class FakeCuda:
            @staticmethod
            def is_initialized():
                return False
            @staticmethod
            def set_device(device):
                events.append(("set_device", str(device)))
            @staticmethod
            def reset_peak_memory_stats(device):
                events.append(("reset_peak", str(device)))
            @staticmethod
            def empty_cache():
                pass

        class StopAfterInit(Exception):
            pass

        prepared = {"test_prepared": True}
        with mock.patch.object(preflight.torch, "cuda", FakeCuda), \
             mock.patch.object(preflight.torch, "set_num_threads"), \
             mock.patch.object(preflight.train.preflight, "_state_rng", return_value={}), \
             mock.patch.object(preflight.train.preflight, "_restore_rng"), \
             mock.patch.object(preflight, "_build_source", side_effect=StopAfterInit):
            with self.assertRaises(StopAfterInit):
                preflight.run_preflight(device="cuda:2", prepared=prepared,
                                        output="unused-report.json")
        self.assertEqual(events, [("set_device", "cuda:2"), ("reset_peak", "cuda:2")])

    @staticmethod
    def _arrays():
        return {
            "image_ids": np.arange(512, dtype=np.int64),
            "rgb": np.broadcast_to(np.zeros((1, 128, 128, 3), dtype=np.uint8),
                                   (512, 128, 128, 3)),
            "source_support": np.broadcast_to(np.zeros((1, 32, 32), dtype=np.int32),
                                              (512, 32, 32)),
            "feasible_targets": np.broadcast_to(np.zeros((1, 32, 32), dtype=np.int32),
                                                (512, 32, 32)),
            "modal_gt": np.broadcast_to(np.zeros((1, 32, 32), dtype=np.int32),
                                         (512, 32, 32)),
        }

    @staticmethod
    def _proof():
        return {
            "source_checkpoint_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "source_cache_state_sha256": "c" * 64,
            "producer_manifest_sha256": "d" * 64,
            "producer_arrays_sha256": "e" * 64,
            "ordered_train_ids": list(range(512)),
        }

    def _control(self, proof):
        bindings = {
            "arm": "critical_component", "seed": 1,
            "source_checkpoint_sha256": proof["source_checkpoint_sha256"],
            "source_manifest_sha256": proof["source_manifest_sha256"],
            "source_state_sha256": proof["source_cache_state_sha256"],
            "input_cache_manifest_sha256": proof["producer_manifest_sha256"],
            "input_cache_arrays_sha256": proof["producer_arrays_sha256"],
            "training_images_per_pass": 512, "passes": 2, "updates": 64,
            "logical_batch": 16, "microbatch": 4, "optimizer": "Adam",
            "context_seed": 171001, "shuffle_seed": 171001,
            "ordered_train_ids": list(range(512)),
            "ordered_train_ids_sha256": train._sha_bytes(np.arange(512, dtype="<i8").tobytes()),
            "training_fingerprint": train._training_fingerprint(),
            "initial_model_state_sha256": "f" * 64,
        }
        return {"experiment": train.EXPERIMENT, "status": "training_complete",
                "arm": "critical_component", "seed": 1, "bindings": bindings}

    def test_actual_public_cpu_prepare_adapter_accepts_registered_cache_schema(self):
        arrays = self._arrays()
        proof = self._proof()
        control = self._control(proof)
        admitted = {"prepared": {"arrays": arrays}, "proof": proof,
                    "admission_report": {"initialization": {}}}
        with mock.patch.object(train, "_load_admitted_inputs", return_value=admitted), \
             mock.patch.object(train, "validate_training_artifact", return_value=control):
            prepared = preflight.prepare_inputs_cpu()
        self.assertEqual(prepared["selected_image_id"], 0)
        self.assertEqual(prepared["source_proof"], proof)
        self.assertFalse(prepared["validation_data_read"])
        self.assertEqual(prepared["optimizer_updates"], 0)
        self.assertEqual(prepared["arrays"]["rgb"].shape, (512, 128, 128, 3))
        self.assertEqual(prepared["arrays"]["modal_gt"].shape, (512, 32, 32))

    def test_tail_control_rejects_objective_or_source_cache_lineage_changes(self):
        proof = self._proof()
        control = self._control(proof)
        preflight._validate_tail_control(control, proof)
        changed = {**control, "bindings": {**control["bindings"],
            "training_fingerprint": {**control["bindings"]["training_fingerprint"],
                "files": {**control["bindings"]["training_fingerprint"]["files"],
                    "collaborative_test/SW_0175_critical_component_binding/critical_component_objective.py": "0" * 64}}}}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            preflight._validate_tail_control(changed, proof)
        with self.assertRaisesRegex(ValueError, "input_cache_manifest_sha256"):
            preflight._validate_tail_control(control,
                {**proof, "producer_manifest_sha256": "1" * 64})

    def test_cache_adapter_rejects_missing_or_wrong_dtype_modal_targets(self):
        arrays = self._arrays()
        with self.assertRaisesRegex(ValueError, "array set"):
            preflight._validate_arrays({key: value for key, value in arrays.items()
                                        if key != "modal_gt"})
        arrays["modal_gt"] = arrays["modal_gt"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "integer labels"):
            preflight._validate_arrays(arrays)

    def test_full_window_gradient_gate_requires_finite_actual_families(self):
        evidence = {name: {"finite": True, "gradient_l2": 0.25}
                    for name in ("live_context_encoder", "graph", "kuramoto", "dendrite", "membrane")}
        preflight._validate_gradient_evidence(evidence, 1.0, 0.5)
        for bad in ({**evidence, "graph": {"finite": False, "gradient_l2": 0.25}},
                    {**evidence, "kuramoto": {"finite": True, "gradient_l2": 0.0}}):
            with self.assertRaises(FloatingPointError):
                preflight._validate_gradient_evidence(bad, 1.0, 0.5)
        with self.assertRaises(FloatingPointError):
            preflight._validate_gradient_evidence(evidence, 0.0, 0.5)

    def test_context_hash_uses_actual_sw0175_producer_report_level(self):
        actual_producer_shape = {"context_zero_init_sha256": "a" * 64,
            "initialization": {"base_encoder_sha256": "b" * 64,
                            "feature_clip": 2.0, "integration": {}}}
        self.assertEqual(preflight._context_zero_init_sha(actual_producer_shape), "a" * 64)
        with self.assertRaisesRegex(ValueError, "top-level"):
            preflight._context_zero_init_sha({"initialization": {
                "context_zero_init_sha256": "a" * 64}})


if __name__ == "__main__":
    unittest.main()
