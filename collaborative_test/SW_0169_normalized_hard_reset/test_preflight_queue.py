from __future__ import annotations

import inspect
import unittest
from unittest import mock

from collaborative_test.SW_0169_normalized_hard_reset import preflight, preflight_queue


class PreflightQueueContractTests(unittest.TestCase):
    def _source(self):
        return {
            "seed": 1, "selected_ids": list(range(8)), "positions": list(range(8)),
            "source_checkpoint_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "source_cache_manifest_sha256": "c" * 64,
            "source_cache_bindings_sha256": "d" * 64,
            "source_cache_state_sha256": "e" * 64,
            "source_cache_trace_sha256": ["f" * 64] * 8,
            "context_cache_manifest_sha256": "1" * 64,
            "context_cache_bindings_sha256": "2" * 64,
            "context_cache_state_sha256": "3" * 64,
            "context_rows": [{"q_sha256": "4" * 64, "adjacency_sha256": "5" * 64,
                              "context_file_sha256": "6" * 64}] * 8,
            "rgb_sha256": ["7" * 64] * 8, "rgb_manifest_sha256": "8" * 64,
            "hdf5_path": "registered/train.h5", "hdf5_size_bytes": 10,
            "hdf5_mtime_ns": 11,
            "implementation_fingerprint": {"leaf": "9" * 64},
        }

    def _report(self, admitted=True):
        source = self._source()
        preflight_fp = preflight.implementation_fingerprint()
        rows = [{"update": update, "batch_start": start,
                 "mean_loss_micro": 1.0, "dL_dQ_norm": 0.2, "dL_dS_norm": 0.3,
                 **({"critical_credit": True} if start == 4 else {})}
                for update in (1, 2) for start in (0, 4)]
        return {
            "experiment": "SW0169_normalized_hard_reset", "source": source,
            "implementation_fingerprint": preflight_fp,
            "queue_fingerprint_sha256": preflight_queue.fingerprint_sha256(),
            "calibration": {
                "source_six_trace_parity": True, "source_q_parity": True,
                "source_a_parity": True, "source_cache_replay_exact": True,
                "source_model_state_sha256_before": source["source_cache_state_sha256"],
                "source_model_state_sha256_after": source["source_cache_state_sha256"],
            },
            "training_admission": admitted, "training_ids": source["selected_ids"],
            "validation_data_read": False, "ground_truth_scope": "TRAIN8 only",
            "arms": {arm: {"updates": rows} for arm in preflight.ARMS} if admitted else {},
        }

    def test_source_admission_uses_actual_calibration_cpu_api_before_targets(self):
        source = preflight.calibration.prepare_calibration_cpu
        prepared = {
            "selected_ids": list(range(8)), "positions": list(range(8)),
            "source_checkpoint_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "source_cache_manifest_sha256": "c" * 64, "source_cache_bindings_sha256": "d" * 64,
            "source_cache_state_sha256": "e" * 64, "source_cache_trace_sha256": ["f" * 64] * 8,
            "context_cache_manifest_sha256": "1" * 64, "context_cache_bindings_sha256": "2" * 64,
            "context_cache_state_sha256": "3" * 64,
            "context_rows": [{"q_sha256": "4" * 64, "adjacency_sha256": "5" * 64,
                              "context_file_sha256": "6" * 64}] * 8,
            "rgb_sha256": ["7" * 64] * 8, "train_rgb_manifest_sha256": "8" * 64,
            "train_hdf5_path": "registered/train.h5", "train_hdf5_size_bytes": 10,
            "train_hdf5_mtime_ns": 11, "implementation_fingerprint": {"leaf": "9" * 64},
            "validation_data_read": False,
        }
        with mock.patch.object(preflight.calibration, "prepare_calibration_cpu",
                               return_value=prepared) as call:
            _, actual, proof = preflight_queue._source_admission()
        call.assert_called_once_with()
        self.assertIs(actual, prepared)
        self.assertEqual(proof, preflight._source_identity(prepared))
        self.assertNotIn("targets", actual)

    def test_queue_and_preflight_validate_actual_admitted_report_schema(self):
        report = self._report(admitted=True)
        proof = self._source()
        accepted = preflight_queue.validate_report(
            report, source_proof=proof,
            fingerprint=report["queue_fingerprint_sha256"])
        self.assertIs(accepted, report)
        bad = dict(report, source={**proof, "seed": 2})
        with self.assertRaisesRegex(ValueError, "source/cache identity"):
            preflight_queue.validate_report(bad, source_proof=proof,
                                            fingerprint=report["queue_fingerprint_sha256"])

    def test_queue_and_preflight_accept_scientific_rejection_receipt(self):
        report = self._report(admitted=False)
        accepted = preflight_queue.validate_report(
            report, source_proof=report["source"],
            fingerprint=report["queue_fingerprint_sha256"])
        self.assertFalse(accepted["training_admission"])

    def test_owner_idle_callback_uses_registered_four_argument_api(self):
        owner = preflight_queue._load_owner_preserving_visibility()
        signature = inspect.signature(owner._stable_idle)
        self.assertEqual(len(signature.parameters), 4)
        self.assertEqual(list(signature.parameters),
                         ["gpu", "expected_uuid", "stop_event", "on_sample"])


if __name__ == "__main__":
    unittest.main()
