"""Focused CPU contract checks for the SW0163 disposable admission receipt."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0163_relation_context_assignment import preflight
from collaborative_test.SW_0163_relation_context_assignment.model import ARMS


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "experiment": "SW0163_relation_context_assignment",
            "context_cache_manifest_sha256": "a" * 64,
            "context_cache_bindings_sha256": "b" * 64,
            "source_trace_manifest_sha256": "c" * 64,
            "source_trace_bindings_sha256": "d" * 64,
            "source_checkpoint_sha256": "e" * 64,
            "source_manifest_sha256": "f" * 64,
            "ordered_image_ids": list(range(512)),
            "context_implementation_fingerprint": {"context_cache.py": "1" * 64},
            "preflight_implementation_fingerprint": {"preflight.py": "2" * 64},
            "source_model_state_sha256": "3" * 64,
            "source_model_unchanged": True,
        }

    def _arm(self, *, hidden=(0.0, 0.2)):
        return {"updates_run": 2, "initial_model_sha256": "9" * 64,
            "updates": [
                {"update": 1, "loss": 2.0, "gradient_norm": 1.0,
                 "row_sum_error": 1e-7, "parameter_delta": 0.02, "finite": True,
                 "gradient_norms": {"binder": 0.5, "relation_output": 0.8,
                                    "relation_hidden": hidden[0]}},
                {"update": 2, "loss": 1.8, "gradient_norm": 0.7,
                 "row_sum_error": 1e-7, "parameter_delta": 0.01, "finite": True,
                 "gradient_norms": {"binder": 0.4, "relation_output": 0.6,
                                    "relation_hidden": hidden[1]}},
            ]}

    def _report(self):
        report = dict(self.expected)
        report.update({"status": "complete", "disposable_updates": 2,
            "ground_truth_used": True,
            "ground_truth_scope": "registered sixteen TRAIN modal32 targets",
            "source_model_unchanged": True,
            "frozen_caches_unchanged": True,
            "context_cache_manifest_sha256_before": self.expected["context_cache_manifest_sha256"],
            "context_cache_manifest_sha256_after": self.expected["context_cache_manifest_sha256"],
            "source_trace_manifest_sha256_before": self.expected["source_trace_manifest_sha256"],
            "source_trace_manifest_sha256_after": self.expected["source_trace_manifest_sha256"],
            "training_admission": True,
            "arms": {name: self._arm() for name in ARMS}})
        return report

    def test_registered_four_arm_receipt_accepts_and_binds_real_producer_schema(self):
        report = self._report()
        self.assertIs(preflight.validate_report_record(report, expected=self.expected), report)

    def test_cache_source_or_arm_drift_cannot_be_admitted(self):
        report = self._report()
        report["context_cache_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "context_cache_manifest_sha256"):
            preflight.validate_report_record(report, expected=self.expected)
        report = self._report()
        report["arms"]["Q_A"]["updates"][1]["gradient_norms"]["relation_hidden"] = 0.0
        report["training_admission"] = True
        with self.assertRaisesRegex(ValueError, "admission disagrees"):
            preflight.validate_report_record(report, expected=self.expected)

    def test_finite_zero_or_missing_gradient_is_preserved_as_nonadmission(self):
        report = self._report()
        report["arms"]["A_only"]["updates"][0]["gradient_norms"]["relation_output"] = 0.0
        report["training_admission"] = False
        self.assertIs(preflight.validate_report_record(report, expected=self.expected), report)
        report["arms"]["A_only"]["updates"][0]["gradient_norm"] = float("nan")
        with self.assertRaisesRegex(ValueError, "malformed"):
            preflight.validate_report_record(report, expected=self.expected)

    def test_gradient_norm_helper_checks_actual_parameter_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        (parameter.square().sum()).backward()
        norm, finite = preflight._norm([parameter])
        self.assertTrue(finite)
        self.assertAlmostEqual(norm, (20.0) ** 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
