"""Focused CPU tests for the independent SW0150 score recovery contract."""
from __future__ import annotations

import copy
import json
import unittest
from unittest import mock

from collaborative_test.SW_0150_spike_spatial_mixture import score_recovery, test_evaluate


class ScoreRecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.baseline_path = score_recovery.sealed.SLOT_BASELINE
        self.baseline = json.loads(self.baseline_path.read_text(encoding="utf-8-sig"))

    def test_actual_registered_baseline_schema_and_digest_are_accepted(self):
        self.assertEqual(score_recovery.sha256_file(self.baseline_path),
                         score_recovery.sealed.SLOT_BASELINE_SHA256)
        self.assertEqual(self.baseline["evaluation_contract"],
                         score_recovery.BASELINE_CONTRACT)
        means = score_recovery._validate_baseline_contract(self.baseline)
        self.assertEqual(set(means), set(score_recovery.contract.METRICS))
        self.assertTrue(all(value >= 0 for value in means.values()))

    def test_missing_registered_descriptive_contract_field_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        del candidate["evaluation_contract"]["native_prediction"]
        with self.assertRaisesRegex(ValueError, "contract/schema changed"):
            score_recovery._validate_baseline_contract(candidate)

    def test_changed_contract_numeric_value_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["evaluation_contract"]["patch_pixels"] = 8
        with self.assertRaisesRegex(ValueError, "contract/schema changed"):
            score_recovery._validate_baseline_contract(candidate)

    def test_boolean_cannot_impersonate_numeric_contract_value(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["evaluation_contract"]["patch_pixels"] = True
        with self.assertRaisesRegex(ValueError, "contract/schema changed"):
            score_recovery._validate_baseline_contract(candidate)

    def test_unregistered_contract_field_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["evaluation_contract"]["display_hint"] = "anything"
        with self.assertRaisesRegex(ValueError, "contract/schema changed"):
            score_recovery._validate_baseline_contract(candidate)

    def test_nonfinite_registered_mean_is_rejected(self):
        candidate = copy.deepcopy(self.baseline)
        candidate["scores"]["slot32"]["fg_ari"]["three_seed_mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "means are nonfinite"):
            score_recovery._validate_baseline_contract(candidate)

    def test_recovery_refuses_mutated_original_scorer(self):
        self.assertEqual(score_recovery.sha256_file(score_recovery.sealed.HERE / "evaluate.py"),
                         score_recovery.EXPECTED_EVALUATE_SHA256)

    def _score_fixture(self):
        fixture = test_evaluate.EvaluationTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.baseline_value["evaluation_contract"] = copy.deepcopy(
            score_recovery.BASELINE_CONTRACT)
        test_evaluate._write_over(fixture.baseline, fixture.baseline_value)
        patcher = mock.patch.object(
            score_recovery.sealed, "SLOT_BASELINE_SHA256",
            score_recovery.sha256_file(fixture.baseline))
        patcher.start()
        self.addCleanup(patcher.stop)
        return fixture

    def test_recovery_scores_full_fixture_with_corrected_registered_contract_gt_last(self):
        fixture = self._score_fixture()
        output = fixture.base / "score_recovery.json"
        real_file = score_recovery.h5py.File

        def check_predictions_before_target(path, *args, **kwargs):
            for name in (*score_recovery.sealed.ARMS, "source_p"):
                directory = fixture.prediction_root / (
                    "source_p_seed1" if name == "source_p" else f"{name}_seed1")
                self.assertTrue((directory / "predictions.npz").is_file())
                self.assertTrue((directory / "protocol.json").is_file())
            self.assertTrue(fixture.source_qcc_npz.is_file())
            self.assertTrue(fixture.slot_npz.is_file())
            return real_file(path, *args, **kwargs)

        with mock.patch.object(score_recovery.h5py, "File",
                               side_effect=check_predictions_before_target):
            report = score_recovery.score_all(
                output=output, dataset=fixture.dataset,
                prediction_root=fixture.prediction_root,
                training_root=fixture.training_root)
        self.assertTrue(report["ground_truth_opened_after_all_predictions_validated"])
        self.assertEqual(report["scores"]["joint"]["primary"]["fg_ari"]["valid_count"], 320)
        self.assertEqual(report["bootstrap"], {
            "samples": 10000, "rng": "numpy RandomState", "seed": 139,
            "common_image_indices": True})
        self.assertFalse(report["may_promote_seed_expansion"])
        self.assertEqual(report["score_recovery_provenance"]["sealed_original_evaluate_sha256"],
                         score_recovery.EXPECTED_EVALUATE_SHA256)
        self.assertTrue(output.is_file())

    def test_tampered_prediction_is_rejected_before_recovery_opens_hdf5(self):
        fixture = self._score_fixture()
        protocol_path = fixture.prediction_root / "joint_seed1" / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["checkpoint_sha256"] = "0" * 64
        test_evaluate._write_over(protocol_path, protocol)
        with mock.patch.object(score_recovery.h5py, "File",
                               side_effect=AssertionError("GT opened before prediction validation")):
            with self.assertRaisesRegex(ValueError, "training checkpoint lineage"):
                score_recovery.score_all(
                    output=fixture.base / "must_not_exist.json", dataset=fixture.dataset,
                    prediction_root=fixture.prediction_root,
                    training_root=fixture.training_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
