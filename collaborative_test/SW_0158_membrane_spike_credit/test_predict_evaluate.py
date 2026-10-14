import unittest
from unittest import mock
import shutil
import uuid
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from collaborative_test.SW_0158_membrane_spike_credit import predict_evaluate as pe
from collaborative_test.SW_0155_supervised_spike_relations import train as sw155_train


class PredictionEvaluationContractTests(unittest.TestCase):
    def scratch(self):
        path = pe.ROOT.parent / f"unit_row_{uuid.uuid4().hex}"
        path.mkdir(parents=False)
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path
    def test_prediction_trace_rejects_wrong_tensor_shape_and_nonfinite_values(self):
        shape = (1, 1024, 1024)
        trace = {
            "theta": torch.zeros((1, 1, 1, 1)).expand(1, 1024, 1024, 4),
            "membrane": torch.zeros(shape),
            "spikes": torch.zeros(shape),
            "component_membrane": torch.zeros((1, 4, 1024, 1024)),
            "component_spikes": torch.zeros((1, 4, 1024, 1024)),
            "component_gates": torch.zeros((1, 4, 1024, 1024)),
        }
        pe._validate_trace(trace, 1)
        trace["component_spikes"] = torch.zeros((1, 3, 1024, 1024))
        with self.assertRaisesRegex(ValueError, "component_spikes"):
            pe._validate_trace(trace, 1)
        trace["component_spikes"] = torch.zeros((1, 4, 1024, 1024))
        trace["spikes"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "spikes"):
            pe._validate_trace(trace, 1)

    def test_paired_bootstrap_is_common_image_paired_and_deterministic(self):
        left = np.linspace(0, 1, pe.COUNT)
        right = left - .02
        first = pe._paired_bootstrap(left, right)
        second = pe._paired_bootstrap(left, right)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], .02)
        self.assertGreater(first["ci95"][0], 0)

    def test_training_artifact_api_rejects_unregistered_arm_before_io(self):
        with self.assertRaisesRegex(ValueError, "unregistered SW0158 arm"):
            pe.validate_training_artifact("unexpected", directory="missing")

    def test_validator_history_and_eta_state_match_real_training_producer(self):
        native = torch.nn.Parameter(torch.tensor([.4, .8]))
        integration = torch.nn.Parameter(torch.tensor([.3, .7]))
        eta = torch.nn.Parameter(torch.tensor([.1, .2, .3, .4]))
        families = {"native_core": [native], "integration_tensors": [integration],
                    "eta": [eta]}
        optimizer = torch.optim.Adam([
            {"name": "native_core", "params": [native], "lr": 3e-5},
            {"name": "integration_tensors", "params": [integration], "lr": 1e-3},
            {"name": "eta", "params": [eta], "lr": 1e-3}])

        class Projection:
            @staticmethod
            def project_integrations_():
                return None

        row = sw155_train.logical_update(optimizer=optimizer, family_map=families,
            compute_micro=lambda micro: (native.square().sum() + integration.square().sum()
                                         + eta.square().sum()) * (micro + 1),
            wrapped=Projection(), micro_count=4)
        self.assertIn("mean_loss", row)
        self.assertNotIn("loss", row)
        self.assertEqual(set(row["family_gradient_norms"]), set(families))
        state = optimizer.state_dict()
        eta_id = state["param_groups"][2]["params"][0]
        self.assertEqual(tuple(state["state"][eta_id]["exp_avg"].shape), (4,))
        self.assertEqual(int(state["state"][eta_id]["step"].item()), 1)

    def test_prediction_fingerprint_resolves_all_sealed_runtime_inputs(self):
        fingerprint = pe.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0158_membrane_spike_credit/train.py", fingerprint)
        self.assertIn("collaborative_test/SW_0143_native32_selective_geodesic/geodesic_operator.py",
                      fingerprint)
        self.assertIn("collaborative_test/SW_0136_native32_transfer/evaluate.py", fingerprint)
        self.assertTrue(all(len(value) == 64 for value in fingerprint.values()))

    def test_prediction_row_atomic_commit_roundtrip_and_corruption_rejection(self):
        labels = np.arange(32 * 32, dtype=np.int64).reshape(32, 32)
        rgb_sha = hashlib.sha256(b"fixed rgb bytes").hexdigest()
        binding_sha = hashlib.sha256(b"prediction bindings").hexdigest()
        work = self.scratch()
        (work / "rows").mkdir()
        row = pe._write_prediction_row(work, 0, int(pe.IMAGE_IDS[0]), labels,
                                       rgb_sha, binding_sha)
        self.assertEqual(row["image_index"], 0)
        loaded = pe._load_prediction_row(work, 0, int(pe.IMAGE_IDS[0]),
                                          rgb_sha, binding_sha)
        np.testing.assert_array_equal(loaded, labels)
        with (work / "rows" / "0000" / "row.npz").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "digest"):
            pe._load_prediction_row(work, 0, int(pe.IMAGE_IDS[0]),
                                    rgb_sha, binding_sha)

    def test_prediction_manifest_dataset_fields_keep_training_and_work_schema_distinct(self):
        training = {"hdf5_path": "/data/train.h5", "hdf5_size": 123,
                    "hdf5_mtime_ns": 456}
        fields = pe._training_dataset_fields(training)
        self.assertEqual(fields, {"training_hdf5_path": "/data/train.h5",
                                  "training_hdf5_size": 123,
                                  "training_hdf5_mtime_ns": 456})
        self.assertNotIn("hdf5_path", fields)
        prediction_work = {"dataset_path": "/data/train.h5", "dataset_size": 123,
                           "dataset_mtime_ns": 456}
        self.assertNotEqual(fields, prediction_work)

    def test_existing_score_adoption_recomputes_complete_report_not_only_status(self):
        expected = {"experiment": pe.train.EXPERIMENT, "status": "complete", "seed_scope": [1],
            "ground_truth_opened_after_all_predictions_validated": True,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "prediction_hashes_before_gt": {"arm": "a" * 64},
            "scores": {"live": {"fg_ari": {"mean": .4, "per_image": [.4] * pe.COUNT}}},
            "means": {"live": {"fg_ari": .4}},
            "promotion_gate": {"registered": False}}
        scratch = self.scratch()
        score = scratch / "score.json"
        score.write_text(json.dumps(expected), encoding="utf-8")
        with mock.patch.object(pe, "score_all", return_value=expected):
            self.assertEqual(pe.validate_score(score), expected)
        changed = dict(expected)
        changed["means"] = {"live": {"fg_ari": .9}}
        score.write_text(json.dumps(changed), encoding="utf-8")
        with mock.patch.object(pe, "score_all", return_value=expected):
            with self.assertRaisesRegex(ValueError, "recomputed"):
                pe.validate_score(score)


if __name__ == "__main__":
    unittest.main()
