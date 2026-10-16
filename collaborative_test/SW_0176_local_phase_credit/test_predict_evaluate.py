"""SW0176 fixed320 prediction and scoring contracts (CPU-only)."""
import json
import unittest
from unittest import mock

import numpy as np  # NumPy/OpenMP must initialize before torch on Windows.
import torch

from collaborative_test.SW_0176_local_phase_credit import predict_evaluate as evaluate
from collaborative_test.SW_0176_local_phase_credit import train, training_queue


class PredictionEvaluationContracts(unittest.TestCase):
    def test_cache_binding_uses_actual_nested_train_producer_schema(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([{
            "params": [parameter], "name": "core", "parameter_names": ["core.weight"]
        }], lr=1e-3)
        contract = train._optimizer_group_contract(optimizer)
        proof = {"source_cache_manifest_sha256": "a" * 64,
                 "source_manifest_sha256": "9" * 64,
                 "source_checkpoint_sha256": "b" * 64,
                 "producer_manifest_sha256": "8" * 64,
                 "producer_arrays_sha256": "7" * 64}
        cpu_factory = {"status": "complete_fresh_cpu_factory",
            "source_state_sha256": "c" * 64, "base_encoder_sha256": "d" * 64,
            "initial_model_state_sha256": "e" * 64,
            "optimizer_group_contract": contract}
        cpu_factory["receipt_sha256"] = train._json_sha(cpu_factory)
        admitted = {"prepared": {"control_manifest": {"bindings": {
                "initial_model_state_sha256": "e" * 64}},
                "arrays": {"image_ids": np.asarray([1320, 1321], dtype=np.int64)}},
            "source_proof": proof,
            "control_receipt": {"experiment": train.base.EXPERIMENT,
                "arm": train.base.ARMS[1], "seed": 1, "updates": 64,
                "manifest_sha256": "f" * 64,
                "initial_model_state_sha256": "e" * 64},
            "fresh_cpu_factory_receipt": cpu_factory,
            "preflight_report_sha256": "5" * 64,
            "preflight_state_sha256": "6" * 64,
            "preflight_fingerprint": {"test": "preflight"}}
        with mock.patch.object(train, "_control_bindings"), \
             mock.patch.object(train, "implementation_fingerprint", return_value={"test": "fp"}):
            produced = train._make_bindings(admitted, "e" * 64, "c" * 64,
                                            "d" * 64, {"core": [("core.weight", parameter)]},
                                            optimizer)
        self.assertEqual(produced["source_proof"], proof)
        self.assertNotIn("source_cache_manifest_sha256", produced)
        self.assertEqual(evaluate._source_cache_manifest_sha256(produced), "a" * 64)
        manifest = {"experiment": train.EXPERIMENT, "status": "training_complete",
            "arm": train.ARM, "seed": 1, "completed_updates": 64,
            "validation_data_read": False, "bindings": produced,
            "source_state_sha256_before": "c" * 64,
            "final_model_state_sha256": "2" * 64}
        training_state = {}
        with mock.patch.object(training_queue, "_write_json",
                side_effect=lambda _path, value, **_kwargs: training_state.update(value)):
            training_queue._write_state("complete", manifest_sha256="3" * 64,
                completed_updates=64)
        lineage = evaluate._validate_training_lineage(manifest, training_state, "3" * 64)
        self.assertEqual(lineage["source_state_sha256"], "c" * 64)
        self.assertEqual(lineage["source_cache_manifest_sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "exact complete 64-update"):
            evaluate._validate_training_lineage(manifest, {**training_state,
                "manifest_sha256": "4" * 64}, "3" * 64)
        with self.assertRaisesRegex(ValueError, "nested source-cache proof"):
            evaluate._source_cache_manifest_sha256({
                "source_cache_manifest_sha256": "a" * 64})

    def test_bootstrap_is_fixed_seed_176_and_requires_exact_paired_count(self):
        values = np.linspace(0.0, 1.0, evaluate.COUNT)
        result = evaluate._bootstrap(values, np.zeros_like(values))
        rng = np.random.RandomState(176)
        indices = rng.randint(0, evaluate.COUNT,
                              size=(evaluate.BOOTSTRAP_REPLICATES, evaluate.COUNT))
        expected = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
        self.assertEqual(result["seed"], 176)
        self.assertEqual(result["replicates"], 10_000)
        np.testing.assert_array_equal(result["ci95"], expected)
        with self.assertRaises(ValueError):
            evaluate._bootstrap(values[:-1], np.zeros(evaluate.COUNT - 1))

    def test_production_metric_recomputation_handles_full_native32_batch(self):
        truth = np.zeros((evaluate.COUNT, 32, 32), dtype=np.int32)
        prediction = np.zeros_like(truth)
        truth[:, 4:12, 5:15] = 1
        prediction[:, 4:12, 5:15] = 7
        scores = evaluate._score_metrics({"candidate": prediction}, truth)
        self.assertEqual(set(scores["candidate"]), set(evaluate.contract.METRICS))
        self.assertEqual(len(scores["candidate"]["fg_ari"]["per_image"]), evaluate.COUNT)
        self.assertTrue(np.isfinite(scores["candidate"]["fg_ari"]["mean"]))

    def test_fingerprint_covers_all_new_producer_consumer_files(self):
        fingerprint = evaluate.implementation_fingerprint()
        expected = {
            "collaborative_test/SW_0176_local_phase_credit/predict_evaluate.py",
            "collaborative_test/SW_0176_local_phase_credit/prediction_queue.py",
            "collaborative_test/SW_0176_local_phase_credit/score_queue.py",
            "collaborative_test/SW_0176_local_phase_credit/test_predict_evaluate.py",
            "collaborative_test/SW_0176_local_phase_credit/test_prediction_score_queue.py",
        }
        self.assertTrue(expected.issubset(fingerprint))
        self.assertTrue(all(len(fingerprint[path]) == 64 for path in expected))


if __name__ == "__main__":
    unittest.main()
