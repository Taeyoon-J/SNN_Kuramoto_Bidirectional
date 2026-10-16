"""SW0176 prediction→score state handoffs and GT-last ordering (CPU-only)."""
import sys
import unittest
from unittest import mock

import numpy as np  # NumPy/OpenMP must initialize before any legacy torch imports.

from collaborative_test.SW_0176_local_phase_credit import (
    predict_evaluate as evaluate, prediction_queue, score_queue, train, training_queue,
)


class FakeOwner:
    def __init__(self, record=None):
        self.record = record

    def _proc_record(self, pid):
        return self.record


class PredictionScoreQueueContracts(unittest.TestCase):
    def test_training_queue_real_state_producer_handoff_complete(self):
        emitted = {}
        def capture(path, value, **_kwargs):
            emitted.update(value)
        manifest_sha = "a" * 64
        with mock.patch.object(training_queue, "_write_json", side_effect=capture):
            training_queue._write_state("complete", manifest_sha256=manifest_sha,
                completed_updates=train.UPDATES)
        state = prediction_queue._validate_training_state(emitted, FakeOwner(),
            manifest={"completed_updates": train.UPDATES}, manifest_sha256=manifest_sha)
        self.assertEqual(state["experiment"], "SW0176_local_phase_training_queue")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["completed_updates"], 64)
        self.assertEqual(state["manifest_sha256"], manifest_sha)

    def test_training_queue_handoff_requires_exact_live_module_or_completed_artifact(self):
        emitted = {}
        with mock.patch.object(training_queue, "_write_json", side_effect=lambda _p, v, **_k: emitted.update(v)):
            training_queue._write_state("running_training", supervisor_pid=12345)
        owner = FakeOwner({"state": "S", "argv": [sys.executable, "-m", prediction_queue.TRAIN_MODULE]})
        self.assertIs(prediction_queue._validate_training_state(emitted, owner), False)
        wrong_owner = FakeOwner({"state": "S", "argv": [sys.executable, "-m", "other.training_queue"]})
        with self.assertRaisesRegex(RuntimeError, "failed/orphaned"):
            prediction_queue._validate_training_state(emitted, wrong_owner)
        complete = {}
        with mock.patch.object(training_queue, "_write_json",
                side_effect=lambda _path, value, **_kwargs: complete.update(value)):
            training_queue._write_state("complete", supervisor_pid=12345,
                manifest_sha256="a" * 64, completed_updates=train.UPDATES)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            prediction_queue._validate_training_state(complete, FakeOwner(),
                manifest={"completed_updates": 64}, manifest_sha256="b" * 64)

    def test_prediction_queue_real_state_producer_handoff_validates_hashes(self):
        emitted = {}
        def capture(_path, value, **_kwargs):
            emitted.update(value)
        expected_prediction = "c" * 64
        expected_protocol = "d" * 64
        with mock.patch.object(prediction_queue, "_atomic_json", side_effect=capture):
            prediction_queue._write_state("complete", output="frozen-output",
                prediction_sha256=expected_prediction, protocol_sha256=expected_protocol,
                ground_truth_used_for_prediction=False)
        consumed = score_queue._validate_prediction_state(emitted, FakeOwner(),
            output_is_dir=True, prediction_sha256=expected_prediction,
            protocol_sha256=expected_protocol)
        self.assertEqual(consumed["experiment"], "SW0176_prediction_queue_v1")
        self.assertEqual(consumed["status"], "complete")
        with self.assertRaisesRegex(ValueError, "artifact hashes"):
            score_queue._validate_prediction_state(emitted, FakeOwner(),
                output_is_dir=True, prediction_sha256="e" * 64,
                protocol_sha256=expected_protocol)

    def test_prediction_queue_active_state_requires_live_exact_module(self):
        emitted = {}
        with mock.patch.object(prediction_queue, "_atomic_json", side_effect=lambda _p, v, **_k: emitted.update(v)):
            prediction_queue._write_state("running_prediction", child_pid=12345)
        owner = FakeOwner({"state": "S", "argv": [sys.executable, "-m", score_queue.PREDICTION_MODULE]})
        self.assertIs(score_queue._validate_prediction_state(emitted, owner), False)
        wrong_owner = FakeOwner({"state": "S", "argv": [sys.executable, "-m", "other.prediction_queue"]})
        with self.assertRaisesRegex(RuntimeError, "failed/orphaned"):
            score_queue._validate_prediction_state(emitted, wrong_owner)

    def test_parent_loads_real_checkpoint_on_cpu_before_lease(self):
        expected_sha = "a" * 64
        prepared = {"manifests": {train.ARM: {"final_model_state_sha256": expected_sha}}}
        class Cuda:
            @staticmethod
            def is_initialized():
                return False
        fake_torch = type("Torch", (), {"cuda": Cuda()})()
        found = object()
        with mock.patch.object(prediction_queue.evaluate, "torch", fake_torch), \
             mock.patch.object(prediction_queue.evaluate, "_load_model",
                               return_value=(found, expected_sha)) as loader:
            result = prediction_queue._validate_completed_model_cpu(prepared)
        self.assertEqual(result, expected_sha)
        loader.assert_called_once_with(train.ARM, prepared, "cpu")

    def test_score_callback_cannot_read_gt_until_predictions_validate(self):
        events = []
        prediction = {"prediction_sha256": "a" * 64, "protocol_sha256": "b" * 64}
        validated = dict(prediction)
        class Evaluator:
            @staticmethod
            def prepare_prediction_inputs_cpu():
                events.append("prepare_cpu")
                return {"inputs": "validated"}

            @staticmethod
            def validate_prediction(arm, *, prepared):
                events.append("validate_prediction")
                assert arm == train.ARM and prepared == {"inputs": "validated"}
                return validated

            @staticmethod
            def score_all(*, prepared):
                events.append("read_gt_and_score")
                assert prepared == {"inputs": "validated"}
                return {"gate": {"co_primary_gate_passed": False}}

        def mark_scoring(row):
            events.append("mark_gt_last_scoring")
            assert row == validated
        report, row = score_queue._score_gt_last(prediction, evaluator=Evaluator,
            arm=train.ARM, mark_scoring=mark_scoring)
        self.assertEqual(row, validated)
        self.assertEqual(report["gate"]["co_primary_gate_passed"], False)
        self.assertEqual(events, ["prepare_cpu", "validate_prediction",
                                  "mark_gt_last_scoring", "read_gt_and_score"])

        class MutatedEvaluator(Evaluator):
            @staticmethod
            def validate_prediction(arm, *, prepared):
                events.append("validate_mutated")
                return {"prediction_sha256": "f" * 64, "protocol_sha256": "b" * 64}

        events.clear()
        with self.assertRaisesRegex(ValueError, "changed after terminal"):
            score_queue._score_gt_last(prediction, evaluator=MutatedEvaluator,
                arm=train.ARM, mark_scoring=mark_scoring)
        self.assertNotIn("mark_gt_last_scoring", events)
        self.assertNotIn("read_gt_and_score", events)


if __name__ == "__main__":
    unittest.main()
