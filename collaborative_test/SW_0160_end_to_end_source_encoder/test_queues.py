from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
from unittest import mock

import numpy

from collaborative_test.SW_0160_end_to_end_source_encoder import (
    model, prediction_queue, score_queue, train, training_queue,
)


class SW0160QueueTests(unittest.TestCase):
    def test_commands_and_runtime_paths_are_arm_specific(self):
        first, second = model.ARMS
        train_a, train_b = (training_queue.configure_arm(arm) for arm in model.ARMS)
        pred_a, pred_b = (prediction_queue.configure_arm(arm) for arm in model.ARMS)
        self.assertNotEqual(train_a["runtime"], train_b["runtime"])
        self.assertNotEqual(train_a["output"], train_b["output"])
        self.assertNotEqual(pred_a["runtime"], pred_b["runtime"])
        self.assertNotEqual(pred_a["output"], pred_b["output"])
        train_argv = training_queue.command(first)
        self.assertIn("collaborative_test.SW_0160_end_to_end_source_encoder.train", train_argv)
        self.assertEqual(train_argv[train_argv.index("--arm") + 1], first)
        pred_argv = prediction_queue.command(second)
        self.assertIn("collaborative_test.SW_0160_end_to_end_source_encoder.predict_evaluate", pred_argv)
        self.assertEqual(pred_argv[pred_argv.index("--arm") + 1], second)
        self.assertEqual(pred_argv[pred_argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            training_queue.configure_arm("unregistered")

    def test_queue_fingerprints_resolve_owner_and_experiment_code(self):
        for fingerprint in (training_queue.implementation_fingerprint,
                            prediction_queue.implementation_fingerprint,
                            score_queue.implementation_fingerprint):
            entries = fingerprint()
            self.assertTrue(entries)
            self.assertTrue(all(len(value) == 64 for value in entries.values()))
        self.assertIn("collaborative_test/SW_0160_end_to_end_source_encoder/train.py",
                      training_queue.implementation_fingerprint())
        self.assertIn("collaborative_test/SW_0160_end_to_end_source_encoder/predict_evaluate.py",
                      score_queue.implementation_fingerprint())

    def test_importing_queue_libraries_preserves_assigned_child_visibility(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(train.ROOT)
        env["CUDA_VISIBLE_DEVICES"] = "3"
        code = ("import os; "
            "import collaborative_test.SW_0160_end_to_end_source_encoder.training_queue; "
            "import collaborative_test.SW_0160_end_to_end_source_encoder.prediction_queue; "
            "import collaborative_test.SW_0160_end_to_end_source_encoder.score_queue; "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES'))")
        result = subprocess.run([sys.executable, "-c", code], cwd=train.ROOT, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3")

    def test_score_queue_waits_for_each_registered_prediction_owner(self):
        root = score_queue.ROOT / f".sw160-queue-test-{uuid.uuid4().hex}"
        arm = model.ARMS[0]
        arm_dir = root / arm
        arm_dir.mkdir(parents=True)
        path = arm_dir / "state.json"
        state = {"experiment": train.EXPERIMENT, "arm": arm, "status": "running",
                 "supervisor_pid": 123, "child_pid": 456}
        path.write_text(json.dumps(state), encoding="utf-8")
        try:
            with mock.patch.object(score_queue, "PREDICTION_QUEUE_ROOT", root), \
                    mock.patch.object(score_queue, "_process_has_args",
                                      side_effect=lambda pid, module, name: pid == 456 and name == arm):
                self.assertTrue(score_queue._prediction_queue_is_live(arm))
                self.assertFalse(score_queue._prediction_queue_is_live(model.ARMS[1]))
                state["status"] = "failed"
                path.write_text(json.dumps(state), encoding="utf-8")
                self.assertFalse(score_queue._prediction_queue_is_live(arm))
        finally:
            path.unlink()
            arm_dir.rmdir()
            root.rmdir()

    def test_preflight_and_score_dependencies_are_not_confused(self):
        state_file = train.ROOT / f".sw160-pref-test-{uuid.uuid4().hex}.json"
        state_file.write_text(json.dumps({"experiment": train.EXPERIMENT,
            "status": "complete_rejected"}), encoding="utf-8")
        try:
            self.assertTrue(training_queue._preflight_terminal_without_report(state_file))
            state_file.write_text(json.dumps({"experiment": train.EXPERIMENT,
                "status": "waiting_for_gpu"}), encoding="utf-8")
            self.assertFalse(training_queue._preflight_terminal_without_report(state_file))
        finally:
            state_file.unlink()

    def test_missing_preflight_report_waits_only_for_live_registered_supervisor(self):
        state_file = train.ROOT / f".sw160-pref-owner-{uuid.uuid4().hex}.json"
        state_file.write_text(json.dumps({"experiment": train.EXPERIMENT,
            "status": "waiting_for_upstream", "supervisor_pid": 789}), encoding="utf-8")
        try:
            with mock.patch.object(training_queue, "_registered_process_live",
                    side_effect=lambda pid, module: pid == 789 and module.endswith("preflight_queue")):
                self.assertTrue(training_queue._preflight_queue_is_live(state_file))
            with mock.patch.object(training_queue, "_registered_process_live", return_value=False):
                self.assertFalse(training_queue._preflight_queue_is_live(state_file))
        finally:
            state_file.unlink()

    def test_prediction_waits_for_training_queue_while_it_waits_for_sw0159_score(self):
        root = prediction_queue.ROOT / f".sw160-training-live-{uuid.uuid4().hex}"
        arm = model.ARMS[0]
        folder = root / arm
        folder.mkdir(parents=True)
        path = folder / "state.json"
        path.write_text(json.dumps({"experiment": train.EXPERIMENT, "arm": arm,
            "status": "waiting_for_sw0159_terminal_score", "supervisor_pid": 321,
            "child_pid": None}), encoding="utf-8")
        try:
            with mock.patch.object(prediction_queue, "TRAINING_QUEUE_ROOT", root), \
                    mock.patch.object(prediction_queue, "_process_has_args",
                        side_effect=lambda pid, module, name:
                            pid == 321 and name == arm and module.endswith("training_queue")):
                self.assertTrue(prediction_queue._training_queue_is_live(arm))
                self.assertFalse(prediction_queue._training_queue_is_live(model.ARMS[1]))
        finally:
            path.unlink()
            folder.rmdir()
            root.rmdir()

    def test_sw159_score_wait_observes_prediction_owner_after_training_terminal(self):
        root = train.ROOT / f".sw160-upstream-live-{uuid.uuid4().hex}"
        training = root / "results_archive" / "training_queues_20261014" / "control"
        prediction = root / "results_archive" / "prediction_queues_20261014" / "control"
        training.mkdir(parents=True)
        prediction.mkdir(parents=True)
        training_state = training / "state.json"
        prediction_state = prediction / "state.json"
        training_state.write_text(json.dumps({"status": "complete", "supervisor_pid": 11}),
                                  encoding="utf-8")
        prediction_state.write_text(json.dumps({"status": "running", "child_pid": 123}),
                                    encoding="utf-8")
        try:
            with mock.patch.object(training_queue.upstream_score_queue, "HERE", root), \
                    mock.patch.object(training_queue.upstream_score_queue, "STATE", root / "missing-score-state.json"), \
                    mock.patch.object(training_queue, "_registered_process_live",
                        side_effect=lambda pid, module: pid == 123 and module.endswith("predict_evaluate")):
                self.assertTrue(training_queue._sw159_score_pipeline_live())
                prediction_state.write_text(json.dumps({"status": "failed", "child_pid": 123}),
                                            encoding="utf-8")
                self.assertFalse(training_queue._sw159_score_pipeline_live())
        finally:
            training_state.unlink(); prediction_state.unlink()
            training.rmdir(); prediction.rmdir()
            (training.parent).rmdir(); (prediction.parent).rmdir()
            (root / "results_archive").rmdir(); root.rmdir()


if __name__ == "__main__":
    unittest.main()
