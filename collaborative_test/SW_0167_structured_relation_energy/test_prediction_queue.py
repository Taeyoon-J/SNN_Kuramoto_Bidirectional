"""CPU-only dependency and queue import contracts for SW0167 evaluation."""
from __future__ import annotations

import json
import sys
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0167_structured_relation_energy import (
    predict_evaluate as evaluation, prediction_queue, score_queue, train, training_queue,
)


class EvaluationQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = (Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
                    / ("q167" + uuid.uuid4().hex[:8]))
        self.tmp.mkdir()

    def tearDown(self):
        root = (Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary").resolve()
        target = self.tmp.resolve()
        if not target.is_relative_to(root):
            raise RuntimeError("unsafe fixture cleanup path")
        shutil.rmtree(target)

    def test_training_dependency_requires_both_complete_bound_manifests(self):
        paths = {arm: self.tmp / f"{arm}.json" for arm in train.ARMS}
        queue_fp = {"code": "a" * 64}
        for arm, path in paths.items():
            path.write_text(json.dumps({"status": "complete", "arm": arm,
                "supervisor_pid": 500,
                "manifest_sha256": "b" * 64}), encoding="utf-8")
        state_path = lambda arm: paths[arm]
        artifact = {"checkpoint_sha256": "c" * 64}
        with (mock.patch.object(training_queue, "_state_path", side_effect=state_path),
              mock.patch.object(train, "validate_training_artifact", return_value=artifact),
              mock.patch.object(prediction_queue, "sha256_file", return_value="b" * 64)):
            got = prediction_queue.training_dependency()
        self.assertEqual(set(got["arms"]), set(train.ARMS))
        paths[train.ARMS[1]].write_text(json.dumps({"status": "complete",
            "arm": "wrong", "manifest_sha256": "b" * 64}), encoding="utf-8")
        with (mock.patch.object(training_queue, "_state_path", side_effect=state_path),
              mock.patch.object(train, "validate_training_artifact", return_value=artifact),
              mock.patch.object(prediction_queue, "sha256_file", return_value="b" * 64),
              self.assertRaises(ValueError)):
            prediction_queue.training_dependency()

    def test_nonterminal_training_requires_real_supervisor_liveness(self):
        paths = {arm: self.tmp / f"{arm}.json" for arm in train.ARMS}
        paths[train.ARMS[0]].write_text(json.dumps({"status": "complete",
            "arm": train.ARMS[0], "manifest_sha256": "b" * 64}), encoding="utf-8")
        paths[train.ARMS[1]].write_text(json.dumps({"status": "running_training",
            "supervisor_pid": 777}), encoding="utf-8")
        with (mock.patch.object(training_queue, "_state_path", side_effect=lambda arm: paths[arm]),
              mock.patch.object(train, "validate_training_artifact", return_value={"checkpoint_sha256": "c"*64}),
              mock.patch.object(prediction_queue, "sha256_file", return_value="b"*64)):
            with self.assertRaisesRegex(ValueError, "supervisor is not live"):
                prediction_queue.training_dependency(process_live=lambda pid: False)
            self.assertIsNone(prediction_queue.training_dependency(process_live=lambda pid: pid == 777))

    def test_score_dependency_checks_prediction_fingerprint_and_owner(self):
        path = self.tmp / "pred-state.json"
        pred_fp = {"prediction": "a" * 64}
        valid = {"status": "complete", "queue_fingerprint": pred_fp,
                 "prediction_hashes": {arm: {"prediction_sha256": "b" * 64,
                                            "protocol_sha256": "c" * 64}
                                       for arm in evaluation.ARMS}}
        path.write_text(json.dumps(valid), encoding="utf-8")
        with mock.patch.object(prediction_queue, "implementation_fingerprint", return_value=pred_fp):
            self.assertEqual(score_queue.prediction_dependency(state_path=path)["state"], valid)
            valid["prediction_hashes"].pop(evaluation.ARMS[0])
            path.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong arm set"):
                score_queue.prediction_dependency(state_path=path)
            path.write_text(json.dumps({"status": "running_prediction", "supervisor_pid": 900}), encoding="utf-8")
            with mock.patch.object(prediction_queue, "_proc_live", return_value=True) as live:
                self.assertIsNone(score_queue.prediction_dependency(state_path=path))
                live.assert_called_once_with(900, module="collaborative_test.SW_0167_structured_relation_energy.prediction_queue")
            with mock.patch.object(prediction_queue, "_proc_live", return_value=False):
                with self.assertRaisesRegex(ValueError, "not live"):
                    score_queue.prediction_dependency(state_path=path)

    def test_proc_liveness_matches_exact_stage_module(self):
        pid = 987654
        cmd = b"python\0-m\0collaborative_test.SW_0167_structured_relation_energy.prediction_queue\0--stage\0predict\0"
        with (mock.patch.object(prediction_queue.sys, "platform", "linux"),
              mock.patch.object(Path, "read_bytes", return_value=cmd)):
            self.assertTrue(prediction_queue._proc_live(
                pid, module="collaborative_test.SW_0167_structured_relation_energy.prediction_queue"))
            self.assertFalse(prediction_queue._proc_live(
                pid, module="collaborative_test.SW_0167_structured_relation_energy.training_queue"))

    def test_queue_commands_and_dry_runs_do_not_expose_gpu_in_parent(self):
        self.assertIn("--stage", prediction_queue.command(2))
        self.assertEqual(prediction_queue.command(2)[-2:], ["--device", "cuda:0"])
        self.assertEqual(score_queue.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
