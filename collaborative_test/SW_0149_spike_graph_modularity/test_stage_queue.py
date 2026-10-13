from __future__ import annotations

import json
from pathlib import Path
from tempfile import gettempdir
import unittest
from unittest import mock
from uuid import uuid4

from collaborative_test.SW_0149_spike_graph_modularity import stage_queue as queue


class StageQueueContractTests(unittest.TestCase):
    def test_fixed_train_predict_score_dag(self):
        rows = queue.task_plan()
        by_id = {row["task_id"]: row for row in rows}
        self.assertEqual(len(rows), 5)
        self.assertEqual(by_id["train_encoder_graph"]["depends_on"], [])
        self.assertEqual(by_id["train_joint"]["depends_on"], [])
        self.assertEqual(by_id["predict_encoder_graph"]["depends_on"], ["train_encoder_graph"])
        self.assertEqual(by_id["predict_joint"]["depends_on"], ["train_joint"])
        self.assertEqual(by_id["score"]["depends_on"], ["predict_encoder_graph", "predict_joint"])

    def test_argv_routing_uses_registered_entrypoints(self):
        rows = {row["stage"]: row for row in queue.task_plan() if row["stage"] != "train"}
        train_row = next(row for row in queue.task_plan() if row["stage"] == "train")
        self.assertIn("SW_0149_spike_graph_modularity.train", queue._argv(train_row)[2])
        self.assertIn("--stage", queue._argv(rows["predict"]))
        self.assertIn("score", queue._argv(rows["score"]))

    def test_safe_foreign_retry_only_resumes_or_restarts_absent_outputs(self):
        row = next(row for row in queue.task_plan() if row["stage"] == "predict")
        row["output"] = str(Path(gettempdir()) / f"sw149-absent-{uuid4().hex}")
        self.assertTrue(queue._safe_foreign_retry(row))
        Path(row["output"]).mkdir()
        try:
            self.assertFalse(queue._safe_foreign_retry(row))
        finally:
            Path(row["output"]).rmdir()

    def test_score_validity_requires_ground_truth_only_for_scoring(self):
        output = Path(gettempdir()) / f"sw149-score-valid-{uuid4().hex}.json"
        row = {"status": "complete", "experiment": queue.train.NAME,
               "seed": queue.train.SEED,
               "ground_truth_opened_after_all_predictions_validated": True,
               "ground_truth_used_for_prediction": False,
               "ground_truth_used_for_scoring": True}
        try:
            output.write_text(json.dumps(row), encoding="utf-8")
            task = {"stage": "score", "output": str(output)}
            self.assertTrue(queue.valid_result(task))
            row["ground_truth_used_for_scoring"] = False
            output.write_text(json.dumps(row), encoding="utf-8")
            self.assertFalse(queue.valid_result(task))
        finally:
            output.unlink(missing_ok=True)

    def test_run_orders_training_then_prediction_then_cpu_score(self):
        obj = queue.StageQueue.__new__(queue.StageQueue)
        obj.tasks = queue.task_plan()
        obj.state = {"tasks": {row["task_id"]: {"status": "queued"}
                               for row in obj.tasks}, "status": "running"}
        obj.state_lock = __import__("threading").RLock()
        obj.stop_event = __import__("threading").Event()
        obj.lock_stream = None
        order = []

        def fake_gpu(task):
            order.append(task["task_id"])
            return True

        def fake_cpu(task):
            order.append(task["task_id"])
            return True

        obj._execute_gpu = fake_gpu
        obj._execute_cpu = fake_cpu
        obj._save = lambda: None
        with mock.patch.object(queue, "valid_result", return_value=True):
            self.assertEqual(obj.run(), 0)
        self.assertEqual(order, ["train_encoder_graph", "train_joint",
                                 "predict_encoder_graph", "predict_joint", "score"])
        self.assertEqual(obj.state["status"], "complete")

    def test_failed_training_blocks_prediction_and_score(self):
        obj = queue.StageQueue.__new__(queue.StageQueue)
        obj.tasks = queue.task_plan()
        obj.state = {"tasks": {row["task_id"]: {"status": "queued"}
                               for row in obj.tasks}, "status": "running"}
        obj.state_lock = __import__("threading").RLock()
        obj.stop_event = __import__("threading").Event()
        obj.lock_stream = None
        calls = []

        def fake_gpu(task):
            calls.append(task["task_id"])
            return task["stage"] != "train"

        obj._execute_gpu = fake_gpu
        obj._execute_cpu = lambda task: calls.append(task["task_id"]) or True
        obj._save = lambda: None
        self.assertEqual(obj.run(), 1)
        self.assertEqual(calls, ["train_encoder_graph", "train_joint"])
        self.assertEqual(obj.state["status"], "train_failed")

    def test_cpu_score_child_is_cuda_hidden_and_does_not_inherit_gpu_lease(self):
        obj = queue.StageQueue.__new__(queue.StageQueue)
        obj.state = {"tasks": {"score": {"status": "queued"}}}
        obj.state_lock = __import__("threading").RLock()
        obj._save = lambda: None
        root = Path(gettempdir()) / f"sw149-queue-cpu-{uuid4().hex}"
        root.mkdir()
        old_runtime = queue.RUNTIME
        try:
            queue.RUNTIME = root
            task = {"task_id": "score", "stage": "score", "output": str(root / "score.json")}
            fake_proc = mock.Mock()
            fake_proc.pid = 123
            fake_proc.wait.return_value = 0
            fake_popen_args = {}

            def popen(argv, **kwargs):
                fake_popen_args.update(kwargs)
                return fake_proc

            with (mock.patch.object(queue, "valid_result", side_effect=[False, True]),
                  mock.patch.object(queue.subprocess, "Popen", side_effect=popen)):
                self.assertTrue(obj._execute_cpu(task))
            self.assertEqual(fake_popen_args["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertNotIn("pass_fds", fake_popen_args)
        finally:
            queue.RUNTIME = old_runtime
            for child in root.iterdir():
                child.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
