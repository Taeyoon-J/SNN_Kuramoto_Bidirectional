from __future__ import annotations

from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0140_native32_causal_screen import evaluation_queue as queue
from collaborative_test.SW_0140_native32_causal_screen import evaluate


class SW0140EvaluationQueueTests(unittest.TestCase):
    def test_task_plan_has_18_independent_predictions_then_single_score(self):
        tasks = queue.task_plan()
        predicts = [row for row in tasks if row["stage"] == "predict"]
        score = tasks[-1]
        self.assertEqual(len(predicts), 18)
        self.assertEqual(len({(row["seed"], row["condition"]) for row in predicts}), 18)
        self.assertTrue(all(row["depends_on"] == [] for row in predicts))
        self.assertEqual(score["stage"], "score")
        self.assertEqual(set(score["depends_on"]), {row["task_id"] for row in predicts})

    def test_cpu_score_is_cuda_hidden_while_prediction_uses_leased_mapping(self):
        score = {"stage": "score"}
        predict = {"stage": "predict"}
        self.assertEqual(queue.child_environment(score)["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(queue.child_environment(predict, 3)["CUDA_VISIBLE_DEVICES"], "3")
        self.assertTrue(queue.child_environment(score)["OMP_NUM_THREADS"] == "1")

    def test_terminal_completion_and_failure_branches_are_production_queue_paths(self):
        def fake_queue():
            obj = queue.EvaluationQueue.__new__(queue.EvaluationQueue)
            obj.tasks = queue.task_plan()
            obj.guard = threading.RLock()
            obj.state = {"tasks": {row["task_id"]: {"status": "queued"} for row in obj.tasks}}
            obj._unlock = lambda: None
            return obj

        all_pass = fake_queue()
        called = []
        all_pass._execute = lambda task: called.append(task["stage"]) or True
        with mock.patch.object(queue, "_atomic", side_effect=lambda _path, _data: None):
            self.assertTrue(all_pass.run())
        self.assertEqual(called.count("predict"), 18)
        self.assertEqual(called[-1], "score")
        self.assertEqual(all_pass.state["status"], "complete")

        one_failure = fake_queue()
        called = []
        def execute(task):
            called.append(task)
            return not (task["stage"] == "predict" and task["seed"] == 0
                        and task["condition"] == evaluate.CONDITIONS_TO_SCORE[0])
        one_failure._execute = execute
        with mock.patch.object(queue, "_atomic", side_effect=lambda _path, _data: None):
            self.assertFalse(one_failure.run())
        self.assertEqual(sum(row["stage"] == "predict" for row in called), 18)
        self.assertFalse(any(row["stage"] == "score" for row in called))
        self.assertEqual(one_failure.state["tasks"]["score_all"]["status"], "blocked_dependency")

    def test_foreign_gpu_owner_check_is_delegated_to_registered_owner_guard(self):
        pids, tree = [123, 456], {123, 789}
        with mock.patch.object(queue.owner, "foreign_owner_pids", return_value=[456]) as guard:
            self.assertEqual(queue.owner.foreign_owner_pids(pids, tree), [456])
        guard.assert_called_once_with(pids, tree)


if __name__ == "__main__":
    unittest.main()
