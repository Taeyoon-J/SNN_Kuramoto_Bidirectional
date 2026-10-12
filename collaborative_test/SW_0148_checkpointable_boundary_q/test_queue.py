from __future__ import annotations

import threading
import unittest
from pathlib import Path
import uuid
from unittest import mock

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue as queue


class QueueContractTests(unittest.TestCase):
    def test_registered_five_stage_dag_and_cpu_score_command(self):
        tasks = queue.task_plan()
        self.assertEqual([t["stage"] for t in tasks],
                         ["train", "train", "predict", "predict", "score"])
        by_id = {row["task_id"]: row for row in tasks}
        for arm in ("encoder_graph", "joint"):
            self.assertEqual(by_id[f"predict_{arm}"]["depends_on"], [f"train_{arm}"])
        self.assertEqual(set(by_id["score"]["depends_on"]),
                         {"predict_encoder_graph", "predict_joint"})
        self.assertNotIn("--device", queue._argv(by_id["score"]))
        self.assertIn("--device", queue._argv(by_id["train_joint"]))

    def _bare_queue(self):
        instance = object.__new__(queue.Queue)
        instance.tasks = queue.task_plan()
        instance.state = {"tasks": {row["task_id"]: {"status": "queued"}
                                    for row in instance.tasks}}
        instance.state_lock = threading.RLock()
        instance.stop_event = threading.Event()
        instance._save = lambda: None
        instance._update = lambda task_id, **fields: instance.state["tasks"][task_id].update(fields)
        return instance

    def test_score_runs_only_after_both_arm_predictions(self):
        instance = self._bare_queue()
        completed = []

        def gpu(task):
            completed.append(task["task_id"])
            return True

        with mock.patch.object(instance, "_execute_gpu", side_effect=gpu), \
             mock.patch.object(instance, "_execute_cpu", side_effect=lambda task: completed.append("score") or True), \
             mock.patch.object(queue, "valid_result", return_value=True):
            self.assertEqual(instance.run(), 0)
        self.assertCountEqual(completed, ["train_encoder_graph", "train_joint",
                                          "predict_encoder_graph", "predict_joint", "score"])
        self.assertEqual(instance.state["status"], "complete")

    def test_failed_training_branch_blocks_predictions_and_score(self):
        instance = self._bare_queue()
        called = []

        def gpu(task):
            called.append(task["task_id"])
            return task["task_id"] != "train_joint"

        with mock.patch.object(instance, "_execute_gpu", side_effect=gpu), \
             mock.patch.object(instance, "_execute_cpu") as score:
            self.assertEqual(instance.run(), 1)
        self.assertEqual(set(called), {"train_encoder_graph", "train_joint"})
        score.assert_not_called()
        self.assertEqual(instance.state["status"], "training_failed_or_interrupted")

    def test_foreign_retry_requires_safe_boundary_or_absent_prediction(self):
        arm_task = {"task_id": "train_joint", "stage": "train", "arm": "joint"}
        pred_task = {"task_id": "predict_joint", "stage": "predict", "arm": "joint"}
        root = Path(__file__).resolve().parent / f".retry-fixture-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            with mock.patch.object(queue.trainer, "WORK", root):
                self.assertTrue(queue.safe_foreign_retry(arm_task, root / "final"))
                checkpoint_dir = root / "joint_seed1" / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                self.assertTrue(queue.safe_foreign_retry(arm_task, root / "final"))
                (checkpoint_dir / "unrecognized.bin").write_bytes(b"keep")
                self.assertFalse(queue.safe_foreign_retry(arm_task, root / "final"))
                self.assertTrue(queue.safe_foreign_retry(pred_task, root / "prediction"))
                (root / "prediction").mkdir()
                self.assertFalse(queue.safe_foreign_retry(pred_task, root / "prediction"))
        finally:
            for child in sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
