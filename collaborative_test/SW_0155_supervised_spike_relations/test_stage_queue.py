from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0155_supervised_spike_relations import stage_queue as queue


class SupervisedStageQueueTests(unittest.TestCase):
    def test_plan_routes_only_the_two_fixed_seed_supervised_predictions(self):
        rows = queue.task_plan()
        self.assertEqual([row["arm"] for row in rows], ["encoder_graph", "joint"])
        self.assertEqual([row["seed"] for row in rows], [1, 1])
        for row in rows:
            argv = queue.command(row, physical_gpu=2)
            self.assertIn("predict_evaluate", argv[2])
            self.assertIn("--stage", argv)
            self.assertEqual(argv[argv.index("--arm") + 1], row["arm"])
        with self.assertRaises(ValueError):
            queue.command({**rows[0], "arm": "source97"}, physical_gpu=2)

    def test_executor_releases_unstable_lease_and_reacquires_before_real_child(self):
        task = queue.task_plan()[0]
        instance = object.__new__(queue.PredictionQueue)
        instance.stop_event = queue.threading.Event()
        instance.guard = queue.threading.RLock()
        instance.state = {"status": "queued", "tasks": {task["task_id"]: {"task": task}}}
        instance._verify = mock.Mock()
        leases = [mock.Mock(), mock.Mock()]
        for index, lease in enumerate(leases):
            lease.fileno.return_value = 81 + index
        child = mock.Mock()
        child.pid, child.poll.return_value = 8811, 0
        child.wait.return_value = 0
        fake_fcntl = SimpleNamespace(LOCK_UN=8, flock=mock.Mock())
        runtime = queue.HERE / "results_archive" / f"queue_executor_test_{os.getpid()}"
        if runtime.exists():
            self.fail("preserve prior executor fixture runtime")
        runtime.mkdir(parents=True)
        try:
            with mock.patch.object(queue, "RUNTIME", runtime), \
                 mock.patch.object(queue, "STATE", runtime / "state.json"), \
                 mock.patch.object(queue, "fcntl", fake_fcntl), \
                 mock.patch.object(queue, "valid_training", return_value=True), \
                 mock.patch.object(queue, "valid_prediction", side_effect=[False, False, True]), \
                 mock.patch.object(queue.owner, "_acquire", side_effect=[
                     (0, "uuid-a", 10, leases[0]), (2, "uuid-c", 12, leases[1])]) as acquire, \
                 mock.patch.object(queue.owner, "_stable_idle", side_effect=[False, True]), \
                 mock.patch.object(queue.subprocess, "Popen", return_value=child) as popen:
                self.assertTrue(instance._execute(task))
            self.assertEqual(acquire.call_count, 2)
            self.assertEqual(popen.call_count, 1)
            argv = popen.call_args.args[0]
            self.assertIn("--arm", argv)
            self.assertEqual(argv[argv.index("--arm") + 1], "encoder_graph")
            self.assertEqual([lease.close.call_count for lease in leases], [1, 1])
            self.assertEqual(fake_fcntl.flock.call_count, 2)
        finally:
            for path in runtime.iterdir():
                path.unlink()
            runtime.rmdir()

    def test_failed_training_artifact_blocks_before_gpu_acquire(self):
        task = queue.task_plan()[0]
        instance = object.__new__(queue.PredictionQueue)
        instance.stop_event = queue.threading.Event()
        instance.guard = queue.threading.RLock()
        instance.state = {"status": "queued", "tasks": {task["task_id"]: {"task": task}}}
        with mock.patch.object(queue, "STATE", queue.HERE / "results_archive" / f"failed_admission_test_{os.getpid()}.json"), \
             mock.patch.object(queue, "_atomic_json"), \
             mock.patch.object(queue, "valid_prediction", return_value=False), \
             mock.patch.object(queue, "valid_training", return_value=False), \
             mock.patch.object(queue.owner, "_acquire") as acquire:
            self.assertFalse(instance._execute(task))
        acquire.assert_not_called()
        self.assertEqual(instance.state["tasks"][task["task_id"]]["status"],
                         "failed_training_artifact_invalid")

    def test_finished_evaluation_validator_requires_both_supervised_arms_and_gt_last(self):
        record = {"status": "complete", "experiment": queue.EXPERIMENT,
            "seed_scope": [1], "ground_truth_opened_after_all_predictions_validated": True,
            "ground_truth_used_for_prediction": False, "ground_truth_used_for_scoring": True,
            "leaderboard": "supervised SW0155 seed1 pilot; separate from unsupervised leaderboard",
            "implementation_fingerprint": queue.evaluation.implementation_fingerprint(),
            "scores": {"encoder_graph": {}, "joint": {}, "source97": {}, "SW0143_T": {}}}
        path = queue.HERE / "results_archive" / f"eval_validator_test_{os.getpid()}.json"
        if path.exists():
            self.fail("preserve pre-existing evaluation validator fixture")
        try:
            with mock.patch.object(queue, "EVALUATION_OUTPUT", path):
                path.write_text(json.dumps(record), encoding="utf-8")
                self.assertTrue(queue.valid_evaluation())
                record["scores"].pop("joint")
                path.write_text(json.dumps(record), encoding="utf-8")
                self.assertFalse(queue.valid_evaluation())
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
