from __future__ import annotations

import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0154_native64_resolution_compare import prediction_queue as queue


class _Lock:
    def fileno(self): return 101
    def close(self): pass


class _Output:
    def exists(self): return False
    def open(self, *args, **kwargs): return mock.MagicMock(__enter__=lambda s: s,
                                                          __exit__=lambda *a: False)


class Native64PredictionQueueTests(unittest.TestCase):
    def test_plan_and_command_cover_only_fixed_registered_seeds(self):
        tasks = queue.task_plan()
        self.assertEqual([row["seed"] for row in tasks], [0, 1, 2])
        self.assertTrue(all(row["stage"] == "native64_fixed320_source_prediction" for row in tasks))
        for row in tasks:
            argv = queue.command(row, physical_gpu=2)
            self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0154_native64_resolution_compare.predict_native64"])
            self.assertEqual(argv[argv.index("--seed") + 1], str(row["seed"]))
            self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            queue.command(dict(tasks[0], seed=8), physical_gpu=2)

    def test_failed_prediction_never_invokes_cpu_score(self):
        instance = object.__new__(queue.PredictionQueue)
        instance.stop_event = threading.Event()
        instance.guard = threading.RLock()
        instance.tasks = queue.task_plan()
        instance.state = {"tasks": {row["task_id"]: {"status": "queued"}
                                    for row in instance.tasks}}
        instance.evaluation_complete = False
        instance.max_parallel_gpu_tasks = 1
        instance.lock_stream = _Lock()
        instance._verify_fingerprints = lambda: None
        outcomes = iter((True, False, True))
        instance._execute_seed = lambda task: next(outcomes)
        with mock.patch.object(queue, "fcntl", SimpleNamespace(LOCK_UN=1, flock=mock.Mock())), \
             mock.patch.object(queue, "_atomic_json"), \
             mock.patch.object(queue, "valid_prediction", return_value=True), \
             mock.patch.object(queue, "EVALUATION_OUTPUT", _Output()), \
             mock.patch("collaborative_test.SW_0154_native64_resolution_compare.evaluate_native64.score_all") as score:
            self.assertEqual(instance.run(), 1)
        score.assert_not_called()
        self.assertEqual(instance.state["status"], "predictions_failed")

    def test_dry_run_requires_valid_resource_probe_and_keeps_fixed_plan(self):
        stream = io.StringIO()
        with mock.patch.object(queue.resource_queue, "validate_report", return_value=True), \
             mock.patch.object(queue, "sha256_file", return_value="a" * 64), \
             mock.patch.object(queue, "implementation_fingerprint", return_value={"queue.py": "b" * 64}), \
             mock.patch("sys.stdout", stream):
            self.assertEqual(queue.main(["--dry-run", "--max-parallel-gpu-tasks", "1"]), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["status"], "resource_probe_validated")
        self.assertEqual(report["max_parallel_gpu_tasks"], 1)
        self.assertEqual([row["seed"] for row in report["tasks"]], [0, 1, 2])
        with mock.patch.object(queue.resource_queue, "validate_report", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "resource probe"):
                queue.main(["--dry-run"])


if __name__ == "__main__":
    unittest.main()
