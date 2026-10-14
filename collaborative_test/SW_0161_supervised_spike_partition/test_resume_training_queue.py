"""Regression tests for the separate SW161 cache_validated retry namespace."""
import json
import unittest
from unittest import mock

from collaborative_test.SW_0161_supervised_spike_partition import resume_training_queue as rq


class ResumeTrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.root = rq.HERE / "_cpu_test_resume_training_queue"
        self.root.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_actual_cache_validated_producer_state_is_retryable_only_while_owner_live(self):
        arm = rq.train.ARMS[0]
        state = {"experiment": rq.resume_cache_queue.EXPERIMENT, "status": "cache_validated", "arm": arm,
                 "supervisor_pid": 123, "child_pid": None,
                 "error": "RuntimeError: upstream state status='cache_validated'"}
        path = self.root / "state.json"
        path.write_text(json.dumps(state), encoding="utf-8")
        with mock.patch.object(rq, "_upstream_owner_live", return_value=True):
            with self.assertRaises(rq.UpstreamPending):
                rq._validated_upstream(path)
        with mock.patch.object(rq, "_upstream_owner_live", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "absent or terminal"):
                rq._validated_upstream(path)

    def test_known_original_failure_requires_dead_supervisor_and_empty_work_output(self):
        arm = rq.train.ARMS[0]
        old = {"experiment": rq.EXPERIMENT, "arm": arm, "status": "failed",
               "error": "RuntimeError: cache/preflight status='cache_validated'",
               "supervisor_pid": 321, "child_pid": None}
        path = self.root / "old_state.json"
        path.write_text(json.dumps(old), encoding="utf-8")
        with mock.patch.object(rq, "_proc_live", return_value=False), \
             mock.patch.object(rq.train, "TRAIN_ROOT", self.root / "train"), \
             mock.patch.object(rq.train, "WORK_ROOT", self.root / "work"):
            self.assertEqual(rq.validate_original_failed_attempt(arm, state_path=path)["status"], "failed")
        with mock.patch.object(rq, "_proc_live", return_value=True), \
             mock.patch.object(rq.train, "TRAIN_ROOT", self.root / "train"), \
             mock.patch.object(rq.train, "WORK_ROOT", self.root / "work"):
            with self.assertRaisesRegex(RuntimeError, "still live"):
                rq.validate_original_failed_attempt(arm, state_path=path)

    def test_unrelated_or_partial_original_attempt_is_preserved_and_rejected(self):
        arm = rq.train.ARMS[0]
        path = self.root / "state.json"
        state = {"experiment": rq.EXPERIMENT, "arm": arm, "status": "failed",
                 "error": "RuntimeError: cache_validated", "supervisor_pid": None, "child_pid": 9}
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            rq.validate_original_failed_attempt(arm, state_path=path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), state)
        state["child_pid"] = None; state["error"] = "other code failure"
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            rq.validate_original_failed_attempt(arm, state_path=path)


if __name__ == "__main__":
    unittest.main()
