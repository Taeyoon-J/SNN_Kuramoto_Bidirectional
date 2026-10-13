"""CPU tests for score-only recovery of the SW0150 terminal queue state."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from collaborative_test.SW_0150_spike_spatial_mixture import (
    evaluate, reconcile_score_queue as reconciler, score_recovery, stage_queue,
    test_evaluate,
)


class ReconcileScoreQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_evaluate.EvaluationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.base = self.fixture.base
        self.state_path = self.base / "queue_state.json"
        self.lock_path = self.base / "queue.lock"
        self.lock_path.write_bytes(b"original lock")
        self.receipt = self.base / "receipt"
        self.score_path = self.base / "canonical_score.json"
        self.summary_patcher = mock.patch.object(evaluate, "SUMMARY", self.score_path)
        self.summary_patcher.start()
        self.addCleanup(self.summary_patcher.stop)
        self.fixture.baseline_value["evaluation_contract"] = score_recovery.BASELINE_CONTRACT.copy()
        test_evaluate._write_over(self.fixture.baseline, self.fixture.baseline_value)
        self.baseline_patcher = mock.patch.object(
            evaluate, "SLOT_BASELINE_SHA256", score_recovery.sha256_file(self.fixture.baseline))
        self.baseline_patcher.start()
        self.addCleanup(self.baseline_patcher.stop)
        self.expected_baseline_patcher = mock.patch.object(
            reconciler, "EXPECTED_BASELINE_SHA256",
            score_recovery.sha256_file(self.fixture.baseline))
        self.expected_baseline_patcher.start()
        self.addCleanup(self.expected_baseline_patcher.stop)
        self.score_recovery_patcher = mock.patch.object(
            score_recovery.sealed, "SUMMARY", self.score_path)
        self.score_recovery_patcher.start()
        self.addCleanup(self.score_recovery_patcher.stop)
        # Produce a real full-320 report through the recovery scorer and fixture
        # validators, then route the immutable queue score task to that file.
        score_recovery.score_all(output=self.score_path, dataset=self.fixture.dataset,
                                 prediction_root=self.fixture.prediction_root,
                                 training_root=self.fixture.training_root)
        self.tasks = stage_queue.task_plan()
        self.score_task = next(row for row in self.tasks if row["task_id"] == "score")
        self.log_path = self.base / "score.attempt01.log"
        self.log_path.write_text(
            "Traceback (most recent call last):\n"
            "ValueError: registered Slot32 baseline contract/schema changed\n",
            encoding="utf-8")
        self.state = {
            "experiment": stage_queue.NAME,
            "status": "score_failed",
            "supervisor_pid": 800001,
            "owner_dispatcher_sha256": stage_queue.OWNER_GUARD_SHA256,
            "implementation_fingerprint": stage_queue.implementation_fingerprint(),
            "implementation_sha256": stage_queue.implementation_sha256(),
            "tasks": {},
        }
        for index, task in enumerate(self.tasks):
            task_id = task["task_id"]
            row = {"task": task, "status": "passed", "artifact_valid": True,
                   "child_pid": 800010 + index}
            if task_id == "score":
                row.update(status="failed", artifact_valid=False, returncode=1,
                           log=str(self.log_path), error="score child returned nonzero")
            self.state["tasks"][task_id] = row
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
        self.original_state_bytes = self.state_path.read_bytes()
        self.original_log_bytes = self.log_path.read_bytes()
        self.original_valid_result = stage_queue.valid_result
        self._patch_posix_lock()

    def _patch_posix_lock(self):
        if reconciler.fcntl is None:
            fake_fcntl = type("FakeFcntl", (), {
                "LOCK_EX": 1, "LOCK_NB": 2, "LOCK_UN": 4,
                "flock": staticmethod(lambda *_args: None),
            })
            patcher = mock.patch.object(reconciler, "fcntl", fake_fcntl)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _valid_result(self, task):
        if task["stage"] == "score":
            return self.original_valid_result(task)
        return True  # The evaluator fixture replaces expensive model validators.

    def test_score_only_reconciliation_archives_exact_failure_and_marks_complete(self):
        with mock.patch.object(stage_queue, "valid_result", side_effect=self._valid_result):
            result = reconciler.reconcile(
                state_path=self.state_path, lock_path=self.lock_path,
                receipt_dir=self.receipt, process_alive=lambda _pid: False, now=12345.0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["tasks"]["score"]["status"], "passed_recovered")
        self.assertEqual(result["tasks"]["score"]["recovery_report_sha256"],
                         reconciler.sha256_file(self.score_path))
        self.assertFalse(result["score_recovery"]["training_or_prediction_relaunched"])
        self.assertEqual((self.receipt / "original_state.json").read_bytes(),
                         self.original_state_bytes)
        self.assertEqual((self.receipt / "original_score_failure.log").read_bytes(),
                         self.original_log_bytes)
        receipt = json.loads((self.receipt / "reconciliation_receipt.json").read_text())
        self.assertEqual(receipt["original_state_sha256"],
                         reconciler.sha256_bytes(self.original_state_bytes))
        self.assertEqual(receipt["score_report_sha256"], reconciler.sha256_file(self.score_path))

    def test_live_recorded_process_blocks_before_archive_or_state_change(self):
        with mock.patch.object(stage_queue, "valid_result", side_effect=self._valid_result):
            with self.assertRaisesRegex(RuntimeError, "processes remain live"):
                reconciler.reconcile(
                    state_path=self.state_path, lock_path=self.lock_path,
                    receipt_dir=self.receipt, process_alive=lambda pid: pid == 800001)
        self.assertEqual(self.state_path.read_bytes(), self.original_state_bytes)
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
