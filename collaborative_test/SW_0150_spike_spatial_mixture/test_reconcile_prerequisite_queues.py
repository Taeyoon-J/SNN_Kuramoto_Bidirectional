"""Guard tests for recreating only untouched SW0150/151 blocked reservations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock
from uuid import uuid4

from collaborative_test.SW_0150_spike_spatial_mixture import reconcile_score_queue as recovery


class PrerequisiteReservationTests(unittest.TestCase):
    def setUp(self):
        self.base = recovery.stage_queue.HERE / f".sw150_requeue_test_{uuid4().hex}"
        self.base.mkdir()
        self.addCleanup(self._cleanup)
        self.patchers = []
        self.main_state = self.base / "main_state.json"
        self.main_receipt_dir = self.base / "main_receipt"
        self.main_receipt_dir.mkdir()
        old_log = self.main_receipt_dir / "original_score_failure.log"
        old_log.write_text(recovery.EXPECTED_FAILURE_TEXT + "\n", encoding="utf-8")
        receipt = {
            "status": "score_only_reconciled",
            "frozen_evaluate_sha256": recovery.EXPECTED_EVALUATE_SHA256,
            "registered_baseline_sha256": recovery.EXPECTED_BASELINE_SHA256,
            "training_or_prediction_relaunched": False,
            "archived_score_log": old_log.name,
            "original_score_log_sha256": recovery.sha256_file(old_log),
        }
        receipt_path = self.main_receipt_dir / "reconciliation_receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.main_state.write_text(json.dumps({
            "status": "complete",
            "score_recovery": {"status": "score_only_reconciled",
                                "receipt_sha256": recovery.sha256_file(receipt_path)},
            "tasks": {"score": {"recovery_receipt": str(receipt_path)}},
        }), encoding="utf-8")
        self._patch(recovery.stage_queue, "STATE", self.main_state)
        if recovery.fcntl is None:
            fake_fcntl = type("FakeFcntl", (), {
                "LOCK_EX": 1, "LOCK_NB": 2, "LOCK_UN": 4,
                "flock": staticmethod(lambda *_args: None),
            })
            self._patch(recovery, "fcntl", fake_fcntl)

    def _patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.patchers.append(patcher)

    def _cleanup(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        root = recovery.stage_queue.HERE.resolve()
        if not self.base.resolve().is_relative_to(root):
            raise RuntimeError("refusing to clean prerequisite fixture outside SW0150")
        for child in sorted(self.base.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        self.base.rmdir()

    def _run_requeue_fixture(self, kind):
        module, blocked_status, reason = recovery._queue_module(kind)
        runtime = self.base / kind
        runtime.mkdir()
        state_path, lock_path = runtime / "state.json", runtime / "queue.lock"
        log_path, output_path = runtime / "attempt.log", runtime / "result.json"
        lock_path.write_bytes(b"queue lock")
        self._patch(module, "STATE", state_path)
        self._patch(module, "LOCK", lock_path)
        self._patch(module, "LOG", log_path)
        self._patch(module, "OUTPUT", output_path)

        if kind == "sw150_diagnosis":
            self._patch(module, "main_queue_state", lambda: ("ready", None))
            task = module.task_plan()[0]
            diag_fp = module.diagnosis.implementation_fingerprint()
            diag_sha = hashlib.sha256(json.dumps(
                diag_fp, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            state = {
                "status": blocked_status, "reason": reason, "supervisor_pid": 7101,
                "diagnostic_fingerprint": diag_fp,
                "diagnostic_fingerprint_sha256": diag_sha,
                "queue_fingerprint": module.implementation_fingerprint(),
                "implementation_sha256": module.implementation_sha256(),
                "tasks": {task["task_id"]: {"task": task, "status": blocked_status}},
            }
        else:
            self._patch(module, "upstream_readiness", lambda: ("ready", None))
            task = module.task_plan()[0]
            queue_fp = module.implementation_fingerprint()
            preflight_fp = module.preflight.implementation_fingerprint()
            state = {
                "status": blocked_status, "reason": reason, "supervisor_pid": 7102,
                "queue_fingerprint": queue_fp,
                "queue_fingerprint_sha256": hashlib.sha256(json.dumps(
                    queue_fp, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode("utf-8")).hexdigest(),
                "preflight_fingerprint": preflight_fp,
                "preflight_fingerprint_sha256": hashlib.sha256(json.dumps(
                    preflight_fp, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode("utf-8")).hexdigest(),
                "task": task, "attempt_policy": "one_shot_no_automatic_retry",
            }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        old_state_bytes = state_path.read_bytes()
        launches = []

        class FakeProcess:
            pid = 7201

        def fake_popen(argv, **kwargs):
            launches.append((argv, kwargs))
            return FakeProcess()

        archive = self.base / f"archive_{kind}"
        receipt = recovery.requeue_blocked_prerequisite(
            kind, archive_dir=archive, process_alive=lambda _pid: False,
            popen=fake_popen, now=1111.0)
        self.assertFalse(state_path.exists())
        self.assertEqual((archive / "blocked_queue_state.json").read_bytes(), old_state_bytes)
        self.assertEqual(receipt["status"], "original_queue_relaunched")
        self.assertEqual(receipt["new_supervisor_pid"], 7201)
        self.assertEqual(len(launches), 1)
        argv, kwargs = launches[0]
        self.assertIn("-m", argv)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["cwd"], module.ROOT)
        self.assertTrue(Path(kwargs["stdout"].name).name.startswith("sup_1111_"))
        self.assertTrue((archive / "relaunch_receipt.json").is_file())

    def test_exact_schema_blocked_diagnosis_state_is_archived_and_queue_relaunched(self):
        self._run_requeue_fixture("sw150_diagnosis")

    def test_exact_schema_blocked_preflight_state_is_archived_and_queue_relaunched(self):
        self._run_requeue_fixture("sw151_preflight")

    def test_unrelated_block_reason_or_existing_artifact_is_never_retried(self):
        module, blocked_status, _reason = recovery._queue_module("sw150_diagnosis")
        runtime = self.base / "unsafe"
        runtime.mkdir()
        state_path, lock_path = runtime / "state.json", runtime / "queue.lock"
        lock_path.write_bytes(b"queue lock")
        output = runtime / "existing_report.json"
        output.write_text("preserve", encoding="utf-8")
        self._patch(module, "STATE", state_path)
        self._patch(module, "LOCK", lock_path)
        self._patch(module, "LOG", runtime / "attempt.log")
        self._patch(module, "OUTPUT", output)
        self._patch(module, "main_queue_state", lambda: ("ready", None))
        task = module.task_plan()[0]
        state_path.write_text(json.dumps({
            "status": blocked_status, "reason": "some other failure", "supervisor_pid": 7301,
            "diagnostic_fingerprint": module.diagnosis.implementation_fingerprint(),
            "diagnostic_fingerprint_sha256": "0" * 64,
            "queue_fingerprint": module.implementation_fingerprint(),
            "implementation_sha256": module.implementation_sha256(),
            "tasks": {task["task_id"]: {"task": task, "status": blocked_status}},
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exact reconciled SW0150 score failure"):
            recovery.requeue_blocked_prerequisite(
                "sw150_diagnosis", archive_dir=self.base / "unsafe_archive",
                process_alive=lambda _pid: False, popen=lambda *_a, **_k: self.fail("must not launch"))
        self.assertTrue(state_path.is_file())
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
