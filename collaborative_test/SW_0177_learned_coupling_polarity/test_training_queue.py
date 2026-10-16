"""CPU dependency and process-identity tests for SW0177 training queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

import numpy as np

from collaborative_test.SW_0177_learned_coupling_polarity import training_queue as queue


class _StopAfterOneWait:
    def __init__(self, callback):
        self.callback, self.waits = callback, 0
    def is_set(self): return False
    def wait(self, _seconds):
        self.waits += 1
        if self.waits == 1: self.callback()
        return False


class TrainingQueueTests(unittest.TestCase):
    def test_worker_command_targets_new_trainer_and_arm(self):
        fake = SimpleNamespace(ARMS=("signed", "nonnegative"))
        with mock.patch.object(queue, "_load", return_value=fake):
            command = queue.worker_command("signed", 2)
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        self.assertEqual(command[command.index("--arm") + 1], "signed")
        self.assertEqual(command[command.index("--physical-gpu") + 1], "2")

    def test_parent_waits_for_live_exact_v2_module_then_returns_only_success_state(self):
        central = queue.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        root = (central / f"sw177_wait_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        self.assertEqual(root.parent, central.resolve())
        root.mkdir()
        state_path, report_path = root / "state.json", root / "report.json"
        state_path.write_text(json.dumps({"status": "running_worker", "supervisor_pid": 7654}), encoding="utf-8")
        report_path.write_text("{}", encoding="utf-8")
        queue_obj = SimpleNamespace(STATE=state_path, REPORT=report_path)
        owner = SimpleNamespace(_proc_record=mock.Mock(return_value={"state": "S", "argv":
            ["python", "-u", "-m", queue.PREFLIGHT_QUEUE]}))
        def finish():
            state_path.write_text(json.dumps({"status": "complete", "supervisor_pid": 7654,
                                              "physical_gpu": 1}), encoding="utf-8")
        stop = queue._stop
        try:
            queue._stop = _StopAfterOneWait(finish)
            with mock.patch.object(queue, "_write_state"):
                result = queue._wait_for_successful_preflight(queue_obj, owner,
                    deadline=queue.time.monotonic() + 30, arm="signed")
            self.assertEqual(result["status"], "complete")
            owner._proc_record.assert_called_once_with(7654)
        finally:
            queue._stop = stop
            if root.resolve().parent != central.resolve():
                raise AssertionError("refusing to clean queue fixture outside central temporary")
            for path in root.iterdir(): path.unlink()
            root.rmdir()

    def test_pending_preflight_wrong_or_missing_owner_is_not_accepted(self):
        state = {"status": "waiting_for_exclusive_gpu", "supervisor_pid": 7654}
        wrong = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv":
            ["python", "-m", "collaborative_test.some_other_queue"]})
        with self.assertRaisesRegex(RuntimeError, "wrong module"):
            queue._preflight_parent_is_live(wrong, state)
        gone = SimpleNamespace(_proc_record=lambda _pid: None)
        with self.assertRaisesRegex(RuntimeError, "not live"):
            queue._preflight_parent_is_live(gone, state)

    def test_pending_preflight_wait_is_not_charged_against_worker_timeout(self):
        central = queue.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        root = (central / f"sw177_unbounded_wait_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        self.assertEqual(root.parent, central.resolve())
        root.mkdir()
        state_path, report_path = root / "state.json", root / "report.json"
        state_path.write_text(json.dumps({"status": "running_worker", "supervisor_pid": 7654}), encoding="utf-8")
        report_path.write_text("{}", encoding="utf-8")
        queue_obj = SimpleNamespace(STATE=state_path, REPORT=report_path)
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv":
            ["python", "-u", "-m", queue.PREFLIGHT_QUEUE]})
        def finish():
            state_path.write_text(json.dumps({"status": "complete", "supervisor_pid": 7654}), encoding="utf-8")
        stop = queue._stop
        try:
            queue._stop = _StopAfterOneWait(finish)
            with mock.patch.object(queue, "_write_state"), \
                 mock.patch.object(queue.time, "monotonic", return_value=10**9):
                state = queue._wait_for_successful_preflight(queue_obj, owner,
                    deadline=None, arm="signed")
            self.assertEqual(state["status"], "complete")
        finally:
            queue._stop = stop
            if root.resolve().parent != central.resolve():
                raise AssertionError("refusing to clean queue fixture outside central temporary")
            for path in root.iterdir(): path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
