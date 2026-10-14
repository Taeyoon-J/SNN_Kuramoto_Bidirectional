"""CPU lifecycle/admission checks for the SW0163 training reservation."""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
import uuid
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0163_relation_context_assignment import context_queue, train, training_queue


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(training_queue.HERE) / f".training_queue_fixture_{uuid.uuid4().hex}"
        self.root.mkdir()
        self.state = self.root / "state.json"
        self.report = self.root / "preflight.json"
        self.report.write_text('{"training_admission":true}', encoding="utf-8")
        self.queue_fp = {"context_queue.py": "a" * 64}

    def tearDown(self):
        shutil.rmtree(self.root)

    def _write_state(self, status, **fields):
        value = {"status": status, "supervisor_pid": 321, "queue_fingerprint": self.queue_fp}
        value.update(fields)
        self.state.write_text(json.dumps(value), encoding="utf-8")

    def _dependency(self, **kwargs):
        return training_queue.preflight_dependency(state_path=self.state,
            report_path=self.report, **kwargs)

    def test_waits_only_for_live_registered_preflight_supervisor(self):
        self._write_state("preflight_running")
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp):
            self.assertIsNone(self._dependency(process_live=lambda pid, module:
                pid == 321 and module == training_queue.PREFLIGHT_MODULE))
            with self.assertRaisesRegex(ValueError, "not live"):
                self._dependency(process_live=lambda _pid, _module: False)

    def test_rejected_or_terminal_preflight_never_admits_or_waits(self):
        self._write_state("complete_rejected")
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp):
            with self.assertRaisesRegex(ValueError, "terminal"):
                self._dependency(process_live=lambda *_: self.fail("must not poll dead prerequisite"))
        self._write_state("failed_child_or_artifact_validation")
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp):
            with self.assertRaisesRegex(ValueError, "not live"):
                self._dependency(process_live=lambda _pid, _module: False)

    def test_admitted_state_requires_current_receipt_hash_and_cpu_validator(self):
        report_sha = training_queue.sha256_file(self.report)
        self._write_state("complete_admitted", preflight_sha256=report_sha)
        prepared = {"preflight_report_sha256": report_sha, "image_ids": list(range(512))}
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp), \
             mock.patch.object(train, "prepare_training_inputs_cpu", return_value=prepared) as validate:
            result = self._dependency(process_live=lambda *_: False)
        self.assertEqual(result["report_sha256"], report_sha)
        self.assertEqual(result["prepared"], prepared)
        validate.assert_called_once_with(preflight_path=self.report)
        self._write_state("complete_admitted", preflight_sha256="0" * 64)
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp):
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                self._dependency(process_live=lambda *_: False)

    def test_command_and_plan_include_exact_four_arms_without_reducing_exposures(self):
        plan = training_queue.dry_run_plan()
        self.assertEqual(plan["arms"], list(train.ARMS))
        self.assertEqual(plan["max_parallel"], 4)
        for arm in train.ARMS:
            argv = training_queue.arm_command(arm)
            self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0163_relation_context_assignment.train"])
            self.assertIn(arm, argv)
            self.assertIn("--device", argv)
        self.assertEqual(train.COUNT * train.PASSES, 8192)
        self.assertEqual(train.UPDATES, 512)

    def test_wrong_preflight_queue_fingerprint_fails_before_receipt_or_gpu(self):
        self._write_state("complete_admitted", preflight_sha256=training_queue.sha256_file(self.report),
                          queue_fingerprint={"context_queue.py": "0" * 64})
        with mock.patch.object(context_queue, "implementation_fingerprint", return_value=self.queue_fp), \
             mock.patch.object(train, "prepare_training_inputs_cpu") as prepare:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self._dependency(process_live=lambda *_: False)
            prepare.assert_not_called()

    def test_real_queue_state_keeps_arm_map_after_admission_transition(self):
        runtime = self.root / "runtime"
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
                                     flock=lambda *_args: None)
        dependency = {"prepared": {"preflight_report_sha256": "a" * 64},
                      "report_sha256": "a" * 64, "queue_state_sha256": "b" * 64}
        with mock.patch.object(training_queue, "fcntl", fake_fcntl), \
             mock.patch.object(training_queue, "POLL_SECONDS", 0.001), \
             mock.patch.object(training_queue, "preflight_dependency", return_value=dependency):
            queue = training_queue.TrainingQueue(runtime=runtime)
            try:
                self.assertIsNotNone(queue._wait_for_admission())
                queue._update("training_arms_queued", registered_arms=list(train.ARMS),
                              max_parallel=training_queue.MAX_PARALLEL)
                queue._arm_update("raw_only", "waiting_for_idle_gpu")
                saved = json.loads(queue.state_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "arms_waiting_for_idle_gpu")
                self.assertEqual(saved["registered_arms"], list(train.ARMS))
                self.assertEqual(saved["arms"]["raw_only"]["status"], "waiting_for_idle_gpu")
            finally:
                queue._unlock()


if __name__ == "__main__":
    unittest.main()
