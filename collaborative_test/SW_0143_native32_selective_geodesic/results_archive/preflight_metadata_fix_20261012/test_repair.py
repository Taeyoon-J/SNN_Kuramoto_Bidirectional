from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import uuid
from unittest import mock

import repair_queue as repair


def terminal_fixture():
    current = repair.run.implementation_fingerprint()
    prior = dict(current)
    prior[repair.RUN_REL] = repair.OLD_RUN_SHA256
    tasks = {task["task_id"]: {"task": task, "status": "queued"}
             for task in repair.frozen.task_plan()}
    for seed in (0, 1):
        tasks[f"preflight_s{seed}"].update(
            status="failed", error="RuntimeError: recovery cancelled while waiting for a stable exclusive GPU")
    tasks["preflight_s2"].update(status="failed", child_pid=repair.EXPECTED_FAILED_CHILD,
        returncode=1, artifact_valid=False,
        log_path=str(repair.V2_LOG_PATH))
    state = {
        "experiment": "SW0143_native32_selective_geodesic_recovery_v2",
        "status": "failed", "supervisor_pid": repair.EXPECTED_SUPERVISOR,
        "original_state_sha256": repair.EXPECTED_ORIGINAL_STATE_SHA,
        "v1_state_sha256": repair.EXPECTED_V1_STATE_SHA,
        "v1_helper_sha256": repair.EXPECTED_V1_HELPER_SHA,
        "queue_sha256": repair.EXPECTED_V2_QUEUE_SHA,
        "recovery_helper_sha256": repair.V2_HELPER_SHA256,
        "implementation_fingerprint": prior,
        "tasks": tasks,
    }
    raw = (json.dumps(state, indent=2, allow_nan=False) + "\n").encode()
    log = "File \"/server/SW_0143_native32_selective_geodesic/run.py\", line 191, in preflight_seed\nTypeError: 'int' object is not subscriptable\n"
    return raw, state, log


class RepairAdmissionTests(unittest.TestCase):
    def test_accepts_only_observed_failure_and_single_run_hash_delta(self):
        raw, state, log = terminal_fixture()
        result = repair.validate_failed_v2(
            raw, state, log, pid_alive=lambda _pid: False, proc_cmdlines=lambda: [])
        self.assertEqual(result["v2_state_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["run_py_previous_sha256"], repair.OLD_RUN_SHA256)
        self.assertEqual(result["run_py_repaired_sha256"], repair.NEW_RUN_SHA256)
        changed = json.loads(json.dumps(state))
        changed["implementation_fingerprint"]["snn_kuramoto_bidirectional/s2net_cls.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "beyond the run.py repair"):
            repair.validate_failed_v2(raw, changed, log, pid_alive=lambda _pid: False,
                                      proc_cmdlines=lambda: [])

    def test_requires_exact_trace_typeerror_and_dead_owned_processes(self):
        raw, state, log = terminal_fixture()
        with self.assertRaisesRegex(ValueError, "known trace-metadata TypeError"):
            repair.validate_failed_v2(raw, state, "unrelated failure", pid_alive=lambda _pid: False,
                                      proc_cmdlines=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "child still running"):
            repair.validate_failed_v2(raw, state, log,
                pid_alive=lambda pid: pid == repair.EXPECTED_FAILED_CHILD,
                proc_cmdlines=lambda: [])

    def test_requires_old_supervisor_and_all_scientific_argv_absent(self):
        raw, state, log = terminal_fixture()
        with self.assertRaisesRegex(RuntimeError, "supervisor or failed child"):
            repair.validate_failed_v2(raw, state, log,
                pid_alive=lambda pid: pid == repair.EXPECTED_SUPERVISOR,
                proc_cmdlines=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "process still live"):
            repair.validate_failed_v2(raw, state, log, pid_alive=lambda _pid: False,
                proc_cmdlines=lambda: [(998, "/repo/collaborative_test/SW_0143_native32_selective_geodesic/run.py --stage preflight")])

    def test_task_plan_is_full_thirteen_stage_recovery_not_partial_retry(self):
        _, state, _ = terminal_fixture()
        task_ids = {task["task_id"] for task in repair.frozen.task_plan()}
        self.assertEqual(task_ids, set(state["tasks"]))
        self.assertEqual(len(task_ids), 13)
        self.assertEqual(state["tasks"]["preflight_s2"]["status"], "failed")

    def test_repair_queue_constructor_uses_new_create_once_runtime(self):
        temp = Path(tempfile.gettempdir()) / ("sw0143_repair_ctor_" + uuid.uuid4().hex[:8])
        temp.mkdir()
        runtime = temp / "recovery_v3"
        old_lock = temp / "original.lock"
        old_lock.touch()
        class FakeFcntl:
            LOCK_EX = 0x10
            LOCK_NB = 0x20
            LOCK_UN = 0x40
            @staticmethod
            def flock(_fd, _operation):
                return None
        try:
            with mock.patch.object(repair.v2, "ORIGINAL_LOCK", old_lock), \
                 mock.patch.object(repair.v2, "fcntl", FakeFcntl):
                _raw, _state, _log = terminal_fixture()
                fake_lineage = {"v2_state_sha256": "a" * 64,
                                "original_state_sha256": repair.EXPECTED_ORIGINAL_STATE_SHA,
                                "v1_state_sha256": repair.EXPECTED_V1_STATE_SHA}
                plan = {"tasks": repair.frozen.task_plan(), "preserved": [],
                        "missing": [task["task_id"] for task in repair.frozen.task_plan()]}
                queue = repair.RepairQueue(fake_lineage, plan, recovery_root=runtime)
                self.assertEqual(queue.state["experiment"], "SW0143_native32_selective_geodesic_recovery_v3")
                self.assertEqual(queue.state["previous_recovery"], "recovery_v2_20261012")
                self.assertEqual(len(queue.state["tasks"]), 13)
                queue.recovery_lock.close()
                queue.original_lock.close()
            for path in sorted(runtime.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            runtime.rmdir()
        finally:
            old_lock.unlink(missing_ok=True)
            temp.rmdir()


if __name__ == "__main__":
    unittest.main()
