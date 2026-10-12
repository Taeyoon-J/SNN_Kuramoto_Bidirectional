from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PATCH_V2_REPO", HERE.parents[1] / "patch_v2_sw"))
sys.path[:0] = [str(HERE), str(REPO), str(REPO / "collaborative_test")]

import recover_queue as recovery
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_queue as frozen


def stalled_state():
    tasks = {}
    for task in frozen.task_plan():
        status = "queued"
        row = {"task": task, "status": status}
        if task["task_id"] in {"preflight_s0", "preflight_s1"}:
            row["status"] = "waiting_for_exclusive_gpu"
        if task["task_id"] == "preflight_s2":
            row.update(status="interrupted_foreign_gpu_owner", child_pid=378677,
                       foreign_owner_pids=[378607], returncode=-15)
        tasks[task["task_id"]] = row
    return {
        "experiment": "SW0143_native32_selective_geodesic", "status": "running",
        "supervisor_pid": 370093, "implementation_fingerprint": recovery.run.implementation_fingerprint(),
        "queue_sha256": recovery.run.sha256_file(Path(frozen.__file__)), "tasks": tasks,
    }


def retirement(state_bytes):
    return {
        "status": "child_free_stalled_supervisor_retired", "supervisor_pid": 370093,
        "verified_scientific_children_absent": True, "original_queue_state_preserved": True,
        "interrupted_task": "preflight_s2", "interrupted_child_pid": 378677,
        "foreign_owner_pids": [378607], "foreign_processes_modified": False,
        "scientific_files_modified": False,
    }


class RecoveryContractTests(unittest.TestCase):
    def test_retirement_guard_binds_original_fingerprint_children_and_argv(self):
        state = stalled_state()
        raw = json.dumps(state).encode() + b"\n"
        digest = recovery.validate_retirement(raw, state, retirement(raw),
            pid_alive=lambda _pid: False, proc_cmdlines=lambda: [])
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(RuntimeError, "supervisor"):
            recovery.validate_retirement(raw, state, retirement(raw),
                pid_alive=lambda pid: pid == 370093, proc_cmdlines=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "child"):
            recovery.validate_retirement(raw, state, retirement(raw),
                pid_alive=lambda pid: pid == 378677, proc_cmdlines=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "argv"):
            recovery.validate_retirement(raw, state, retirement(raw),
                pid_alive=lambda _pid: False,
                proc_cmdlines=lambda: [(9988, "/Data0/kevinswk/patch_v2_sw/collaborative_test/"
                                        "SW_0143_native32_selective_geodesic/run.py --stage preflight")])

    def test_retirement_process_scan_ignores_recovery_helper_and_state_copy_whitespace(self):
        state = stalled_state()
        original = json.dumps(state, separators=(",", ":")).encode() + b"\n"
        copied = (json.dumps(state, indent=2) + "\n").encode()
        parsed = recovery._decode_state_copy(original, copied)
        self.assertEqual(parsed, state)
        self.assertFalse(recovery._is_live_original_queue(
            os.getpid(), "/ACMLab/SW0143_recovery_v1_20261012/recover_queue.py"))
        self.assertTrue(recovery._is_live_original_queue(
            555, "/Data0/patch_v2_sw/collaborative_test/SW_0143_native32_selective_geodesic/run.py --stage preflight"))

    def test_plan_preserves_valid_artifacts_and_only_schedules_missing_outputs(self):
        state = stalled_state()
        valid_ids = {"preflight_s0", "reference_R_s0", "predict_C_s0"}
        plan = recovery.build_recovery_plan(
            state, artifact_exists=lambda task: task["task_id"] in valid_ids,
            artifact_valid=lambda task: task["task_id"] in valid_ids)
        self.assertEqual(set(plan["preserved"]), valid_ids)
        self.assertNotIn("preflight_s0", plan["missing"])
        self.assertIn("preflight_s2", plan["missing"])
        self.assertEqual(len(plan["tasks"]), 13)

    def test_plan_refuses_invalid_existing_artifact_and_unregistered_retry(self):
        state = stalled_state()
        with self.assertRaisesRegex(FileExistsError, "invalid"):
            recovery.build_recovery_plan(state, artifact_exists=lambda task: task["task_id"] == "preflight_s0",
                                         artifact_valid=lambda _task: False)
        state["tasks"]["reference_R_s0"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "terminal state"):
            recovery.build_recovery_plan(state, artifact_exists=lambda _task: False,
                                         artifact_valid=lambda _task: False)

    def test_stop_event_aborts_gpu_reservation_before_any_lease_or_gpu_query(self):
        instance = recovery.RecoveryQueue.__new__(recovery.RecoveryQueue)
        instance.stop_event = threading.Event()
        instance.stop_event.set()
        with mock.patch.object(recovery.frozen.owner, "_nvidia_gpu_info") as query:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                instance._reserve("preflight_s1")
            query.assert_not_called()

    def _run_queue(self, fail_task=None):
        temp = Path(tempfile.gettempdir()) / ("sw0143_rec_test_" + uuid.uuid4().hex[:10])
        temp.mkdir()
        original_lock = temp / "original.lock"
        original_lock.touch()
        recovery_root = temp / "recovery"
        passed = set()
        lock = threading.Lock()
        calls = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_NB = 2
            LOCK_UN = 3

            @staticmethod
            def flock(_fd, _operation):
                return None

        def fake_execute(instance, task):
            calls.append(task["task_id"])
            okay = task["task_id"] != fail_task
            if okay:
                with lock:
                    passed.add(task["task_id"])
            else:
                instance.stop_event.set()
            return okay

        def fake_validate(task):
            with lock:
                return task["task_id"] in passed

        try:
            with mock.patch.object(recovery, "RECOVERY_ROOT", recovery_root), \
                 mock.patch.object(recovery, "ORIGINAL_LOCK", original_lock), \
                 mock.patch.object(recovery, "fcntl", FakeFcntl), \
                 mock.patch.object(recovery.frozen, "validate_result", side_effect=fake_validate):
                plan = {"tasks": frozen.task_plan(), "preserved": [], "missing": [
                    row["task_id"] for row in frozen.task_plan()]}
                instance = recovery.RecoveryQueue(stalled_state(), "a" * 64, plan)
                with mock.patch.object(recovery.RecoveryQueue, "_execute", new=fake_execute):
                    code = instance.run()
                state = json.loads((recovery_root / "recovery_state.json").read_text(encoding="utf-8"))
            return code, state, calls
        finally:
            for path in recovery_root.rglob("*") if recovery_root.exists() else []:
                if path.is_file():
                    path.unlink()
            if recovery_root.exists():
                for path in sorted(recovery_root.rglob("*"), reverse=True):
                    if path.is_dir():
                        path.rmdir()
                recovery_root.rmdir()
            original_lock.unlink(missing_ok=True)
            temp.rmdir()

    def test_real_constructor_and_run_complete_dag_without_run_method_shadowing(self):
        code, state, calls = self._run_queue()
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(calls), 13)
        self.assertEqual(calls[-1], "score_all")

    def test_real_run_stops_after_task_failure_and_never_scores(self):
        code, state, calls = self._run_queue("preflight_s2")
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("score_all", calls)


if __name__ == "__main__":
    unittest.main()
