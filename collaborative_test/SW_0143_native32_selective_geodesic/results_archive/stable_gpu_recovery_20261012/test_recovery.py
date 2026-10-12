from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from contextlib import ExitStack
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ACMLAB = HERE.parents[1]
REPO = Path(os.environ.get("PATCH_V2_REPO", ACMLAB / "patch_v2_sw")).resolve()
sys.path[:0] = [str(HERE), str(ACMLAB), str(REPO), str(REPO / "collaborative_test")]

import recover_queue as recovery
from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_queue as frozen


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def original_state_fixture():
    tasks = {}
    for task in frozen.task_plan():
        row = {"task": task, "status": "queued"}
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


def retirement_fixture():
    return {
        "status": "child_free_stalled_supervisor_retired", "supervisor_pid": 370093,
        "verified_scientific_children_absent": True, "original_queue_state_preserved": True,
        "interrupted_task": "preflight_s2", "interrupted_child_pid": 378677,
        "foreign_owner_pids": [378607], "foreign_processes_modified": False,
        "scientific_files_modified": False,
    }


def v1_state_fixture(original_state):
    tasks = {}
    for task in frozen.task_plan():
        tasks[task["task_id"]] = {"task": task, "status": "queued"}
    tasks["preflight_s0"].update(status="failed",
        error="RuntimeError: recovery cancelled while waiting for an exclusive GPU")
    tasks["preflight_s2"].update(status="failed",
        error="RuntimeError: recovery cancelled while waiting for an exclusive GPU")
    tasks["preflight_s1"].update(status="interrupted_foreign_gpu_owner",
        child_pid=387322, foreign_owner_pids=[387302], returncode=-15)
    raw = json.dumps(original_state, separators=(",", ":")).encode() + b"\n"
    return {
        "experiment": "SW0143_native32_selective_geodesic_recovery_v1", "status": "failed",
        "supervisor_pid": 381401,
        "original_state_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "implementation_fingerprint": recovery.run.implementation_fingerprint(),
        "tasks": tasks,
    }


class StableRecoveryTests(unittest.TestCase):
    def test_stable_gate_waits_full_sixty_seconds_and_takes_final_sample(self):
        clock = FakeClock()
        samples = []

        def sample():
            samples.append(clock.now)
            return "GPU-A", 2, []

        okay, detail = recovery.stable_idle_window(
            sample, stop_event=threading.Event(), monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertTrue(okay)
        self.assertEqual(detail["stable_seconds"], 60.0)
        self.assertEqual(clock.now, 60.0)
        self.assertEqual(samples[-1], 60.0)
        self.assertGreaterEqual(len(samples), 14)

    def test_late_owner_or_memory_rise_breaks_window_for_release(self):
        for changed in ("owner", "memory"):
            clock = FakeClock()

            def sample():
                if clock.now >= 20:
                    return ("GPU-A", 2, [99]) if changed == "owner" else ("GPU-A", 513, [])
                return "GPU-A", 2, []

            okay, reason = recovery.stable_idle_window(
                sample, stop_event=threading.Event(), monotonic=clock.monotonic, sleep=clock.sleep)
            self.assertFalse(okay)
            self.assertEqual(reason, "owner_or_memory")
            self.assertEqual(clock.now, 20.0)

    def test_uuid_change_releases_window_and_query_error_releases(self):
        clock = FakeClock()

        def changed_uuid():
            return ("GPU-B" if clock.now >= 20 else "GPU-A", 2, [])

        okay, reason = recovery.stable_idle_window(
            changed_uuid, stop_event=threading.Event(), monotonic=clock.monotonic, sleep=clock.sleep)
        self.assertFalse(okay)
        self.assertEqual(reason, "uuid_changed")
        self.assertEqual(clock.now, 20.0)
        failed, reason = recovery.stable_idle_window(
            lambda: (_ for _ in ()).throw(OSError("nvidia-smi transient")),
            stop_event=threading.Event(), monotonic=lambda: 0.0, sleep=lambda _x: None)
        self.assertFalse(failed)
        self.assertEqual(reason, "gpu_query_error")

    def test_final_query_error_releases_first_lease_and_reserves_next_gpu(self):
        clock = FakeClock()
        temp = Path(tempfile.gettempdir()) / ("sw0143_gate_" + uuid.uuid4().hex[:8])
        temp.mkdir()
        unlocks = []
        samples_at_final = 0

        class FakeFcntl:
            LOCK_EX = 0x10
            LOCK_NB = 0x20
            LOCK_UN = 0x40

            @staticmethod
            def flock(fd, operation):
                if operation == FakeFcntl.LOCK_UN:
                    unlocks.append(fd)

        try:
            queue = object.__new__(recovery.StableRecoveryQueue)
            queue.stop_event = threading.Event()
            queue.monotonic, queue.sleep = clock.monotonic, clock.sleep
            queue._update = lambda *_args, **_kwargs: None

            def sample(gpu):
                nonlocal samples_at_final
                if gpu == 0 and clock.now >= 60:
                    samples_at_final += 1
                    if samples_at_final == 2:
                        raise OSError("transient final query")
                return f"GPU-{gpu}", 2, []

            queue.sample = sample
            with mock.patch.object(recovery, "fcntl", FakeFcntl), \
                 mock.patch.object(recovery.frozen, "GPU_LEASES", temp):
                gpu, lease, gpu_uuid = queue._reserve("preflight_s0")
                self.assertEqual(gpu, 1)
                self.assertEqual(gpu_uuid, "GPU-1")
                self.assertEqual(len(unlocks), 1)
                recovery.fcntl.flock(lease.fileno(), FakeFcntl.LOCK_UN)
                lease.close()
        finally:
            for path in temp.iterdir():
                path.unlink()
            temp.rmdir()

    def test_execute_rechecks_gpu_immediately_before_child_launch(self):
        temp = Path(tempfile.gettempdir()) / ("sw0143_exec_" + uuid.uuid4().hex[:8])
        temp.mkdir()
        (temp / "tmp").mkdir()
        leases = []
        for index in range(2):
            lease_path = temp / f"lease{index}"
            lease_path.touch()
            leases.append(lease_path.open("r+"))
        updates = []
        seen = {}
        sample_calls = []
        reserve_calls = []

        class FakeFcntl:
            LOCK_EX = 0x10
            LOCK_NB = 0x20
            LOCK_UN = 0x40

            @staticmethod
            def flock(_fd, _operation):
                return None

        class FinishedProc:
            pid = 123456
            returncode = 0

            def poll(self):
                return 0

        queue = object.__new__(recovery.StableRecoveryQueue)
        queue.stop_event = threading.Event()
        queue.preserved = set()
        queue.log_root = temp
        queue._update = lambda task_id, **values: updates.append((task_id, values))
        def sample(gpu):
            sample_calls.append(gpu)
            if len(sample_calls) == 1:
                raise OSError("transient final query")
            return f"GPU-{gpu}", 2, []

        def reserve(_task):
            index = len(reserve_calls)
            reserve_calls.append(index)
            return index, leases[index], f"GPU-{index}"

        queue.sample = sample
        queue._reserve = reserve
        task = {"task_id": "preflight_s0", "stage": "preflight", "seed": 0, "deps": []}

        def fake_popen(argv, **options):
            seen.update(argv=argv, options=options)
            return FinishedProc()

        try:
            with mock.patch.object(recovery, "fcntl", FakeFcntl), \
                 mock.patch.object(recovery, "Path", side_effect=lambda value: temp / "tmp" if str(value) == "/tmp" else Path(value)), \
                 mock.patch.object(recovery.frozen, "environment", side_effect=lambda _task, physical_gpu, triton_cache: {
                     "CUDA_VISIBLE_DEVICES": str(physical_gpu), "TRITON_CACHE_DIR": str(triton_cache)}), \
                 mock.patch.object(recovery.frozen, "command", return_value=["python", "run.py"]), \
                 mock.patch.object(recovery.frozen, "validate_result", return_value=True), \
                 mock.patch.object(recovery.frozen, "output_path", return_value=temp / "missing-output"), \
                 mock.patch.object(recovery.subprocess, "Popen", side_effect=fake_popen):
                self.assertTrue(queue._execute(task), updates)
            self.assertEqual(seen["options"]["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(sample_calls, [0, 1])
            self.assertEqual(reserve_calls, [0, 1])
            self.assertTrue(any(row.get("status") == "passed" for _, row in updates))
            self.assertTrue(any("child_pid" in row for _, row in updates))
        finally:
            for lease in leases:
                lease.close()
            for path in temp.iterdir():
                if path.is_dir():
                    for child in path.iterdir():
                        if child.is_dir():
                            child.rmdir()
                        else:
                            child.unlink()
                    path.rmdir()
                else:
                    path.unlink()
            temp.rmdir()

    def test_v1_terminal_binding_checks_exact_failed_attempt_and_sha(self):
        original = original_state_fixture()
        original_raw = json.dumps(original, separators=(",", ":")).encode() + b"\n"
        prior = v1_state_fixture(original)
        prior_raw = json.dumps(prior, indent=2).encode() + b"\n"
        lineage = recovery.validate_v1_terminal(
            original_raw, original, retirement_fixture(), prior_raw, prior,
            expected_v1_sha256=__import__("hashlib").sha256(prior_raw).hexdigest(),
            pid_alive=lambda _pid: False, proc_cmdlines=lambda: [])
        self.assertEqual(lineage["v1_state_sha256"],
                         __import__("hashlib").sha256(prior_raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "SHA"):
            recovery.validate_v1_terminal(
                original_raw, original, retirement_fixture(), prior_raw, prior,
                expected_v1_sha256="0" * 64, pid_alive=lambda _pid: False,
                proc_cmdlines=lambda: [])

    def test_plan_preserves_valid_original_outputs_and_has_no_duplicate_tasks(self):
        original = original_state_fixture()
        valid = {"preflight_s0", "reference_R_s0"}
        plan = recovery.v1.build_recovery_plan(
            original, artifact_exists=lambda task: task["task_id"] in valid,
            artifact_valid=lambda task: task["task_id"] in valid)
        self.assertEqual(set(plan["preserved"]), valid)
        self.assertEqual(len(plan["tasks"]), len({task["task_id"] for task in plan["tasks"]}))
        self.assertIn("preflight_s1", plan["missing"])

    def _run_queue(self, fail_task=None):
        temp = Path(tempfile.gettempdir()) / ("sw0143_v2_test_" + uuid.uuid4().hex[:10])
        temp.mkdir()
        lockpath = temp / "original.lock"
        lockpath.touch()
        outdir = temp / "v2"
        passed, calls = set(), []
        mutex = threading.Lock()

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_NB = 2
            LOCK_UN = 3

            @staticmethod
            def flock(_fd, _operation):
                return None

        def fake_execute(instance, task):
            calls.append(task["task_id"])
            ok = task["task_id"] != fail_task
            if ok:
                with mutex:
                    passed.add(task["task_id"])
            else:
                instance.stop_event.set()
            return ok

        def fake_valid(task):
            with mutex:
                return task["task_id"] in passed

        try:
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(recovery, "V2_ROOT", outdir))
                stack.enter_context(mock.patch.object(recovery, "ORIGINAL_LOCK", lockpath))
                stack.enter_context(mock.patch.object(recovery, "fcntl", FakeFcntl))
                stack.enter_context(mock.patch.object(
                    recovery.frozen, "validate_result", side_effect=fake_valid))
                plan = {"tasks": frozen.task_plan(), "preserved": [], "missing": [
                    row["task_id"] for row in frozen.task_plan()]}
                lineage = {"original_state_sha256": "a" * 64, "v1_state_sha256": "b" * 64}
                instance = recovery.StableRecoveryQueue(lineage, plan)
                with mock.patch.object(recovery.StableRecoveryQueue, "_execute", new=fake_execute):
                    code = instance.run()
                state = json.loads((outdir / "recovery_state.json").read_text(encoding="utf-8"))
            return code, state, calls
        finally:
            if outdir.exists():
                for path in sorted(outdir.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                outdir.rmdir()
            lockpath.unlink(missing_ok=True)
            temp.rmdir()

    def test_constructor_and_queue_success(self):
        code, state, calls = self._run_queue()
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(calls), 13)
        self.assertEqual(calls[-1], "score_all")

    def test_queue_failure_cancels_dag_before_scoring(self):
        code, state, calls = self._run_queue("preflight_s1")
        self.assertEqual(code, 1)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("score_all", calls)


if __name__ == "__main__":
    unittest.main()
