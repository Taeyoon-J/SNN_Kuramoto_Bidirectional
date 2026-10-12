from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recover_train_core as recovery

def fixture_dir():
    root = HERE.parents[1].resolve()
    path = root / ("sw142_" + uuid.uuid4().hex[:8])
    path.mkdir()
    return path

def cleanup_fixture(path):
    root = HERE.parents[1].resolve()
    path = Path(path).resolve()
    if path.parent != root or not path.name.startswith("sw142_"):
        raise RuntimeError(f"refusing to remove unexpected test fixture: {path}")
    shutil.rmtree(path)


class RecoveryGuardTests(unittest.TestCase):
    def _state(self):
        rows = {}
        for name in ("cache_seed1", "warm_seed1", "preflight_encoder_graph",
                     "preflight_core_trainable", "train_encoder_graph"):
            rows[name] = {"status": "passed", "artifact_valid": True, "child_pid": None}
        rows["train_core_trainable"] = {
            "status": "interrupted_foreign_gpu_owner",
            "returncode": -15, "foreign_owner_pids": [38618], "child_pid": 12345,
        }
        for name in ("predict_encoder_graph", "predict_core_trainable", "score_seed1"):
            rows[name] = {"status": "queued", "child_pid": None}
        return {"experiment": recovery.EXPERIMENT, "status": "failed", "supervisor_pid": 12344,
                "implementation_fingerprint": {"run.py": "abc"}, "queue_sha256": "def", "tasks": rows}

    def test_recovery_plan_contains_only_a_prediction_b_retry_b_prediction_and_score(self):
        run = SimpleNamespace(ARMS=("encoder_graph", "core_trainable"))
        tasks = recovery.recovery_tasks(run)
        self.assertEqual([t["stage"] for t in tasks], ["predict", "train", "predict", "score"])
        self.assertEqual([t.get("arm") for t in tasks], ["encoder_graph", "core_trainable", "core_trainable", None])
        self.assertEqual(recovery.recovery_groups(), [("predict_encoder_graph_recovery",),
            ("train_core_trainable_recovery",), ("predict_core_trainable_recovery",),
            ("score_seed1_recovery",)])
        self.assertNotIn("cache", [t["stage"] for t in tasks])
        self.assertNotIn("warm", [t["stage"] for t in tasks])
        self.assertNotIn("preflight", [t["stage"] for t in tasks])

    def test_terminal_failure_guard_requires_exact_known_interruption_and_fingerprint(self):
        temp = fixture_dir()
        try:
            proc = temp; state = self._state()
            evidence = recovery.validate_failed_queue(state, expected_fingerprint={"run.py": "abc"},
                expected_queue_sha256="def", proc_root=proc, run_path=Path("/x/run.py"),
                queue_path=Path("/x/stage_queue.py"))
            self.assertEqual(evidence["interrupted_task_returncode"], -15)
            self.assertEqual(evidence["foreign_owner_pids"], [38618])
            bad = self._state(); bad["tasks"]["train_core_trainable"]["returncode"] = 1
            with self.assertRaises(ValueError):
                recovery.validate_failed_queue(bad, expected_fingerprint={"run.py": "abc"},
                    expected_queue_sha256="def", proc_root=proc, run_path=Path("/x/run.py"),
                    queue_path=Path("/x/stage_queue.py"))
            with self.assertRaises(ValueError):
                recovery.validate_failed_queue(state, expected_fingerprint={"run.py": "changed"},
                    expected_queue_sha256="def", proc_root=proc, run_path=Path("/x/run.py"),
                    queue_path=Path("/x/stage_queue.py"))
        finally:
            cleanup_fixture(temp)
        temp = fixture_dir()
        try:
            proc = temp; bad = self._state(); bad["tasks"]["train_core_trainable"]["artifact_valid"] = True
            with self.assertRaises(ValueError):
                recovery.validate_failed_queue(bad, expected_fingerprint={"run.py": "abc"},
                    expected_queue_sha256="def", proc_root=proc, run_path=Path("/x/run.py"),
                    queue_path=Path("/x/stage_queue.py"))
        finally:
            cleanup_fixture(temp)

    def test_recovery_refuses_any_original_pid_still_present(self):
        temp = fixture_dir()
        try:
            proc = temp; (proc / "12344").mkdir()
            state = self._state()
            with self.assertRaisesRegex(RuntimeError, "process IDs still exist"):
                recovery.validate_failed_queue(state, expected_fingerprint={"run.py": "abc"},
                    expected_queue_sha256="def", proc_root=proc, run_path=Path("/x/run.py"),
                    queue_path=Path("/x/stage_queue.py"))
        finally:
            cleanup_fixture(temp)

    def test_recovery_refuses_unrecorded_live_snn_runner_process(self):
        temp = fixture_dir()
        try:
            proc = temp; row = proc / "54321"; row.mkdir()
            run_path = (Path("/opt/patch/collaborative_test/SW_0142_native32_masked_completion/run.py")).resolve()
            queue_path = (Path("/opt/patch/collaborative_test/SW_0142_native32_masked_completion/stage_queue.py")).resolve()
            (row / "cmdline").write_bytes(f"python\0{run_path}\0--stage\0train\0".encode())
            with self.assertRaisesRegex(RuntimeError, "live SW0142 runner"):
                recovery.validate_failed_queue(self._state(), expected_fingerprint={"run.py": "abc"},
                    expected_queue_sha256="def", proc_root=proc,
                    run_path=run_path, queue_path=queue_path)
        finally:
            cleanup_fixture(temp)

    def test_empty_incomplete_b_directory_is_archived_and_nonempty_is_preserved(self):
        root = fixture_dir()
        try:
            train = root / "training"; train.mkdir()
            source = train / "core_trainable_seed1"; source.mkdir()
            original_state = root / "original_queue.json"; original_state.write_bytes(b"unchanged")
            run = SimpleNamespace(TRAIN=train)
            destination = recovery._archive_partial_b(run, root / "new_runtime")
            self.assertIsNotNone(destination); self.assertEqual(destination.parent, train)
            self.assertFalse(source.exists()); self.assertTrue(destination.is_dir())
            self.assertEqual(original_state.read_bytes(), b"unchanged")
            source.mkdir(); payload = source / "unexpected.bin"; payload.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "not empty"):
                recovery._archive_partial_b(run, root / "new_runtime")
            self.assertEqual(payload.read_bytes(), b"preserve")
            (source / "TRAINING_COMPLETED.json").write_text("{}")
            with self.assertRaises(FileExistsError): recovery._archive_partial_b(run, root / "new_runtime")
        finally:
            cleanup_fixture(root)

    def test_missing_interrupted_b_directory_is_valid_and_not_archived(self):
        root = fixture_dir()
        try:
            source = root / "training" / "core_trainable_seed1"
            self.assertFalse(recovery.validate_interrupted_b_directory(source))
            run = SimpleNamespace(TRAIN=source.parent)
            self.assertIsNone(recovery._archive_partial_b(run, root / "new_runtime"))
            self.assertFalse(source.exists())
        finally:
            cleanup_fixture(root)

    def test_recovery_queue_uses_frozen_owned_child_terminator(self):
        calls = []
        frozen = SimpleNamespace(Queue=SimpleNamespace(_terminate=lambda proc: calls.append(proc)))
        queue = recovery.RecoveryQueue.__new__(recovery.RecoveryQueue)
        queue.sq = frozen
        process = object()
        queue._terminate(process)
        self.assertEqual(calls, [process])

    def test_queue_constructs_and_runs_four_stages_without_shadowing_run_method(self):
        root = fixture_dir()
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None)
        previous_fcntl = sys.modules.get("fcntl")
        sys.modules["fcntl"] = fake_fcntl
        try:
            class FakeRun:
                HERE = root
                @staticmethod
                def fingerprint(): return {"run": "frozen"}
                @staticmethod
                def sha256(_path): return "a" * 64

            class FakeStageQueue:
                __file__ = str(root / "stage_queue.py")
                owner = SimpleNamespace(__file__=str(root / "dispatcher.py"))
                Queue = object
                @staticmethod
                def task_plan():
                    return [
                        {"task_id": "train_encoder_graph", "stage": "train", "arm": "encoder_graph"},
                        {"task_id": "preflight_core_trainable", "stage": "preflight", "arm": "core_trainable"},
                    ]
                @staticmethod
                def validate_result(_task): return True
                @staticmethod
                def environment(task):
                    return {"CUDA_VISIBLE_DEVICES": "" if task["stage"] == "score" else "0"}

            archive = root / "runtime"; archive.mkdir()
            queue = recovery.RecoveryQueue(FakeRun, FakeStageQueue, archive,
                archive / "state.json", archive / "lock")
            executed = []
            def mocked_execute(task):
                env = FakeStageQueue.environment(task)
                if task["stage"] == "score":
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                else:
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
                executed.append(task["task_id"])
                queue._update(task["task_id"], status="passed", artifact_valid=True)
                return True
            queue._execute = mocked_execute
            rc = queue.run("original-state-sha", None)
            self.assertEqual(rc, 0)
            self.assertEqual(executed, [
                "predict_encoder_graph_recovery", "train_core_trainable_recovery",
                "predict_core_trainable_recovery", "score_seed1_recovery",
            ])
            self.assertEqual(queue.state["status"], "complete")
            self.assertIs(queue.experiment_run, FakeRun)
            self.assertTrue(callable(queue.run))
            queue.lock_handle.close()
        finally:
            if previous_fcntl is None:
                sys.modules.pop("fcntl", None)
            else:
                sys.modules["fcntl"] = previous_fcntl
            cleanup_fixture(root)

    def test_cli_propagates_recovery_failure_exit_code(self):
        old = recovery.run_recovery
        try:
            recovery.run_recovery = lambda *args, **kwargs: 1
            self.assertEqual(recovery.main(["--repo", ".", "--original-state", "missing", "--execute"]), 1)
        finally:
            recovery.run_recovery = old

    def test_recovery_namespaces_and_outputs_are_create_only(self):
        root = fixture_dir()
        try:
            archive = root / "recovery"
            self.assertEqual(recovery.require_new_recovery_archive(archive), archive)
            archive.mkdir()
            with self.assertRaises(FileExistsError): recovery.require_new_recovery_archive(archive)
            run = SimpleNamespace(RESULTS=root / "results", HERE=root)
            run.RESULTS.mkdir(); (run.RESULTS / "encoder_graph").mkdir()
            with self.assertRaises(FileExistsError): recovery.check_outputs_clear(run)
        finally:
            cleanup_fixture(root)


if __name__ == "__main__":
    unittest.main()
