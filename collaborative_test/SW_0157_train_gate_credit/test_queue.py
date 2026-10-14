import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

from collaborative_test.SW_0157_train_gate_credit import diagnostic, diagnostic_queue


class GateCreditQueueTests(unittest.TestCase):
    def test_single_read_only_gpu_job_plan_and_fixed_argv(self):
        plan = diagnostic_queue.task_plan()
        self.assertEqual(plan["task_id"], "train_gate_credit_eight_train_images")
        self.assertEqual(plan["models"], list(diagnostic.MODEL_ORDER))
        self.assertEqual(plan["source_positions"], list(diagnostic.POSITIONS))
        self.assertEqual(plan["watchdog_seconds"], 300.0)
        self.assertIn("SW0155_encoder_graph_training_complete", plan["depends_on"])
        self.assertIn("SW0155_joint_training_complete", plan["depends_on"])
        argv = diagnostic_queue.command()
        self.assertIn("collaborative_test.SW_0157_train_gate_credit.diagnostic", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertEqual(argv[argv.index("--watchdog-seconds") + 1], "300.0")

    def test_queue_fingerprint_resolves_real_owner_and_report_inputs(self):
        report_fp = diagnostic_queue.report_fingerprint()
        self.assertEqual(report_fp, diagnostic.implementation_fingerprint())
        queue_fp = diagnostic_queue.implementation_fingerprint()
        self.assertEqual(queue_fp[
            "collaborative_test/SW_0157_train_gate_credit/diagnostic_queue.py"],
            diagnostic_queue.sha256_file(diagnostic_queue.HERE / "diagnostic_queue.py"))
        owner_key = diagnostic_queue.owner.OWNER_GUARD.relative_to(diagnostic_queue.ROOT).as_posix()
        self.assertEqual(queue_fp[owner_key], diagnostic_queue.owner.OWNER_GUARD_SHA256)

    def test_missing_upstream_is_pending_without_gpu_acquisition(self):
        missing = Path("missing-upstream")
        with mock.patch.object(diagnostic_queue.sw143, "_preflight_path", return_value=missing), \
             mock.patch.object(diagnostic_queue.sw143, "_arm_dir", return_value=missing), \
             mock.patch.object(diagnostic_queue.Path, "is_file", autospec=True, return_value=False):
            ready, state = diagnostic_queue._upstream_ready()
        self.assertFalse(ready)
        self.assertEqual(state, "waiting_for_sw0143_T_seed1")

    def test_diagnostic_child_import_preserves_assigned_cuda_and_stays_uninitialized(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "3"
        env["PYTHONPATH"] = str(diagnostic.ROOT)
        code = (
            "import os, numpy, torch; "
            "import collaborative_test.SW_0157_train_gate_credit.diagnostic; "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES')); "
            "print(torch.cuda.is_initialized())"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=diagnostic.ROOT,
            env=env, text=True, capture_output=True, timeout=60, check=True)
        self.assertEqual(completed.stdout.splitlines(), ["3", "False"])

    def test_final_gpu_admission_rejects_owner_uuid_memory_and_query_failure(self):
        guard = diagnostic_queue.owner.gpu_guard
        cases = [
            ("uuid", ("other", 2, [])),
            ("owner", ("expected", 2, [99123])),
            ("memory", ("expected", diagnostic_queue.MAX_MEMORY_MIB + 1, [])),
        ]
        for _name, value in cases:
            with self.subTest(case=_name), mock.patch.object(guard, "_gpu_info", return_value=value):
                admitted, evidence = diagnostic_queue._final_gpu_admission(0, "expected")
                self.assertFalse(admitted)
                self.assertNotEqual(evidence["reason"], "idle")
        with mock.patch.object(guard, "_gpu_info", side_effect=RuntimeError("query failed")):
            admitted, evidence = diagnostic_queue._final_gpu_admission(0, "expected")
        self.assertFalse(admitted)
        self.assertEqual(evidence["reason"], "gpu_query_error")

    def test_run_releases_and_reacquires_if_gpu_becomes_busy_after_upstream_validation(self):
        queue = object.__new__(diagnostic_queue.DiagnosticQueue)
        queue.stop_event = threading.Event()
        queue.state_lock = threading.RLock()
        queue._unlock = mock.Mock()
        queue._verify = mock.Mock()
        queue.existing_complete = False
        queue.report_fp = {"diagnostic": "fp"}
        updates = []
        queue._update = lambda status, **fields: updates.append((status, fields))
        upstream = {"validated": "fixed"}
        order = []
        def upstream_ready():
            order.append("upstream")
            return True, upstream
        first, second = mock.Mock(), mock.Mock()
        first.fileno.return_value, second.fileno.return_value = 71, 72
        acquisitions = iter([(0, "gpu-uuid", 4, first), (0, "gpu-uuid", 4, second)])
        proc = mock.Mock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        guard = diagnostic_queue.owner.gpu_guard
        gpu_reads = iter([("gpu-uuid", 4, [54321]), ("gpu-uuid", 4, [])])

        class FakeFcntl:
            LOCK_UN = 8
            def __init__(self): self.unlocked = []
            def flock(self, fd, operation): self.unlocked.append((fd, operation))
        fake_fcntl = FakeFcntl()
        with mock.patch.object(diagnostic_queue, "_upstream_ready", side_effect=upstream_ready), \
             mock.patch.object(diagnostic_queue, "_check_absent"), \
             mock.patch.object(diagnostic_queue.owner, "_acquire", side_effect=lambda _stop: next(acquisitions)), \
             mock.patch.object(diagnostic_queue.owner, "_stable_idle", return_value=True), \
             mock.patch.object(guard, "_gpu_info", side_effect=lambda _gpu: (order.append("gpu_query"), next(gpu_reads))[1]), \
             mock.patch.object(diagnostic_queue.owner, "_stop_owned") as stop_owned, \
             mock.patch.object(diagnostic_queue.subprocess, "Popen", side_effect=lambda *a, **k: (order.append("popen"), proc)[1]), \
             mock.patch.object(diagnostic_queue, "validate_report", return_value=True), \
             mock.patch.object(diagnostic_queue, "OUTPUT", Path("unused-report.json")), \
             mock.patch.object(diagnostic_queue, "LOG", Path("unused-attempt.log")), \
             mock.patch.object(Path, "open", mock.mock_open()), \
             mock.patch.object(diagnostic_queue, "fcntl", fake_fcntl):
            result = queue.run()
        self.assertEqual(result, 0)
        self.assertEqual(order.count("popen"), 1)
        self.assertLess(order.index("upstream", 1), order.index("gpu_query"))
        self.assertLess(order.index("gpu_query", order.index("upstream", 1)), order.index("popen"))
        self.assertTrue(any(status == "waiting_for_exclusive_gpu" for status, _ in updates))
        self.assertIn((71, fake_fcntl.LOCK_UN), fake_fcntl.unlocked)
        self.assertIn((72, fake_fcntl.LOCK_UN), fake_fcntl.unlocked)
        stop_owned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
