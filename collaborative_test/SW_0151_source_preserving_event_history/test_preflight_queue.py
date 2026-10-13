from __future__ import annotations

import hashlib
import json
import sys
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (ROOT, ROOT / "collaborative_test", ROOT / "snn_kuramoto_bidirectional"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from collaborative_test.SW_0151_source_preserving_event_history import preflight, preflight_queue as queue


class WorkspaceTempDirectory:
    """Use authorized workspace scratch; system-temp ACLs can be restricted on Windows."""
    def __enter__(self):
        self.path = ROOT.parent / "tmpo5v9ehjh" / f"sw151_queue_{uuid.uuid4().hex}"
        self.path.mkdir(parents=True)
        return str(self.path)

    def __exit__(self, *_exc):
        for child in self.path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir() and not any(child.iterdir()):
                child.rmdir()
        self.path.rmdir()


class FakeLease:
    def __init__(self, fd):
        self.fd = fd
        self.closed = False

    def fileno(self):
        return self.fd

    def close(self):
        self.closed = True


class FakeFcntl:
    LOCK_UN = 8

    @staticmethod
    def flock(_fd, _op):
        return None


class SW0151PreflightQueueTests(unittest.TestCase):
    def test_registered_plan_is_single_cuda_task_with_both_upstream_gates(self):
        plan = queue.task_plan()
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["depends_on"], [
            "sw150_main_complete", "sw150_diagnostic_complete"])
        argv = queue.command(plan[0], device="cuda:0")
        self.assertIn("collaborative_test.SW_0151_source_preserving_event_history.preflight", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            queue.command({**plan[0], "seed": 0})

    def test_pending_or_failed_upstream_never_launches(self):
        for status, expected in (("waiting", "waiting"), ("failed", "failed")):
            with mock.patch.object(queue, "main_queue_readiness", return_value=(status, "main")), \
                    mock.patch.object(queue, "diagnostic_queue_readiness", return_value=("ready", None)):
                self.assertEqual(queue.upstream_readiness()[0], expected)

        q = queue.PreflightQueue.__new__(queue.PreflightQueue)
        q.existing_complete = False
        q.stop_event = threading.Event()
        q.state = {"task": queue.task_plan()[0]}
        q._unlock = lambda: None
        with mock.patch.object(queue, "upstream_readiness", return_value=("failed", "terminal upstream")), \
                mock.patch.object(q, "_launch_once") as launch:
            # State writes are redirected into a temporary directory.
            with WorkspaceTempDirectory() as temp:
                state_path = Path(temp) / "state.json"
                with mock.patch.object(queue, "STATE", state_path):
                    self.assertEqual(q.run(poll_seconds=0), 1)
                    self.assertEqual(json.loads(state_path.read_text())["status"],
                                     "blocked_sw150_terminal_failure")
            launch.assert_not_called()

    def test_diagnostic_waits_for_terminal_pass_and_valid_report(self):
        with WorkspaceTempDirectory() as temp:
            state_path = Path(temp) / "diagnosis.json"
            task_id = "first64_train_color_bin_assignment_permutation"
            with mock.patch.object(queue.sw150_diagnosis_queue, "STATE", state_path):
                self.assertEqual(queue.diagnostic_queue_readiness()[0], "waiting")
                state_path.write_text(json.dumps({"status": "running", "tasks": {}}))
                self.assertEqual(queue.diagnostic_queue_readiness()[0], "waiting")
                state_path.write_text(json.dumps({"status": "complete", "tasks": {
                    task_id: {"status": "failed"}}}))
                self.assertEqual(queue.diagnostic_queue_readiness()[0], "failed")
                state_path.write_text(json.dumps({"status": "passed", "tasks": {
                    task_id: {"status": "passed"}}}))
                with mock.patch.object(queue.sw150_diagnosis_queue, "validate_report", return_value=True):
                    self.assertEqual(queue.diagnostic_queue_readiness(), ("ready", None))

    def _valid_report_fixture(self, temp):
        root = Path(temp)
        manifest = root / "source_manifest.json"
        parity = root / "parity.json"
        manifest.write_text("{}\n", encoding="utf-8")
        parity.write_text("{}\n", encoding="utf-8")
        ids = list(range(4096))
        rows = np.arange(1000, 5096, dtype=np.int64)
        asset_record = {"train_cache_sha256": "a" * 64, "train_manifest_sha256": "b" * 64}
        source_sha = "c" * 64
        foundation_fp = {"foundation.py": "d" * 64}
        expected_fp = preflight.implementation_fingerprint()
        with mock.patch.object(preflight, "PARITY_PATH", parity), \
                mock.patch.object(preflight.sw130, "validate_rgb_assets", return_value=asset_record), \
                mock.patch.object(preflight.sw130, "source_contract",
                                  return_value=(root / "source.pt", manifest, {"steps": 70000},
                                                rows, ids, source_sha)), \
                mock.patch.object(preflight.sw135, "implementation_fingerprint",
                                  return_value=foundation_fp):
            report = {
                "experiment": queue.NAME,
                "stage": "actual_source_native32_parity_and_scale_preflight",
                "status": "passed_kappa_zero_actual_source_parity", "seed": 1,
                "training_admission": False, "ground_truth_used": False,
                "masks_read": False, "validation_or_test_ids_read": False,
                "optimizer_updates": 0, "implementation_fingerprint": expected_fp,
                "source": {"first64_train_ids": ids[:64],
                           "first64_pool_rows": rows[:64].tolist(),
                           "core_checkpoint_sha256": source_sha,
                           "source_manifest_sha256": queue.sha256_file(manifest),
                           "ordered_train_ids_sha256": hashlib.sha256(
                               np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                           "saved_native32_parity_path": str(parity),
                           "saved_native32_parity_sha256": queue.sha256_file(parity),
                           "foundation_implementation_fingerprint": foundation_fp,
                           "rgb_asset_validation": asset_record},
                "parity": {"batches": 16, "exact_trace_batches": 16,
                           "exact_production_q_batches": 16,
                           "all_six_trace_fields_exact": True,
                           "production_actual_spike_q_exact": True},
                "calibration": {"source_rollouts_reused_for_calibration": True,
                                "fixed_per_component_scale": [1., 2., 3., 4.]},
                "model": {"native32_grid": [32, 32], "patches": 1024,
                          "steps": 1024, "settle": 512, "live_tail_steps": 0,
                          "new_parameters": 8, "kappa_initial": 0.0,
                          "registered_module_state_unchanged": True,
                          "registered_module_state_sha256_before": {
                              "encoder": "1" * 64, "patcher": "2" * 64,
                              "wrapped": "3" * 64, "event_history": "4" * 64},
                          "registered_module_state_sha256_after": {
                              "encoder": "1" * 64, "patcher": "2" * 64,
                              "wrapped": "3" * 64, "event_history": "4" * 64}},
                "descriptive_events": {
                    "settled_event_on_fraction_by_component": [.1] * 4,
                    "settled_event_exact_zero_fraction_by_component": [.8] * 4,
                    "settled_event_exact_one_fraction_by_component": [.2] * 4,
                    "settled_history_variance_by_component": [.01] * 4},
            }
            return report, expected_fp, parity

    def test_output_validator_binds_real_source_order_and_proof_fields(self):
        with WorkspaceTempDirectory() as temp:
            report, fingerprint, parity = self._valid_report_fixture(temp)
            output = Path(temp) / "result.json"
            # Re-enter the fixture patches while validating its production-shaped report.
            with mock.patch.object(preflight, "PARITY_PATH", parity), \
                    mock.patch.object(preflight.sw130, "validate_rgb_assets",
                                      return_value=report["source"]["rgb_asset_validation"]), \
                    mock.patch.object(preflight.sw130, "source_contract", return_value=(
                        Path(temp) / "source.pt", Path(temp) / "source_manifest.json",
                        {"steps": 70000}, np.arange(1000, 5096, dtype=np.int64),
                        list(range(4096)), "c" * 64)), \
                    mock.patch.object(preflight.sw135, "implementation_fingerprint",
                                      return_value=report["source"]["foundation_implementation_fingerprint"]):
                output.write_text(json.dumps(report), encoding="utf-8")
                self.assertTrue(queue.validate_report(output, expected_fingerprint=fingerprint))
                report["source"]["first64_train_ids"][0] = -1
                output.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.validate_report(output, expected_fingerprint=fingerprint))
                report["source"]["first64_train_ids"][0] = 0
                report["parity"]["production_actual_spike_q_exact"] = False
                output.write_text(json.dumps(report), encoding="utf-8")
                self.assertFalse(queue.validate_report(output, expected_fingerprint=fingerprint))

    def _bare_queue(self, temp):
        q = queue.PreflightQueue.__new__(queue.PreflightQueue)
        q.stop_event = threading.Event()
        q.task = queue.task_plan()[0]
        q.fingerprint = {"queue": "a" * 64}
        q.preflight_fingerprint = {"preflight": "b" * 64}
        q.state = {"task_id": q.task["task_id"]}
        q._verify_fingerprint = lambda: None
        return q

    def test_unstable_gpu_releases_lease_and_reacquires_before_launch(self):
        with WorkspaceTempDirectory() as temp:
            q = self._bare_queue(temp)
            output, log, state = (Path(temp) / name for name in ("out.json", "attempt.log", "state.json"))
            with mock.patch.object(queue, "OUTPUT", output), mock.patch.object(queue, "LOG", log), \
                    mock.patch.object(queue, "STATE", state), mock.patch.object(queue, "RUNTIME", Path(temp)), \
                    mock.patch.object(queue, "fcntl", FakeFcntl), \
                    mock.patch.object(queue.owner_queue, "_acquire", side_effect=[
                        (2, "uuid2", 40, FakeLease(21)), (3, "uuid3", 20, FakeLease(22))]) as acquire, \
                    mock.patch.object(queue.owner_queue, "_stable_idle", side_effect=[False, True]), \
                    mock.patch.object(queue.subprocess, "Popen") as popen, \
                    mock.patch.object(queue, "validate_report", return_value=True):
                class Child:
                    pid = 123
                    def poll(self): return 0
                    def wait(self): return 0
                def launch(argv, **kwargs):
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
                    self.assertEqual(kwargs["pass_fds"], (22,))
                    output.write_text("{}", encoding="utf-8")
                    return Child()
                popen.side_effect = launch
                result = q._launch_once()
                self.assertEqual(result, "passed")
                self.assertEqual(acquire.call_count, 2)

    def test_foreign_collision_stops_only_owned_child_without_retry(self):
        with WorkspaceTempDirectory() as temp:
            q = self._bare_queue(temp)
            output, log, state = (Path(temp) / name for name in ("out.json", "attempt.log", "state.json"))
            class Child:
                pid = 456
                calls = 0
                def poll(self):
                    self.calls += 1
                    return 0 if self.calls >= 3 else None
                def wait(self): return -15
            child = Child()
            with mock.patch.object(queue, "OUTPUT", output), mock.patch.object(queue, "LOG", log), \
                    mock.patch.object(queue, "STATE", state), mock.patch.object(queue, "RUNTIME", Path(temp)), \
                    mock.patch.object(queue, "fcntl", FakeFcntl), \
                    mock.patch.object(queue.owner_queue, "_acquire", return_value=(1, "uuid1", 10, FakeLease(31))) as acquire, \
                    mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
                    mock.patch.object(queue.subprocess, "Popen") as popen, \
                    mock.patch.object(queue.time, "sleep"), \
                    mock.patch.object(queue.gpu_guard, "_gpu_info", return_value=("uuid1", 10, [999])), \
                    mock.patch.object(queue.gpu_guard, "_process_tree", return_value={456}), \
                    mock.patch.object(queue.gpu_guard, "_foreign_owners", return_value=[999]), \
                    mock.patch.object(queue.owner_queue, "_stop_owned") as stop_owned:
                popen.return_value = child
                self.assertEqual(q._launch_once(), "interrupted_foreign_gpu_owner")
                stop_owned.assert_called_once_with(child)
                self.assertEqual(acquire.call_count, 1)
                self.assertTrue(log.is_file())
                self.assertFalse(output.exists())

    def test_gpu_query_failure_does_not_kill_owned_child(self):
        with WorkspaceTempDirectory() as temp:
            q = self._bare_queue(temp)
            output, log, state = (Path(temp) / name for name in ("out.json", "attempt.log", "state.json"))
            class Child:
                pid = 789
                calls = 0
                def poll(self):
                    self.calls += 1
                    return 0 if self.calls >= 3 else None
                def wait(self): return 0
            child = Child()
            with mock.patch.object(queue, "OUTPUT", output), mock.patch.object(queue, "LOG", log), \
                    mock.patch.object(queue, "STATE", state), mock.patch.object(queue, "RUNTIME", Path(temp)), \
                    mock.patch.object(queue, "fcntl", FakeFcntl), \
                    mock.patch.object(queue.owner_queue, "_acquire", return_value=(0, "uuid0", 10, FakeLease(41))), \
                    mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
                    mock.patch.object(queue.subprocess, "Popen") as popen, \
                    mock.patch.object(queue.time, "sleep"), \
                    mock.patch.object(queue.gpu_guard, "_gpu_info", side_effect=RuntimeError("transient query")), \
                    mock.patch.object(queue.owner_queue, "_stop_owned") as stop_owned, \
                    mock.patch.object(queue, "validate_report", return_value=True):
                def launch(*_args, **_kwargs):
                    output.write_text("{}", encoding="utf-8")
                    return child
                popen.side_effect = launch
                self.assertEqual(q._launch_once(), "passed")
                stop_owned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
