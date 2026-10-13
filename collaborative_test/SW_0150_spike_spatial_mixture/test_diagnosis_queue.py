"""Focused CPU lifecycle checks for the SW0150 diagnostic reservation."""
from __future__ import annotations

import json
import hashlib
import os
import threading
import uuid
import unittest
from unittest import mock

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collaborative_test.SW_0150_spike_spatial_mixture import diagnosis_queue as queue
from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import make_decoder


class _Lease:
    def __init__(self, fd):
        self.fd = fd
        self.closed = False

    def fileno(self):
        return self.fd

    def close(self):
        self.closed = True


class _Child:
    pid = 45678

    def poll(self):
        return 0

    def wait(self):
        return 0


class DiagnosisQueueTests(unittest.TestCase):
    def _case_dir(self):
        path = queue.HERE / "results_archive" / f".queue_test_{os.getpid()}_{uuid.uuid4().hex}"
        path.mkdir()
        def cleanup():
            for item in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    item.rmdir()
            path.rmdir()
        self.addCleanup(cleanup)
        return path

    def test_dry_run_is_one_diagnostic_after_three_predictions(self):
        plan = queue.task_plan()
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["stage"], "diagnosis")
        self.assertEqual(set(plan[0]["depends_on"]), {
            "main_queue_complete", "predict_encoder_graph", "predict_joint",
            "predict_source_p"})
        argv = queue.command(plan[0])
        self.assertIn("collaborative_test.SW_0150_spike_spatial_mixture.diagnosis", argv)
        self.assertIn("--device", argv)
        self.assertIn("cuda:0", argv)

    def test_main_queue_absent_or_running_waits_without_prediction_validation(self):
        tmp = self._case_dir()
        state_path = tmp / "main.json"
        with mock.patch.object(queue, "MAIN_STATE", state_path), \
             mock.patch.object(queue.evaluate, "validate_prediction") as validate:
            self.assertEqual(queue.main_queue_state(), ("waiting", None))
            state_path.write_text(json.dumps({"status": "running", "tasks": {}}))
            self.assertEqual(queue.main_queue_state(), ("waiting", None))
            state_path.write_text(json.dumps({"status": {"bad": "shape"}, "tasks": {}}))
            self.assertEqual(queue.main_queue_state()[0], "failed")
            validate.assert_not_called()

    def test_failed_main_queue_is_terminal_and_never_reserves_gpu(self):
        tmp = self._case_dir()
        state_path = tmp / "main.json"
        state_path.write_text(json.dumps({"status": "score_failed", "tasks": {}}))
        with mock.patch.object(queue, "MAIN_STATE", state_path), \
             mock.patch.object(queue.owner_queue, "_acquire") as acquire:
            status, detail = queue.main_queue_state()
            self.assertEqual(status, "failed")
            self.assertIn("score_failed", detail)
            acquire.assert_not_called()

    def test_complete_main_queue_revalidates_each_frozen_prediction(self):
        required = {"predict_encoder_graph", "predict_joint", "predict_source_p"}
        tmp = self._case_dir()
        state_path = tmp / "main.json"
        state_path.write_text(json.dumps({
            "status": "complete",
            "tasks": {name: {"status": "passed"} for name in required}}))
        with mock.patch.object(queue, "MAIN_STATE", state_path), \
             mock.patch.object(queue.evaluate, "validate_prediction") as validate:
            self.assertEqual(queue.main_queue_state(), ("ready", None))
            self.assertEqual({call.args[0] for call in validate.call_args_list},
                             {"encoder_graph", "joint", "source_p"})
            self.assertEqual(validate.call_count, 3)

    def test_bad_prediction_in_completed_queue_fails_closed(self):
        names = {"predict_encoder_graph", "predict_joint", "predict_source_p"}
        tmp = self._case_dir()
        state_path = tmp / "main.json"
        state_path.write_text(json.dumps({"status": "complete", "tasks": {
            name: {"status": "passed"} for name in names}}))
        with mock.patch.object(queue, "MAIN_STATE", state_path), \
             mock.patch.object(queue.evaluate, "validate_prediction",
                               side_effect=[None, ValueError("bad prediction"), None]):
            status, detail = queue.main_queue_state()
            self.assertEqual(status, "failed")
            self.assertIn("bad prediction", detail)

    def test_unstable_gpu_releases_lease_then_reacquires_before_one_launch(self):
        tmp = self._case_dir()
        output, log, state = tmp / "result.json", tmp / "attempt.log", tmp / "state.json"
        task = dict(queue.task_plan()[0], output=str(output), log=str(log))
        obj = object.__new__(queue.DiagnosisQueue)
        obj.task = task
        obj.stop_event = threading.Event()
        obj.state = {"tasks": {task["task_id"]: {}}}
        leases = [_Lease(41), _Lease(42)]
        obj.diagnostic_fingerprint = queue.diagnosis.implementation_fingerprint()
        obj.queue_fingerprint = queue.implementation_fingerprint()

        class Flock:
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None

        acquire_rows = [(2, "uuid-a", 4, leases[0]), (1, "uuid-b", 2, leases[1])]
        with mock.patch.object(queue, "STATE", state), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue.owner_queue, "_acquire",
                               side_effect=acquire_rows) as acquire, \
             mock.patch.object(queue.owner_queue, "_stable_idle",
                               side_effect=[False, True]), \
             mock.patch.object(queue.subprocess, "Popen", return_value=_Child()) as popen, \
             mock.patch.object(queue, "validate_report", return_value=True):
            outcome = obj._launch_once()
        self.assertEqual(outcome, "passed")
        self.assertEqual(acquire.call_count, 2)
        self.assertTrue(leases[0].closed)
        self.assertTrue(leases[1].closed)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(popen.call_args.args[0][-4], "--device")
        self.assertEqual(popen.call_args.args[0][-3], "cuda:0")
        self.assertEqual(popen.call_args.args[0][-1], str(output))
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertTrue(log.is_file())

    def test_launch_validates_actual_diagnostic_fingerprint_and_completed_lineage(self):
        tmp = self._case_dir()
        output, log, state = tmp / "report.json", tmp / "attempt.log", tmp / "state.json"
        ids = list(range(1320, 1384))
        pool_rows = np.arange(64, dtype=np.int64)
        source_manifest = tmp / "source_manifest.json"
        source_manifest.write_text("source lineage")
        source = {"ids": ids, "rows": pool_rows, "core_sha256": "e" * 64,
                  "manifest_path": source_manifest}
        assets, preflight_sha, warm_sha = {"train_cache_sha256": "a" * 64}, "c" * 64, "d" * 64
        validated, training_report = {}, {}
        for arm in ("encoder_graph", "joint"):
            paths = {key: tmp / f"{arm}.{key}" for key in
                     ("manifest", "checkpoint", "optimizer", "history")}
            for key, path in paths.items():
                path.write_text(f"{arm}-{key}")
            manifest = {
                "source_core_sha256": source["core_sha256"],
                "source_manifest_sha256": queue.diagnosis.train.sha(source_manifest),
                "source_training_ids_sha256": queue.diagnosis.train.ids_sha(ids),
                "rgb_asset_validation": assets,
                "preflight_sha256": preflight_sha,
                "preflight_warm_sha256": warm_sha,
                "implementation_sha256": "f" * 64,
            }
            validated[arm] = {
                "manifest": manifest,
                "manifest_sha256": queue.diagnosis.train.sha(paths["manifest"]),
                "checkpoint_sha256": queue.diagnosis.train.sha(paths["checkpoint"]),
                "optimizer": paths["optimizer"], "history": paths["history"],
            }
            training_report[arm] = {
                "manifest_sha256": validated[arm]["manifest_sha256"],
                "checkpoint_sha256": validated[arm]["checkpoint_sha256"],
                "optimizer_sha256": queue.diagnosis.train.sha(paths["optimizer"]),
                "history_sha256": queue.diagnosis.train.sha(paths["history"]),
                "implementation_sha256": manifest["implementation_sha256"],
            }

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

        wrapped = torch.nn.Module()
        wrapped.spatial_decoder = make_decoder(8, seed=150001)
        foundation = type("TestFoundation", (), {})()
        foundation.wrapped, foundation.encoder = wrapped, TinyEncoder()
        rgb_cache = np.zeros((64, 128, 128, 3), dtype=np.uint8)
        diag_source = {"ids": ids, "rows": pool_rows}

        def deterministic_forward(_foundation, clean_rgb, *, live_tail_steps):
            p = torch.ones((clean_rgb.shape[0], 1024, 1), dtype=torch.float32,
                           device=clean_rgb.device)
            gamma = torch.zeros((clean_rgb.shape[0], 1024, 8), dtype=torch.float32,
                                device=clean_rgb.device)
            return {"p": p, "gamma_content": gamma,
                    "live_tail_steps": live_tail_steps}

        with mock.patch.object(queue.diagnosis.integration, "forward_assignment",
                               side_effect=deterministic_forward):
            arm_rows = {
                arm: queue.diagnosis._diagnose_arm(
                    arm, foundation=foundation, source=diag_source,
                    admission={"sigma2": 0.02}, rgb_cache=rgb_cache)
                for arm in ("encoder_graph", "joint")
            }
        diag_fp = queue.diagnosis.implementation_fingerprint()
        report = {
            "status": "complete", "experiment": queue.NAME,
            "diagnostic": queue.diagnosis.DIAGNOSTIC,
            "ground_truth_used": False, "masks_read": False, "optimizer_updates": 0,
            "protocol_sha256": queue.sha256_file(queue.diagnosis.PROTOCOL_PATH),
            "implementation_fingerprint": diag_fp,
            "implementation_sha256": hashlib.sha256(json.dumps(
                diag_fp, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest(),
            "source": {"seed": 1, "core_sha256": source["core_sha256"],
                       "manifest_sha256": queue.diagnosis.train.sha(source_manifest),
                       "training_ids_sha256": queue.diagnosis.train.ids_sha(ids),
                       "rgb_asset_validation": assets,
                       "preflight_sha256": preflight_sha, "warm_sha256": warm_sha,
                       "first64_image_ids": ids,
                       "first64_pool_rows": [int(value) for value in pool_rows]},
            "training": training_report, "arms": arm_rows,
        }
        task = dict(queue.task_plan()[0], output=str(output), log=str(log))
        obj = object.__new__(queue.DiagnosisQueue)
        obj.task, obj.stop_event = task, threading.Event()
        obj.state = {"tasks": {task["task_id"]: {}}}
        obj.diagnostic_fingerprint = diag_fp
        obj.queue_fingerprint = queue.implementation_fingerprint()
        lease = _Lease(52)

        class Flock:
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None

        def child_writes_real_report(_argv, **_kwargs):
            output.write_text(json.dumps(report))
            return _Child()

        with mock.patch.object(queue, "STATE", state), \
             mock.patch.object(queue, "OUTPUT", output), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue.owner_queue, "_acquire",
                               return_value=(2, "gpu-uuid", 3, lease)), \
             mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
             mock.patch.object(queue.subprocess, "Popen",
                               side_effect=child_writes_real_report), \
             mock.patch.object(queue.diagnosis.train, "_source_record", return_value=source), \
             mock.patch.object(queue.diagnosis.train, "validate_training",
                               side_effect=lambda arm: validated[arm]):
            self.assertEqual(obj._launch_once(), "passed")
            original = json.loads(output.read_text())
            bad_ids = json.loads(json.dumps(original))
            bad_ids["source"]["first64_image_ids"][0] += 1
            output.write_text(json.dumps(bad_ids))
            self.assertFalse(queue.validate_report(output, expected_fingerprint=diag_fp))
            bad_checkpoint = json.loads(json.dumps(original))
            bad_checkpoint["training"]["joint"]["checkpoint_sha256"] = "0" * 64
            output.write_text(json.dumps(bad_checkpoint))
            self.assertFalse(queue.validate_report(output, expected_fingerprint=diag_fp))

    def test_launch_accepts_actual_diagnostic_report_fingerprint_and_lineage(self):
        tmp = self._case_dir()
        output, log, state = tmp / "report.json", tmp / "attempt.log", tmp / "state.json"
        ids = list(range(1320, 1384))
        rows = np.arange(64, dtype=np.int64)
        assets = {"train_cache_sha256": "a" * 64, "manifest_sha256": "b" * 64}
        preflight_sha, warm_sha = "c" * 64, "d" * 64
        source_manifest = tmp / "source_manifest.json"
        source_manifest.write_text("source lineage")
        source = {"ids": ids, "rows": rows, "core_sha256": "e" * 64,
                  "manifest_path": source_manifest}
        validated, training_summary = {}, {}
        for arm in ("encoder_graph", "joint"):
            manifest_path = tmp / f"{arm}.manifest"
            checkpoint_path = tmp / f"{arm}.checkpoint"
            optimizer_path = tmp / f"{arm}.optimizer"
            history_path = tmp / f"{arm}.history"
            for path, value in ((manifest_path, "manifest"), (checkpoint_path, "checkpoint"),
                                (optimizer_path, "optimizer"), (history_path, "history")):
                path.write_text(f"{arm}-{value}")
            manifest = {
                "source_core_sha256": source["core_sha256"],
                "source_manifest_sha256": queue.diagnosis.train.sha(source_manifest),
                "source_training_ids_sha256": queue.diagnosis.train.ids_sha(ids),
                "rgb_asset_validation": assets,
                "preflight_sha256": preflight_sha,
                "preflight_warm_sha256": warm_sha,
                "implementation_sha256": "f" * 64,
            }
            validated[arm] = {
                "manifest": manifest,
                "manifest_sha256": queue.diagnosis.train.sha(manifest_path),
                "checkpoint_sha256": queue.diagnosis.train.sha(checkpoint_path),
                "optimizer": optimizer_path,
                "history": history_path,
            }
            training_summary[arm] = {
                "manifest_sha256": validated[arm]["manifest_sha256"],
                "checkpoint_sha256": validated[arm]["checkpoint_sha256"],
                "optimizer_sha256": queue.diagnosis.train.sha(optimizer_path),
                "history_sha256": queue.diagnosis.train.sha(history_path),
                "implementation_sha256": manifest["implementation_sha256"],
            }

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

        wrapped = torch.nn.Module()
        wrapped.spatial_decoder = make_decoder(8, seed=150001)
        foundation = type("Foundation", (), {})()
        foundation.wrapped = wrapped
        foundation.encoder = TinyEncoder()
        rgb_cache = np.zeros((64, 128, 128, 3), dtype=np.uint8)
        diagnostic_source = {"ids": ids, "rows": rows}

        def fake_forward(_foundation, clean_rgb, *, live_tail_steps):
            p = torch.zeros((clean_rgb.shape[0], 1024, 1), dtype=torch.float32,
                            device=clean_rgb.device)
            p[..., 0] = 1.0
            gamma = torch.zeros((clean_rgb.shape[0], 1024, 8), dtype=torch.float32,
                                device=clean_rgb.device)
            return {"p": p, "gamma_content": gamma,
                    "live_tail_steps": live_tail_steps}

        with mock.patch.object(queue.diagnosis.integration, "forward_assignment",
                               side_effect=fake_forward):
            arms = {
                arm: queue.diagnosis._diagnose_arm(
                    arm, foundation=foundation, source=diagnostic_source,
                    admission={"sigma2": 0.02}, rgb_cache=rgb_cache)
                for arm in ("encoder_graph", "joint")
            }
        diagnostic_fingerprint = queue.diagnosis.implementation_fingerprint()
        report = {
            "status": "complete", "experiment": queue.NAME,
            "diagnostic": queue.diagnosis.DIAGNOSTIC,
            "ground_truth_used": False, "masks_read": False, "optimizer_updates": 0,
            "protocol_sha256": queue.sha256_file(queue.diagnosis.PROTOCOL_PATH),
            "implementation_fingerprint": diagnostic_fingerprint,
            "implementation_sha256": hashlib.sha256(json.dumps(
                diagnostic_fingerprint, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest(),
            "source": {"seed": 1, "core_sha256": source["core_sha256"],
                       "manifest_sha256": queue.diagnosis.train.sha(source_manifest),
                       "training_ids_sha256": queue.diagnosis.train.ids_sha(ids),
                       "rgb_asset_validation": assets,
                       "preflight_sha256": preflight_sha, "warm_sha256": warm_sha,
                       "first64_image_ids": ids,
                       "first64_pool_rows": [int(value) for value in rows]},
            "training": training_summary, "arms": arms,
        }
        task = dict(queue.task_plan()[0], output=str(output), log=str(log))
        obj = object.__new__(queue.DiagnosisQueue)
        obj.task = task
        obj.stop_event = threading.Event()
        obj.state = {"tasks": {task["task_id"]: {}}}
        obj.diagnostic_fingerprint = diagnostic_fingerprint
        obj.queue_fingerprint = queue.implementation_fingerprint()
        lease = _Lease(51)

        class Flock:
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None

        def write_report_and_return_child(argv, **_kwargs):
            output.write_text(json.dumps(report))
            return _Child()

        with mock.patch.object(queue, "STATE", state), \
             mock.patch.object(queue, "OUTPUT", output), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue.owner_queue, "_acquire",
                               return_value=(0, "gpu-uuid", 3, lease)), \
             mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
             mock.patch.object(queue.subprocess, "Popen",
                               side_effect=write_report_and_return_child), \
             mock.patch.object(queue.diagnosis.train, "_source_record", return_value=source), \
             mock.patch.object(queue.diagnosis.train, "validate_training",
                               side_effect=lambda arm: validated[arm]):
            outcome = obj._launch_once()
            self.assertTrue(queue.validate_report(
                output, expected_fingerprint=diagnostic_fingerprint))
        self.assertEqual(outcome, "passed")

    def test_existing_output_is_preserved_and_not_restarted(self):
        tmp = self._case_dir()
        output = tmp / "already.json"
        output.write_text('{"status":"complete"}')
        class Flock:
            LOCK_EX = 2
            LOCK_NB = 4
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None
        with mock.patch.object(queue, "RUNTIME", tmp / "runtime"), \
             mock.patch.object(queue, "LOCK", tmp / "runtime" / "lock"), \
             mock.patch.object(queue, "OUTPUT", output), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue, "validate_report", return_value=False):
            with self.assertRaises(FileExistsError):
                queue.DiagnosisQueue()
        self.assertEqual(output.read_text(), '{"status":"complete"}')

    def test_existing_valid_report_is_adopted_without_rewrite(self):
        tmp = self._case_dir()
        output = tmp / "already.json"
        output.write_text('{"status":"complete"}')
        class Flock:
            LOCK_EX = 2
            LOCK_NB = 4
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None
        with mock.patch.object(queue, "RUNTIME", tmp / "runtime"), \
             mock.patch.object(queue, "LOCK", tmp / "runtime" / "lock"), \
             mock.patch.object(queue, "OUTPUT", output), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue, "implementation_fingerprint", return_value={"fp": "x"}), \
             mock.patch.object(queue, "validate_report", return_value=True) as validate:
            obj = queue.DiagnosisQueue()
            self.assertTrue(obj.existing_complete)
            validate.assert_called_once_with(
                output, expected_fingerprint=obj.diagnostic_fingerprint)
            obj._unlock()
        self.assertEqual(output.read_text(), '{"status":"complete"}')

    def test_transient_gpu_query_error_does_not_stop_owned_diagnostic(self):
        tmp = self._case_dir()
        output, log, state = tmp / "result.json", tmp / "attempt.log", tmp / "state.json"
        task = dict(queue.task_plan()[0], output=str(output), log=str(log))
        obj = object.__new__(queue.DiagnosisQueue)
        obj.task, obj.stop_event = task, threading.Event()
        obj.state = {"tasks": {task["task_id"]: {}}}
        obj.diagnostic_fingerprint = queue.diagnosis.implementation_fingerprint()
        obj.queue_fingerprint = queue.implementation_fingerprint()
        lease = _Lease(61)

        class Flock:
            LOCK_UN = 8
            @staticmethod
            def flock(_fd, _op):
                return None

        class QueryChild:
            pid = 45679
            def __init__(self):
                self.calls = 0
            def poll(self):
                self.calls += 1
                return 0 if self.calls >= 3 else None
            def wait(self):
                return 0

        child = QueryChild()
        with mock.patch.object(queue, "STATE", state), \
             mock.patch.object(queue, "OUTPUT", output), \
             mock.patch.object(queue, "fcntl", Flock), \
             mock.patch.object(queue.owner_queue, "_acquire",
                               return_value=(0, "gpu-uuid", 3, lease)), \
             mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
             mock.patch.object(queue.owner_queue.gpu_guard, "_gpu_info",
                               side_effect=[RuntimeError("temporary query failure"),
                                            ("gpu-uuid", 3, [])]), \
             mock.patch.object(queue.owner_queue.gpu_guard, "_foreign_owners", return_value=[]), \
             mock.patch.object(queue.owner_queue.gpu_guard, "_process_tree", return_value=[]), \
             mock.patch.object(queue.owner_queue, "_stop_owned") as stop_owned, \
             mock.patch.object(queue.subprocess, "Popen", return_value=child), \
             mock.patch.object(queue.time, "sleep", return_value=None), \
             mock.patch.object(queue, "validate_report", return_value=True):
            self.assertEqual(obj._launch_once(), "passed")
        stop_owned.assert_not_called()
        self.assertTrue(lease.closed)
        self.assertTrue(log.is_file())


if __name__ == "__main__":
    unittest.main()
