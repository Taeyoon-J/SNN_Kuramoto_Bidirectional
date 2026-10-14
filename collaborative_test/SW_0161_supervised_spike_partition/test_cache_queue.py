"""Focused CPU contract tests for the SW161 cache/preflight reservation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

import numpy as np  # Keep NumPy ahead of torch on the Windows workstation.
from collaborative_test.SW_0161_supervised_spike_partition import cache_queue as queue


class CacheQueueTests(unittest.TestCase):
    @staticmethod
    def _bare_queue():
        instance = object.__new__(queue.CacheQueue)
        instance.stop_event = threading.Event()
        instance.state_lock = threading.RLock()
        instance._child = None
        instance._lease = None
        instance._update = mock.Mock()
        instance._stage_update = mock.Mock()
        instance._unlock = mock.Mock()
        instance._release_lease = mock.Mock()
        return instance

    def test_import_preserves_assigned_cuda_without_initializing_device(self):
        root = queue.ROOT
        code = (
            "import os, numpy; os.environ['CUDA_VISIBLE_DEVICES']='3'; "
            "import collaborative_test.SW_0161_supervised_spike_partition.cache_queue as q; "
            "import torch; assert os.environ['CUDA_VISIBLE_DEVICES']=='3'; "
            "assert not torch.cuda.is_initialized(); print('ok')"
        )
        env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_stage_argv_routes_cache_then_disposable_preflight(self):
        cache_argv = queue.stage_command("cache")
        preflight_argv = queue.stage_command("preflight")
        self.assertEqual(cache_argv[1:4], ["-m", "collaborative_test.SW_0161_supervised_spike_partition.cache",
                                           "--device"])
        self.assertEqual(preflight_argv[1:4], ["-m", "collaborative_test.SW_0161_supervised_spike_partition.preflight",
                                               "--device"])
        self.assertIn("cuda:0", cache_argv)
        self.assertIn("cuda:0", preflight_argv)
        self.assertIn(str(queue.CACHE_ROOT), cache_argv)
        self.assertIn(str(queue.PREFLIGHT_OUTPUT), preflight_argv)
        self.assertNotIn("train", " ".join(cache_argv + preflight_argv))

    def test_live_score_waits_cpu_then_adopts_publication(self):
        event = threading.Event()
        calls = []
        def dependency():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("score owner is still live")
            return {"score_sha256": "a" * 64, "status": "complete"}
        with mock.patch.object(queue, "_score_dependency", side_effect=dependency), \
             mock.patch.object(queue, "POLL_SECONDS", 0.001):
            result = queue._wait_score(event, lambda _detail: None)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(calls), 2)

    def test_run_drives_cache_then_preflight_with_real_existing_artifact_branches(self):
        instance = self._bare_queue()
        cache_manifest = {"count": 512, "bindings_sha256": "c" * 64}
        admitted_report = {"training_admission": True}
        launches = []
        def launch(stage, digest):
            launches.append((stage, digest))
            return True

        with mock.patch.object(queue, "_wait_score", return_value={"status": "complete"}), \
             mock.patch.object(instance, "_verify_code"), \
             mock.patch.object(queue, "_validate_cache", return_value=cache_manifest), \
             mock.patch.object(instance, "_acquire_and_launch", side_effect=launch), \
             mock.patch.object(queue, "_queue_report", return_value=admitted_report), \
             mock.patch.object(queue, "sha256_file", return_value="d" * 64), \
             mock.patch.object(queue.preflight, "validate_upstream", return_value={"upstream": "ok"}), \
             mock.patch.object(Path, "is_file", return_value=False):
            result = instance.run()
        self.assertEqual(result, 0)
        self.assertEqual([stage for stage, _ in launches], ["cache", "preflight"])
        statuses = [call.args[0] for call in instance._update.call_args_list]
        self.assertIn("complete_admitted", statuses)
        instance._unlock.assert_called_once_with()

    def test_run_adopts_complete_cache_and_preflight_without_gpu_launch(self):
        instance = self._bare_queue()
        cache_manifest = {"count": 512, "bindings_sha256": "c" * 64}
        admitted_report = {"training_admission": True}
        with mock.patch.object(queue, "_wait_score", return_value={"status": "complete"}), \
             mock.patch.object(instance, "_verify_code"), \
             mock.patch.object(queue, "_validate_cache", return_value=cache_manifest), \
             mock.patch.object(instance, "_acquire_and_launch") as launch, \
             mock.patch.object(queue, "_queue_report", return_value=admitted_report), \
             mock.patch.object(queue, "sha256_file", return_value="e" * 64), \
             mock.patch.object(Path, "is_file", return_value=True):
            result = instance.run()
        self.assertEqual(result, 0)
        launch.assert_not_called()
        self.assertIn("complete_admitted", [call.args[0] for call in instance._update.call_args_list])

    def test_final_gpu_admission_rejects_uuid_owner_and_memory_changes(self):
        cases = [
            ("uuid", ("new", 2, []), "gpu_uuid_changed"),
            ("owner", ("expected", 2, [77]), "gpu_has_owners"),
            ("memory", ("expected", 513, []), "gpu_memory_over_limit"),
        ]
        for _name, sample, reason in cases:
            with self.subTest(reason=reason), mock.patch.object(
                queue.owner.gpu_guard, "_gpu_info", return_value=sample
            ):
                accepted, report = queue._final_gpu_admission(2, "expected")
            self.assertFalse(accepted)
            self.assertEqual(report["reason"], reason)
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info",
                               return_value=("expected", 10, [])):
            accepted, report = queue._final_gpu_admission(2, "expected")
        self.assertTrue(accepted)
        self.assertEqual(report["reason"], "idle")

    def test_monitor_ignores_owned_child_but_flags_foreign_gpu_process(self):
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info",
                               return_value=("gpu-uuid", 20, [101, 202])), \
             mock.patch.object(queue.owner.gpu_guard, "_process_tree",
                               return_value={101, 102}), \
             mock.patch.object(queue.owner.gpu_guard, "_foreign_owners",
                               return_value=[202]) as foreign:
            result = queue._foreign_gpu_owners(1, "gpu-uuid", 101)
        foreign.assert_called_once_with([101, 202], {101, 102})
        self.assertEqual(result, [202])

    def test_valid_scientific_rejection_is_a_terminal_receipt_not_corruption(self):
        expected = {"experiment": queue.preflight.EXPERIMENT}
        rejected = {"training_admission": False, "status": "complete"}
        with mock.patch.object(queue.preflight, "validate_upstream", return_value=expected), \
             mock.patch.object(Path, "read_text", return_value='{"status":"complete"}'), \
             mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(queue.preflight, "validate_report_record", return_value=rejected):
            with mock.patch.object(queue, "PREFLIGHT_OUTPUT", Path("report.json")), \
                 mock.patch.object(queue, "CACHE_ROOT", Path("cache")), \
                 mock.patch.object(queue, "sha256_file", return_value="b" * 64):
                record = queue._queue_report("preflight")
        self.assertIs(record["training_admission"], False)

    def test_wrong_or_malformed_preflight_receipt_fails_closed(self):
        with mock.patch.object(queue.preflight, "validate_upstream", return_value={"x": 1}), \
             mock.patch.object(Path, "read_text", return_value='{"status":"complete"}'), \
             mock.patch.object(queue.preflight, "validate_report_record",
                               side_effect=ValueError("binding mismatch")), \
             mock.patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                queue._queue_report("preflight")

    def test_existing_queue_state_is_preserved_and_never_restarted(self):
        # Exercise the same create-only guard used by the locked supervisor.
        with mock.patch.object(Path, "exists", return_value=True):
            with self.assertRaises(FileExistsError):
                queue._assert_state_unwritten(Path("prior-state.json"))

    def test_lease_cleanup_unlocks_and_closes_exact_held_lease(self):
        lease = mock.Mock()
        instance = object.__new__(queue.CacheQueue)
        instance._lease = lease
        with mock.patch.object(queue, "fcntl") as flock:
            instance._release_lease()
        flock.flock.assert_called_once_with(lease.fileno(), flock.LOCK_UN)
        lease.close.assert_called_once_with()
        self.assertIsNone(instance._lease)

    def test_dry_run_is_plan_only_and_contains_no_model_training_command(self):
        plan = queue.dry_run_plan()
        self.assertEqual(plan["stages"], ["cache", "preflight"])
        self.assertEqual(plan["gpu_candidates"], [0, 1, 2, 3])
        self.assertTrue(all("--device" in plan[f"{stage}_argv"]
                            for stage in ("cache", "preflight")))
        self.assertNotIn("--stage", plan["cache_argv"])
        self.assertNotIn("--stage", plan["preflight_argv"])


if __name__ == "__main__":
    unittest.main()
