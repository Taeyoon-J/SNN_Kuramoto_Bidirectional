from __future__ import annotations

import json
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from collaborative_test.SW_0155_supervised_spike_relations import preflight_queue as queue


class _Lease:
    def __init__(self, fd):
        self._fd, self.closed, self.unlocked = fd, False, False

    def fileno(self):
        return self._fd

    def close(self):
        self.closed = True


class _Child:
    pid = 78321

    def __init__(self, output, returncode=0):
        self.returncode = returncode
        self.output = output

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class SW0155PreflightQueueTests(unittest.TestCase):
    def test_plan_is_exact_single_actual_source_preflight(self):
        task = queue.task_plan()
        argv = queue.command(task)
        self.assertEqual(task["stage"], "actual_source_masked_two_update_preflight")
        self.assertEqual(task["seed"], 1)
        self.assertEqual(argv[1:3], ["-m", "collaborative_test.SW_0155_supervised_spike_relations.preflight"])
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        with self.assertRaises(ValueError):
            queue.command(dict(task, seed=0))

    def test_report_validation_delegates_to_strict_admission_and_binds_fp(self):
        report = {"implementation_fingerprint": {"train.py": "a" * 64}}
        with mock.patch.object(queue.Path, "read_text", return_value=json.dumps(report)), \
             mock.patch.object(queue, "report_fingerprint", return_value=report["implementation_fingerprint"]), \
             mock.patch.object(queue.preflight, "validate_admission", return_value=True) as admitted, \
             mock.patch.object(queue.train, "source_contract", return_value={"dummy": True}), \
             mock.patch.object(queue.sw130, "validate_rgb_assets", return_value={"asset": "bound"}), \
             mock.patch.object(queue.train, "HDF5_PATH", queue.HERE / "README.md"):
            self.assertTrue(queue.validate_report(queue.OUTPUT))
            admitted.assert_called_once()
            self.assertFalse(queue.validate_report(queue.OUTPUT,
                expected_fingerprint={"train.py": "b" * 64}))

    def test_unstable_lease_releases_then_stable_lease_launches_once(self):
        leases = [_Lease(101), _Lease(102)]
        instance = object.__new__(queue.PreflightQueue)
        instance.existing_complete = False
        instance.stop_event = threading.Event()
        instance.task = queue.task_plan()
        instance.report_fp = {"train.py": "c" * 64}
        instance._verify_fingerprints = lambda: None
        instance._update = lambda *args, **kwargs: None
        def fake_popen(argv, **kwargs):
            self.assertEqual(argv[argv.index("--output") + 1], str(queue.OUTPUT))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertIn(leases[1].fileno(), kwargs["pass_fds"])
            return _Child(queue.OUTPUT)
        with mock.patch.object(queue, "_check_unwritten"), \
             mock.patch.object(queue.Path, "open", return_value=mock.MagicMock()), \
             mock.patch.object(queue, "fcntl", SimpleNamespace(
                 LOCK_EX=1, LOCK_NB=2, LOCK_UN=4,
                 flock=mock.Mock())), \
             mock.patch.object(queue.owner, "_acquire", side_effect=[
                (2, "GPU-2", 3, leases[0]), (2, "GPU-2", 4, leases[1])]) as acquire, \
             mock.patch.object(queue.owner, "_stable_idle", side_effect=[False, True]), \
             mock.patch.object(queue, "validate_report", return_value=True), \
             mock.patch.object(queue.subprocess, "Popen", side_effect=fake_popen):
            result = instance.launch_once()
        self.assertEqual(result, "passed")
        self.assertEqual(acquire.call_count, 2)
        self.assertTrue(leases[0].closed)
        self.assertTrue(leases[1].closed)


if __name__ == "__main__":
    unittest.main()
