"""Admission failures must be caught before touching CUDA or launching a probe."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from collaborative_test.SW_0173_gradient_fidelity import resource_queue as queue


class ResourceQueueTests(unittest.TestCase):
    def test_wrong_physical_visibility_never_imports_probe(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
             patch.object(queue, "load_probe") as load:
            with self.assertRaisesRegex(RuntimeError, "visibility"):
                queue.run_worker(1, Path("unused.json"))
            load.assert_not_called()

    def test_cpu_prepare_cannot_change_leased_gpu(self):
        cuda = SimpleNamespace(is_initialized=Mock(return_value=False), device_count=Mock())
        probe = SimpleNamespace(torch=SimpleNamespace(cuda=cuda), run_probe=Mock())
        def prepare():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return {}
        probe.prepare_cpu = prepare
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
             patch.object(queue, "load_probe", return_value=probe):
            with self.assertRaisesRegex(RuntimeError, "changed physical visibility"):
                queue.run_worker(1, Path("unused.json"))
        cuda.device_count.assert_not_called()
        probe.run_probe.assert_not_called()

    def test_worker_command_keeps_canonical_module_identity(self):
        command = queue.worker_command(3, Path("report.json"))
        self.assertIn("collaborative_test.SW_0173_gradient_fidelity.resource_queue", command)
        self.assertNotIn("__main__", command)
        self.assertEqual(command[-3:], ["3", "--output", "report.json"])


if __name__ == "__main__":
    unittest.main()
