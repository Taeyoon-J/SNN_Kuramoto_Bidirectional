"""Command and process-state contracts for the SW0168 owner queue."""
from __future__ import annotations

import os
import unittest

from collaborative_test.SW_0168_end_to_end_connectivity import train, training_queue


class TrainingQueueContractTests(unittest.TestCase):
    def test_worker_command_is_exact_module_and_registered_arm(self):
        command = training_queue._worker_command("constrained_maximin", 3)
        self.assertEqual(command[1:3], ["-m", "collaborative_test.SW_0168_end_to_end_connectivity.training_queue"])
        self.assertEqual(command[3:], ["--worker", "--arm", "constrained_maximin",
                                       "--physical-gpu", "3"])
        with self.assertRaises(ValueError):
            training_queue._worker_command("other", 0)

    def test_per_arm_runtime_and_parent_cuda_hiding(self):
        self.assertEqual(training_queue.RUNTIME.name, "training_queue_20261015")
        self.assertNotEqual(training_queue._state_path(train.ARMS[0]),
                            training_queue._state_path(train.ARMS[1]))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")

    def test_queue_yields_only_for_foreign_gpu_identity_or_owner(self):
        self.assertFalse(training_queue._foreign_owner_changed("GPU-a", "GPU-a", [123], 123))
        self.assertTrue(training_queue._foreign_owner_changed("GPU-a", "GPU-a", [123, 456], 123))
        self.assertTrue(training_queue._foreign_owner_changed("GPU-a", "GPU-b", [123], 123))


if __name__ == "__main__":
    unittest.main()
