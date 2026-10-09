"""CPU-only checks for the exclusive-GPU evaluation queue contract."""
import numpy as np  # Keep NumPy before Torch on the Windows development host.
import unittest
from unittest import mock

from collaborative_test.SW_0124_temporal_prototype_readout import coordinator, dispatcher


class DispatcherContractTests(unittest.TestCase):
    def test_queue_orders_three_source_preflights_before_single_evaluation(self):
        tasks = coordinator.task_plan("/tmp/sw0124-contract-test")
        self.assertEqual([task["stage"] for task in tasks],
                         ["preflight", "preflight", "preflight", "evaluate"])
        self.assertEqual([task["seed"] for task in tasks[:3]], [0, 1, 2])
        self.assertEqual(tasks[3]["depends_on"], [task["task_id"] for task in tasks[:3]])
        self.assertIn("--preflight-dir", coordinator.command(tasks[3]))

    def test_gpu_free_gate_requires_low_memory_and_no_compute_owner(self):
        self.assertTrue(dispatcher.gpu_is_exclusive(512, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(513, []))
        self.assertFalse(dispatcher.gpu_is_exclusive(10, [123]))

    def test_only_owned_tree_or_confirmed_vanished_pids_are_ignored(self):
        with mock.patch.object(dispatcher, "_pid_is_confirmed_gone",
                               side_effect=lambda pid: pid == 22):
            self.assertEqual(dispatcher.foreign_owner_pids([11, 22, 33], [11, 12]), [33])
            # A live descendant is part of the current owned evaluator process tree.
            self.assertEqual(dispatcher.foreign_owner_pids([12], [11, 12]), [])


if __name__ == "__main__":
    unittest.main()
