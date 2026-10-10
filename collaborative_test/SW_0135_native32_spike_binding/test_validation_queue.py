"""Queue scheduling contract tests; no GPU or remote work is performed."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0135_native32_spike_binding import validation_queue as queue


class ValidationQueueTest(unittest.TestCase):
    def test_four_unique_create_once_outputs_and_no_training_stage(self):
        tasks = queue.task_plan()
        paths = [queue.result_path(task) for task in tasks]
        self.assertEqual(len(set(paths)), 4)
        self.assertTrue(all(path.parent == queue.ARCHIVE for path in paths))
        self.assertEqual([task["seed"] for task in tasks], [0, 1, 2, 0])
        self.assertTrue(all("preflight" not in queue.command(task) for task in tasks))
        self.assertEqual(queue.MAX_PARALLEL, 3)

    def test_diagnostic_completion_validator_is_finite_but_not_equivalence_gate(self):
        task = {"kind": "diagnostic", "seed": 0}
        from collaborative_test.SW_0135_native32_spike_binding import diagnostic
        fp = diagnostic.implementation_fingerprint()
        report = {
            "experiment": "SW0135_native32_spike_binding", "stage": "batching_diagnostic",
            "status": "diagnostic_complete", "seed": 0,
            "ground_truth_used": False, "masks_read": False,
            "optimizer_updates": 0, "training_admission": False,
            "implementation_fingerprint": fp, "image_ids": [1320, 1321, 1322, 1323],
            "batch_size": 4, "steps": 1024, "settle": 512, "live_tail_steps": 64,
            "registered_reduction": {"batch_vs_mean_loss_abs": .7},
            "true_rollout_comparison": {
                "old_loss_b4": 1., "old_loss_mean_b1": 4., "old_loss_abs_difference": 3.,
                "gamma_b4_vs_independent_b1": {"max_abs": 1.},
                "prepared_graph_b4_vs_b1": {"max_abs": 2.},
                "q_b4_vs_concatenated_b1": {"max_abs": .9},
                "old_gradient_b4_vs_accumulated_b1": {
                    "max_abs": 1., "relative_l2": 1.2, "cosine": -.2}},
        }
        self.assertTrue(queue.validate_record(report, task, fp))
        report["true_rollout_comparison"]["old_gradient_b4_vs_accumulated_b1"]["cosine"] = float("nan")
        self.assertFalse(queue.validate_record(report, task, fp))


if __name__ == "__main__":
    unittest.main()
