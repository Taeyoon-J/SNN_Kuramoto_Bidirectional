import json
import unittest
import uuid
import hashlib
from pathlib import Path

import numpy as np  # initialize shared OpenMP runtime before torch on Windows

from collaborative_test.SW_0126_history_event_binding import screen_queue


class ScreenQueueContractTests(unittest.TestCase):
    def test_registered_screen_plan_has_one_unique_output_per_source_seed(self):
        tasks = screen_queue.task_plan()
        self.assertEqual([task["seed"] for task in tasks], [0, 1, 2])
        self.assertEqual(len({task["output"] for task in tasks}), 3)
        self.assertTrue(all(task["task_id"] == f"sw0126_screen_seed{task['seed']}"
                            for task in tasks))

    def test_gpu_slot_requires_both_low_memory_and_no_compute_owner(self):
        self.assertTrue(screen_queue.gpu_is_exclusive(512, []))
        self.assertFalse(screen_queue.gpu_is_exclusive(513, []))
        self.assertFalse(screen_queue.gpu_is_exclusive(2, [12345]))

    def test_result_adoption_rejects_incomplete_or_non_screen_artifacts(self):
        task = {"seed": 1}
        ids = list(range(1000, 1064))
        gradient = {"assignment_norm": 1.0, "beta_norm": 1.0,
                    "beta_component_abs": [1.0] * 4,
                    "membrane_tau_m_norm": 1.0, "dendritic_norm": 1.0}
        gate_gradient = {"assignment_norm": 1.0, "beta_norm": 0.0,
                         "membrane_tau_m_norm": 0.0, "dendritic_norm": 0.0}
        batches = [{"theta_carrier_gate_exact": {"theta": True, "carrier": True, "gate": True},
                    "activity": {"mixed_unit_pass_by_image_component": [[True] * 4 for _ in range(16)]},
                    "centered_event_trace_variance_by_component": [0.1] * 4,
                    "arms": {"history_event": {"gradient_credit": gradient,
                                               "assignment_patch_mass_max_error": 0.0},
                             "gate_only": {"gradient_credit": gate_gradient}}}
                   for _ in range(4)]
        batches[0]["native_source_rollout_exact_first_batch"] = {
            "theta": True, "component_membrane": True, "component_spikes": True}
        record = {"status": "passed_feasibility_screen", "experiment": "SW0126",
                  "seed": 1, "optimizer_updates": 0,
                  "ground_truth_used_for_prediction_or_training": False,
                  "implementation_fingerprint": {"screen.py": "fp"},
                  "source_core_sha256": "source-sha",
                  "source_ids_sha256": hashlib.sha256(
                      __import__("numpy").asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                  "matched_train_ids": ids,
                  "activity_thresholds": {"component_occupancy": [0.02, 0.98],
                      "minimum_mixed_unit_fraction_per_image_component": 0.10},
                  "aggregate_activity": {"pass": True}, "batches": batches}
        args = (record, task, ids, "source-sha", {"screen.py": "fp"})
        self.assertTrue(screen_queue.validate_screen_record(*args))
        record["batches"][2]["activity"]["mixed_unit_pass_by_image_component"][7][1] = False
        self.assertFalse(screen_queue.validate_screen_record(*args))
        record["batches"][2]["activity"]["mixed_unit_pass_by_image_component"][7][1] = True
        record["optimizer_updates"] = 1
        self.assertFalse(screen_queue.validate_screen_record(*args))


if __name__ == "__main__":
    unittest.main()
