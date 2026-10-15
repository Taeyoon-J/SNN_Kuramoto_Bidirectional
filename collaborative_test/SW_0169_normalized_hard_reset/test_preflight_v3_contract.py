from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0169_normalized_hard_reset import (
    preflight_v3, preflight_queue_v3)
from collaborative_test.SW_0155_supervised_spike_relations import objective as sw155_objective


class PreflightV3ContractTests(unittest.TestCase):
    def test_real_modal_target_api_flat_output_is_reshaped_without_relabeling(self):
        masks = torch.zeros((8, 128, 128), dtype=torch.int64)
        masks[:, :64, :64] = 3
        masks[:, 64:, 64:] = 9
        flattened = sw155_objective.modal_train_targets32(masks)
        self.assertEqual(tuple(flattened.shape), (8, 1024))
        self.assertEqual(flattened.dtype, torch.int64)
        grid = preflight_v3._modal32_grid(flattened.cpu().numpy())
        self.assertEqual(grid.shape, (8, 32, 32))
        self.assertTrue(np.array_equal(grid.reshape(8, 1024), flattened.numpy()))
        self.assertEqual(set(np.unique(grid[0]).tolist()), {0, 3, 9})

    def test_modal_grid_rejects_bad_geometry_or_noninteger_labels(self):
        with self.assertRaisesRegex(ValueError, r"\[B,1024\]"):
            preflight_v3._modal32_grid(np.zeros((8, 32, 32), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            preflight_v3._modal32_grid(np.zeros((8, 1024), dtype=np.float32))

    def test_successful_calibration_is_persisted_before_target_handoff(self):
        prepared = {
            "source_cache_manifest_sha256": "1" * 64,
            "context_cache_manifest_sha256": "2" * 64,
            "source_cache_state_sha256": "3" * 64,
            "context_cache_state_sha256": "4" * 64,
        }
        calibration_report = {
            "source_six_trace_parity": True, "source_q_parity": True,
            "source_a_parity": True, "source_cache_replay_exact": True,
            "source_model_state_sha256_before": "3" * 64,
            "source_model_state_sha256_after": "3" * 64,
            "r": [1.0] * 4,
        }
        order = []
        def fail_target_handoff(receipt):
            order.append(("targets", receipt))
            raise RuntimeError("target handoff test")

        with (mock.patch.object(torch.cuda, "is_available", return_value=True),
              mock.patch.object(torch.cuda, "set_device"),
              mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
              mock.patch.object(preflight_v3.calibration, "run_source_calibration_gpu",
                                return_value=calibration_report),
              mock.patch.object(preflight_v3, "_atomic_json",
                                side_effect=lambda path, value: order.append(("save", value))),
              mock.patch.object(preflight_v3, "_prepare_targets_fresh_cpu",
                                side_effect=fail_target_handoff)):
            with self.assertRaisesRegex(RuntimeError, "target handoff test"):
                preflight_v3.run_two_update_admission(
                    prepared, device="cuda:0", calibration_output="calibration.json")
        self.assertEqual([item[0] for item in order], ["save", "targets"])
        self.assertEqual(order[0][1]["r"], [1.0] * 4)
        self.assertEqual(order[1][1]["source_cache_manifest_sha256"], "1" * 64)

    def test_queue_and_post_gpu_recheck_are_v3_namespace_bound(self):
        command = preflight_queue_v3._cpu_revalidation_command()
        self.assertIn("calibration_v2 as c", command[2])
        self.assertIn("preflight_v3 as p", command[2])
        self.assertNotIn("preflight_v2 as p", command[2])
        worker = preflight_queue_v3._worker_command(2)
        self.assertIn("preflight_queue_v3", " ".join(worker))
        self.assertIn("source_admission_v3_20261015", str(preflight_queue_v3.REPORT))


if __name__ == "__main__":
    unittest.main()
