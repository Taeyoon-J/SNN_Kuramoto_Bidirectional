import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0111_frozen_partition_rgb"))
import coordinator
import run
from SW_0106_spike_partition_rgb.partition_rgb import SharedRGBDecoder, reconstruct_one
from summarize import _bootstrap_lower


class RunnerContractTests(unittest.TestCase):
    def test_registered_selection_is_ordered_unique_and_maps_to_global_ids(self):
        for seed in run.SEEDS:
            pool, ids = run.pool_indices(seed)
            self.assertEqual(pool.shape, (4096,))
            self.assertEqual(ids.shape, (4096,))
            self.assertEqual(len(set(ids.tolist())), 4096)
            np.testing.assert_array_equal(ids, np.where(pool < 1000, pool, pool + 640))

    def test_candidate_reconstruction_has_assignment_gradient_control_does_not(self):
        torch.manual_seed(41)
        q = torch.rand(256, 256, requires_grad=True)
        hard = torch.zeros(256, 2)
        hard[:128, 0] = 1
        hard[128:, 1] = 1
        features = torch.randn(256, 8)
        target = torch.rand(256, 3)
        decoder = SharedRGBDecoder()
        candidate_pred, candidate_loss, _ = reconstruct_one(q, hard, features, target, decoder,
                                                             assignment_credit=True)
        control_pred, control_loss, _ = reconstruct_one(q.detach(), hard, features, target, decoder,
                                                         assignment_credit=False)
        self.assertTrue(torch.equal(candidate_pred, control_pred))
        self.assertTrue(torch.equal(candidate_loss, control_loss))
        grad, = torch.autograd.grad(candidate_loss, q)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        control_grad, = torch.autograd.grad(control_loss, q, allow_unused=True)
        self.assertIsNone(control_grad)

    def test_task_graph_keeps_seed0_lambda_before_other_preflights(self):
        tasks = coordinator.task_plan()
        by_id = {t["task_id"]: t for t in tasks}
        self.assertEqual(len(tasks), 13)
        self.assertEqual(by_id["sw0111_preflight_s0"]["depends_on"], [])
        self.assertEqual(by_id["sw0111_preflight_s1"]["depends_on"], ["sw0111_preflight_s0"])
        self.assertEqual(by_id["sw0111_preflight_s2"]["depends_on"], ["sw0111_preflight_s0"])
        for seed in run.SEEDS:
            self.assertEqual(set(by_id[f"sw0111_evaluate_s{seed}"]["depends_on"]),
                             {f"sw0111_train_control_s{seed}", f"sw0111_train_candidate_s{seed}"})

    def test_fixed_seed_bootstrap_is_reproducible(self):
        rows = [np.linspace(-.2, .2, 320) + seed * .001 for seed in range(3)]
        first = _bootstrap_lower(rows)
        second = _bootstrap_lower(rows)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)


if __name__ == "__main__":
    unittest.main()
