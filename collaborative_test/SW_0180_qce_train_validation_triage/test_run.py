import unittest

import numpy as np
import torch

from collaborative_test.SW_0180_qce_train_validation_triage import run
from collaborative_test.SW_0179_qce_seed_replication import train as train179


class SW0180ReplayContractTests(unittest.TestCase):
    def test_batched_loss_and_margin_match_registered_vectorized_qce(self):
        torch.manual_seed(1801)
        q = torch.rand(3, 7, 7, dtype=torch.float64, requires_grad=True)
        q = (q + q.transpose(-1, -2)) * 0.5
        q.retain_grad()
        # Includes background zero, non-contiguous instance IDs, and a singleton
        # class which remains a negative prototype but is not a query.
        labels = torch.tensor([
            [0, 0, 3, 3, 9, 9, 22],
            [0, 4, 4, 11, 11, 11, 22],
            [5, 5, 5, 5, 5, 5, 5],
        ], dtype=torch.long)
        actual, margins = run._class_loss_and_margin(q, labels)
        expected = torch.stack([
            train179._vectorized_class_loss(q[i], labels[i], temperature=0.1)
            for i in range(3)
        ])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        self.assertEqual(tuple(margins.shape), (3,))
        self.assertTrue(torch.isfinite(margins).all())
        ga = torch.autograd.grad(actual.sum(), q, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.sum(), q)[0]
        self.assertTrue(torch.allclose(ga, ge, atol=1e-11, rtol=1e-11))
        self.assertEqual(float(actual[2].detach()), 0.0)
        self.assertEqual(float(margins[2].detach()), 0.0)

    def test_metric_report_is_per_image_finite_and_json_safe(self):
        gt = np.asarray([
            [[0, 0, 2], [2, 3, 3]],
            [[0, 4, 4], [5, 5, 5]],
        ], dtype=np.int32)
        pred = np.asarray([
            [[0, 0, 7], [7, 8, 8]],
            [[0, 9, 9], [10, 10, 10]],
        ], dtype=np.int32)
        metrics = run._metric_arrays(pred, gt)
        self.assertEqual(set(metrics), {
            "fg_ari", "foreground_iou", "matched_object_iou",
            "foreground_precision", "foreground_recall"})
        self.assertTrue(all(len(row) == 2 for row in metrics.values()))
        self.assertTrue(all(np.isfinite(row).all() for row in metrics.values()))

    def test_paired_summary_assembles_two_b4_batches_and_serializes(self):
        gt = np.zeros((8, 32, 32), dtype=np.int32)
        for index in range(8):
            gt[index, 4:28, 5:27] = 1 + index % 2
        pred = gt.copy()
        pred[:, 4:28, 5:27] = np.where(gt[:, 4:28, 5:27] == 1, 7, 8)

        def rows(offset):
            output = []
            for batch in range(2):
                sl = slice(batch * 4, batch * 4 + 4)
                qcc = np.where(pred[sl] > 0, 1, 0).astype(np.int32)
                output.append({"q_loss": [offset + 0.1] * 4,
                    "true_minus_best_other_margin": [offset + 0.2] * 4,
                    "outputs": {"conditional": pred[sl].copy(), "standalone": pred[sl].copy()},
                    "qcc": qcc, "gain": [7.5] * 4})
            return output

        summary = run._summarize_pair(rows(0.0), rows(-0.1), gt)
        self.assertEqual(len(summary["qce_per_image"]["fresh"]), 8)
        self.assertEqual(summary["qce_delta_mean"], -0.1)
        self.assertEqual(len(summary["readout_metrics"]["fresh"]["standalone"]["fg_ari"]), 8)
        import json
        json.dumps(summary, allow_nan=False)
        self.assertIn("mixed or tied", summary["triage_interpretation"])

    def test_cache_contract_requires_exact_train_geometry(self):
        rgb = np.zeros((512, 128, 128, 3), dtype=np.uint8)
        modal = np.zeros((512, 32, 32), dtype=np.int32)
        ids = list(range(1000, 1512))
        checked = run._assert_cache({"arrays": {"rgb": rgb, "modal_gt": modal},
                                     "ordered_train_ids": ids})
        self.assertEqual(checked["ids"][256:264], ids[256:264])
        with self.assertRaises(ValueError):
            run._assert_cache({"arrays": {"rgb": rgb[:511], "modal_gt": modal[:511]},
                               "ordered_train_ids": ids[:511]})

    def test_diagnostic_recipe_is_exact_and_has_no_updates(self):
        fp = run.implementation_fingerprint()
        self.assertEqual(fp["positions"], list(range(256, 264)))
        self.assertEqual(fp["seeds"], [0, 1, 2])
        self.assertEqual((fp["steps"], fp["settle"], fp["batch"]), (1024, 512, 4))
        self.assertEqual(fp["optimizer_updates"], 0)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
