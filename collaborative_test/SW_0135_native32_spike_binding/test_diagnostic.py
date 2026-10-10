"""Focused CPU-only contracts for the native32 batching diagnostic and queue."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from types import SimpleNamespace
from collaborative_test.SW_0135_native32_spike_binding import diagnostic, parity, validation_queue
from collaborative_test.SW_0135_native32_spike_binding.foundation import make_criterion32


class DiagnosticTest(unittest.TestCase):
    def test_b1_b4_resource_records_are_bound_to_same_source_and_order(self):
        b1 = __import__("json").loads((HERE / "results_archive/resource_probe_seed0.json").read_text())
        foundation = SimpleNamespace(
            image_ids=b1["image_ids"],
            provenance={"source_core_sha256": b1["source_core_sha256"],
                        "source_manifest_sha256": b1["source_manifest_sha256"]})
        refs = diagnostic.validate_resource_references(foundation)
        self.assertEqual(refs["image_ids"], b1["image_ids"])
        self.assertEqual(refs["b1_sha256"], diagnostic.B1_REPORT_SHA256)

    def test_tensor_stats_reports_nonzero_finite_difference_at_zero_reference(self):
        stats = diagnostic._tensor_stats(torch.tensor([0.0, 1.0]),
                                         torch.tensor([1e-8, 1.1]))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for k, v in stats.items()
                            if isinstance(v, float)))
        self.assertAlmostEqual(stats["max_abs"], .1, places=6)
        self.assertGreater(stats["relative_l2"], 0.0)

    def test_fixed_trace_loss_reduction_and_gradient_are_image_mean(self):
        class TinyCriterion:
            def __call__(self, *, plv=None, theta=None):
                value = plv.mean() if plv is not None else theta.square().mean()
                return value, {"value": value}
        criterion = TinyCriterion()
        old_settle = diagnostic.SETTLE
        diagnostic.SETTLE = 8
        theta = torch.randn(4, 16, 16, 4, dtype=torch.float64, requires_grad=True)
        q = torch.rand(4, 16, 16, dtype=torch.float64, requires_grad=True)
        q = (q + q.transpose(1, 2)) / 2
        q = q - torch.diag_embed(torch.diagonal(q, dim1=1, dim2=2))
        loss, _ = diagnostic._loss(theta, q, criterion)
        gb = torch.autograd.grad(loss, (theta, q), retain_graph=True)
        per_t, per_q = [], []
        for i in range(4):
            ti = theta[i:i + 1]
            qi = q[i:i + 1]
            li, _ = diagnostic._loss(ti, qi, criterion)
            gt, gq = torch.autograd.grad(li / 4, (theta, q), retain_graph=True)
            per_t.append(gt[i:i + 1]); per_q.append(gq[i:i + 1])
        self.assertTrue(torch.allclose(loss, torch.stack([
            diagnostic._loss(theta[i:i+1], q[i:i+1], criterion)[0] for i in range(4)
        ]).mean(), atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(gb[0], torch.cat(per_t), atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(gb[1], torch.cat(per_q), atol=1e-10, rtol=1e-10))
        diagnostic.SETTLE = old_settle

    def test_queue_plan_and_owner_candidate_are_conservative(self):
        tasks = validation_queue.task_plan()
        self.assertEqual(len(tasks), 4)
        self.assertEqual(sum(t["kind"] == "parity" for t in tasks), 3)
        self.assertEqual(sum(t["kind"] == "diagnostic" for t in tasks), 1)
        self.assertTrue(validation_queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(validation_queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(validation_queue.gpu_is_exclusive_candidate(2, [123]))
        for task in tasks:
            self.assertIn("-m", validation_queue.command(task))

    def test_parity_validator_requires_exact_registered_evidence(self):
        task = {"kind": "parity", "seed": 1}
        fp = parity.implementation_fingerprint()
        report = {
            "experiment": "SW0135_native32_spike_binding", "stage": "native32_zero_adapter_parity",
            "status": "native32_zero_adapter_parity_passed", "seed": 1,
            "ground_truth_used": False, "masks_read": False, "optimizer_updates": 0,
            "training_admission": False, "implementation_fingerprint": fp,
            "image_ids": [1320, 1321, 1322, 1323], "microbatch_size": 4,
            "horizons": [{"steps": 64, "settle": 32}, {"steps": 1024, "settle": 512}],
            "arms": {arm: [{"steps": steps, "settle": settle, "exact_parity": True}
                           for steps, settle in ((64, 32), (1024, 512))]
                     for arm in ("phase", "constant")},
        }
        self.assertTrue(validation_queue.validate_record(report, task, fp))
        report["arms"]["phase"][1]["exact_parity"] = False
        self.assertFalse(validation_queue.validate_record(report, task, fp))

    def test_diagnostic_validator_accepts_descriptive_differences(self):
        task = {"kind": "diagnostic", "seed": 0}
        fp = diagnostic.implementation_fingerprint()
        report = {
            "experiment": "SW0135_native32_spike_binding", "stage": "batching_diagnostic",
            "status": "diagnostic_complete", "seed": 0,
            "ground_truth_used": False, "masks_read": False, "optimizer_updates": 0,
            "training_admission": False, "implementation_fingerprint": fp,
            "image_ids": [1320, 1321, 1322, 1323], "batch_size": 4,
            "steps": 1024, "settle": 512, "live_tail_steps": 64,
            "registered_reduction": {"batch_vs_mean_loss_abs": .02},
            "true_rollout_comparison": {"old_loss_b4": 1., "old_loss_mean_b1": 1.1,
                                         "old_loss_abs_difference": .1,
                                         "gamma_b4_vs_independent_b1": {"max_abs": .4},
                                         "prepared_graph_b4_vs_b1": {"max_abs": .2},
                                         "q_b4_vs_concatenated_b1": {"max_abs": .1},
                                         "old_gradient_b4_vs_accumulated_b1": {
                                             "max_abs": .3, "relative_l2": .4, "cosine": .99}},
        }
        self.assertTrue(validation_queue.validate_record(report, task, fp))
        report["ground_truth_used"] = True
        self.assertFalse(validation_queue.validate_record(report, task, fp))


if __name__ == "__main__":
    unittest.main()
