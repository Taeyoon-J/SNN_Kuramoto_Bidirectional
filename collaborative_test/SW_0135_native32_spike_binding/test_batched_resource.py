from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import uuid

# Keep these CPU-only checks from creating a CUDA context on shared GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0123_adaptive_temporal_assignment import dispatcher as owner
from collaborative_test.SW_0135_native32_spike_binding import batched_resource_queue as queue
from collaborative_test.SW_0135_native32_spike_binding.batched_resource_probe import (
    _accumulate, _install_average_grad,
)


class BatchedResourceTests(unittest.TestCase):
    def test_queue_fingerprint_binds_batched_queue_source(self):
        self.assertEqual(queue.queue_source_sha256(), queue._sha(Path(queue.__file__)))
        self.assertNotEqual(queue.queue_source_sha256(),
                            queue._sha(HERE / "resource_queue.py"))

    def test_four_microbatches_match_logical_mean_gradients_and_one_adam_step(self):
        torch.manual_seed(13544)
        x = torch.tensor([-.7, .2, .9, 1.3, -.1, .8, 1.7, -1.2])
        target_old = torch.linspace(-.2, .3, 8)
        target_rgb = torch.linspace(.8, -.4, 8)
        def accumulate(width):
            p = nn.Parameter(torch.tensor(.45))
            h = nn.Parameter(torch.tensor(-.15))
            old_sum, rgb_joint_sum, rgb_head_sum = [None], [None], [None]
            old_total = rgb_total = 0.0
            for start in range(0, x.numel(), width):
                stop = min(start + width, x.numel())
                xb = x[start:stop]
                old = (p * xb - target_old[start:stop]).square().mean()
                rgb = (p * xb + h - target_rgb[start:stop]).square().mean()
                count = xb.numel()
                old_total += float(old.detach()) * count
                rgb_total += float(rgb.detach()) * count
                _accumulate(old * count, [p], old_sum, retain_graph=True)
                _accumulate(rgb * count, [p], rgb_joint_sum, retain_graph=True)
                _accumulate(rgb * count, [h], rgb_head_sum, retain_graph=False)
            n = x.numel()
            return (p, h, (old_sum[0] + rgb_joint_sum[0]) / n,
                    rgb_head_sum[0] / n, old_total / n, rgb_total / n)

        p_b1, h_b1, joint_b1, head_b1, old_b1, rgb_b1 = accumulate(1)
        p_b4, h_b4, joint_b4, head_b4, old_b4, rgb_b4 = accumulate(4)
        p_full = nn.Parameter(torch.tensor(.45))
        h_full = nn.Parameter(torch.tensor(-.15))
        old_full = (p_full * x - target_old).square().mean()
        rgb_full = (p_full * x + h_full - target_rgb).square().mean()
        expected_joint, expected_head = torch.autograd.grad(
            old_full + rgb_full, (p_full, h_full))
        for joint, head in ((joint_b1, head_b1), (joint_b4, head_b4)):
            self.assertTrue(torch.allclose(joint, expected_joint, atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.allclose(head, expected_head, atol=1e-7, rtol=1e-6))
        for loss in (old_b1, old_b4):
            self.assertAlmostEqual(loss, float(old_full.detach()), places=7)
        for loss in (rgb_b1, rgb_b4):
            self.assertAlmostEqual(loss, float(rgb_full.detach()), places=7)
        self.assertTrue(torch.allclose(joint_b1, joint_b4, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(head_b1, head_b4, atol=1e-7, rtol=1e-6))

        reference_p, reference_h = nn.Parameter(torch.tensor(.45)), nn.Parameter(torch.tensor(-.15))
        ref_p_opt = torch.optim.Adam([reference_p], lr=1e-2)
        ref_h_opt = torch.optim.Adam([reference_h], lr=1e-2)
        _install_average_grad([reference_p], [expected_joint], 1.)
        _install_average_grad([reference_h], [expected_head], 1.)
        torch.nn.utils.clip_grad_norm_([reference_p], 1.)
        torch.nn.utils.clip_grad_norm_([reference_h], 1.)
        ref_p_opt.step(); ref_h_opt.step()
        self.assertEqual(int(ref_p_opt.state[reference_p]["step"]), 1)
        self.assertEqual(int(ref_h_opt.state[reference_h]["step"]), 1)
        for p, h, joint_grad, head_grad in (
                (p_b1, h_b1, joint_b1, head_b1), (p_b4, h_b4, joint_b4, head_b4)):
            opt_p = torch.optim.Adam([p], lr=1e-2)
            opt_h = torch.optim.Adam([h], lr=1e-2)
            _install_average_grad([p], [joint_grad], 1.)
            _install_average_grad([h], [head_grad], 1.)
            torch.nn.utils.clip_grad_norm_([p], 1.)
            torch.nn.utils.clip_grad_norm_([h], 1.)
            opt_p.step(); opt_h.step()
            self.assertEqual(int(opt_p.state[p]["step"]), 1)
            self.assertEqual(int(opt_h.state[h]["step"]), 1)
            self.assertNotEqual(float(p.detach()), .45)
            self.assertNotEqual(float(h.detach()), -.15)
            self.assertTrue(torch.allclose(p, reference_p, atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.allclose(h, reference_h, atol=1e-7, rtol=1e-6))

    def test_batched_record_and_paths_are_distinct_and_create_once(self):
        task, = queue.task_plan()
        argv = queue.command(task)
        self.assertIn("-m", argv)
        self.assertEqual(argv[argv.index("-m") + 1],
                         "collaborative_test.SW_0135_native32_spike_binding.batched_resource_probe")
        self.assertNotEqual(queue.result_path(task),
                            queue.ARCHIVE / "resource_probe_seed0.json")
        tmp = HERE / f"tmp_batched_resource_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        record_path = tmp / "probe.json"
        try:
            record = {
                "experiment": "SW0135_native32_spike_binding",
                "status": "batched_resource_probe_complete", "resource_only": True,
                "training_admission": False, "ground_truth_used": False,
                "optimizer_updates": 0,
                "resource_optimizer_steps": {"joint": 1, "head_decoder": 1},
                "image_id_count": 16, "microbatch_size": 4,
                "microbatch_count": 4, "microbatch_seconds": [.1] * 4,
                "logical_update_seconds": .5,
                "b1_comparison": {"sha256": queue.B1_REPORT_SHA256},
            }
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertTrue(queue.validate_probe_record(record_path))
            record["microbatch_count"] = 16
            record_path.write_text(json.dumps(record), encoding="utf-8")
            self.assertFalse(queue.validate_probe_record(record_path))
        finally:
            if record_path.exists():
                record_path.unlink()
            tmp.rmdir()

    def test_resource_queue_retains_exclusive_owner_and_memory_contract(self):
        self.assertTrue(queue.gpu_is_exclusive_candidate(512, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(513, []))
        self.assertFalse(queue.gpu_is_exclusive_candidate(8, [24680]))
        with mock.patch.object(owner, "_pid_is_confirmed_gone", side_effect=lambda pid: pid == 2):
            self.assertEqual(queue.foreign_owners([1, 2, 3], {1}), [3])


if __name__ == "__main__":
    unittest.main()
