from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for item in (str(ROOT), str(ROOT / "collaborative_test"), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from collaborative_test.SW_0135_native32_spike_binding.resource_probe import (
    _accumulate, _install_average_grad, _named_joint_parameters,
    _scaled_family_norm, _unique_parameters, run_probe,
)


class ResourceProbeTests(unittest.TestCase):
    def test_microbatch_vectors_equal_batch_mean_and_adam_steps_once(self):
        torch.manual_seed(13560)
        joint = nn.Parameter(torch.tensor(0.7))
        head = nn.Parameter(torch.tensor(-0.2))
        joint_parameters = _unique_parameters([joint], "joint")
        head_parameters = _unique_parameters([head], "head")
        old_accum = [None]
        rgb_joint_accum = [None]
        rgb_head_accum = [None]
        xs = (torch.tensor(0.5), torch.tensor(1.7))
        old_losses, rgb_losses = [], []

        for x in xs:
            old = (joint * x - 0.4).square()
            rgb = (joint * x + head - 1.1).square()
            old_losses.append(old)
            rgb_losses.append(rgb)
            _accumulate(old, joint_parameters, old_accum, retain_graph=True)
            _accumulate(rgb, joint_parameters, rgb_joint_accum, retain_graph=True)
            _accumulate(rgb, head_parameters, rgb_head_accum, retain_graph=False)

        mean = len(xs)
        batch_old = torch.stack([(joint * x - .4).square() for x in xs]).mean()
        batch_rgb = torch.stack([(joint * x + head - 1.1).square() for x in xs]).mean()
        expected_joint = torch.autograd.grad(batch_old + batch_rgb, joint, retain_graph=True)[0]
        expected_head = torch.autograd.grad(batch_rgb, head)[0]
        accumulated_joint = (old_accum[0] + rgb_joint_accum[0]) / mean
        accumulated_head = rgb_head_accum[0] / mean
        self.assertTrue(torch.allclose(accumulated_joint, expected_joint, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.allclose(accumulated_head, expected_head, atol=1e-7, rtol=1e-6))
        self.assertGreater(float((rgb_head_accum[0] / mean -
                                  (old_accum[0] + rgb_head_accum[0]) / mean).abs()), 0.)

        joint_optimizer = torch.optim.Adam([joint], lr=1e-2)
        head_optimizer = torch.optim.Adam([head], lr=1e-2)
        _install_average_grad(joint_parameters, [old_accum[0] + rgb_joint_accum[0]], 1. / mean)
        _install_average_grad(head_parameters, rgb_head_accum, 1. / mean)
        joint_optimizer.step()
        head_optimizer.step()
        self.assertEqual(int(joint_optimizer.state[joint]["step"]), 1)
        self.assertEqual(int(head_optimizer.state[head]["step"]), 1)
        self.assertNotEqual(float(joint.detach()), .7)
        self.assertNotEqual(float(head.detach()), -.2)

    def test_parameter_union_rejects_duplicate_optimizer_membership(self):
        parameter = nn.Parameter(torch.ones(()))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _unique_parameters([parameter, parameter], "joint")

    def test_joint_autograd_order_stays_named_when_optimizer_groups_are_reordered(self):
        integration = nn.Parameter(torch.tensor(1.0))
        native = nn.Parameter(torch.tensor(2.0))
        named = [("core.a_d", integration), ("core.core.kuramoto.weight", native)]
        groups = [{"params": [native]}, {"params": [integration]}]
        actual_names, actual_parameters = _named_joint_parameters(named, groups)
        self.assertEqual([name for name, _ in actual_names],
                         ["core.a_d", "core.core.kuramoto.weight"])
        self.assertIs(actual_parameters[0], integration)
        self.assertIs(actual_parameters[1], native)
        family_norms = _scaled_family_norm(
            actual_names, [torch.tensor(3.0), torch.tensor(4.0)], .5)
        self.assertEqual(family_norms, {"integration_a_d": 1.5, "kuramoto": 2.0})

    def test_resource_probe_refuses_cpu_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            run_probe(seed=0, device="cpu", output=HERE / "must_not_be_created.json")
        self.assertFalse((HERE / "must_not_be_created.json").exists())


if __name__ == "__main__":
    unittest.main()
