"""Exact-value and late-credit contract tests for SW0125's isolated helper."""
import numpy as np  # Import before Torch on the Windows development host.
import unittest

import torch

from collaborative_test.SW_0110_xy_graph_route import run as base
from collaborative_test.SW_0125_late_rollout_credit.late_rollout import late_rollout


class LateRolloutTests(unittest.TestCase):
    @staticmethod
    def core(steps):
        core = base.make_core("cpu", steps=steps).eval()
        core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.shape[0])]
        return core

    def test_no_grad_helper_matches_production_full_rollout(self):
        torch.manual_seed(125)
        core = self.core(8)
        gamma = torch.randn(1, 8, 256)
        with torch.no_grad():
            _groups, spikes, membrane, theta = core(
                gamma, return_core_out=True, return_theta=True, num_time_steps=8
            )
            production_components = core.last_component_spikes.clone()
            production_component_membrane = core.last_component_out.clone()
        replay = late_rollout(core, gamma, total_steps=8, live_tail_steps=0)
        self.assertTrue(torch.equal(replay["component_spikes"], production_components))
        self.assertTrue(torch.equal(replay["component_membrane"], production_component_membrane))
        self.assertTrue(torch.equal(replay["spikes"], spikes))
        self.assertTrue(torch.equal(replay["membrane"], membrane))
        self.assertTrue(torch.equal(replay["theta"], theta))

    def test_boundary_preserves_delayed_values_and_live_tail_parameter_credit(self):
        torch.manual_seed(126)
        core = self.core(8)
        gamma = torch.randn(1, 8, 256)
        full = late_rollout(core, gamma, total_steps=8, live_tail_steps=0)
        # The helper explicitly re-enables only its live setup/tail even when
        # its caller is in a no-grad evaluation context.
        with torch.no_grad():
            split = late_rollout(core, gamma, total_steps=8, live_tail_steps=4)
        for key in ("component_spikes", "component_membrane", "theta"):
            self.assertTrue(torch.equal(split[key], full[key]), key)
        self.assertEqual(split["prefix_steps"], 4)
        self.assertEqual(split["live_tail_steps"], 4)
        # The phase history immediately across the two-step delayed gate is
        # present and value-identical at the detached/live boundary.
        self.assertTrue(torch.equal(split["theta"][:, 2:4], full["theta"][:, 2:4]))
        loss = split["component_membrane"][..., 4:].square().mean()
        loss.backward()
        expected = ("gamma_channel_proj.weight", "dendric_layer.tau_n", "membrane_layer.tau_m")
        for name in expected:
            parameter = dict(core.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        self.assertTrue(np.isfinite(float(loss.detach())))

    def test_rejects_nonstatic_and_unsupported_coupling(self):
        core = self.core(8)
        gamma = torch.zeros(1, 8, 256)
        core.gamma_drive_mode = "sequence"
        with self.assertRaisesRegex(ValueError, "static-gamma"):
            late_rollout(core, gamma, total_steps=8, live_tail_steps=4)
        core.gamma_drive_mode = "static"
        core.kuramoto.spike_pulse_gain = 0.0
        with self.assertRaisesRegex(ValueError, "spike-pulse"):
            late_rollout(core, gamma, total_steps=8, live_tail_steps=4)


if __name__ == "__main__":
    unittest.main()
