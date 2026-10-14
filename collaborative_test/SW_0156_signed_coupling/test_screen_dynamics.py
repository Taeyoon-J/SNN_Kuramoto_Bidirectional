import unittest

import torch

from collaborative_test.SW_0156_signed_coupling.screen import _component_dynamics


class ScreenDynamicsTests(unittest.TestCase):
    def test_component_statistics_use_settled_actual_traces_and_q_offdiagonal(self):
        spikes = torch.zeros(1, 4, 3, 514)
        gates = torch.zeros_like(spikes)
        gates[:, :, 1:, 512:] = 0.5
        gates[:, 0] = 0
        # The recorded spike trace is emitted E*g; never place a spike where
        # its gate is zero. Component 0 remains an unobserved (zero support) case.
        spikes[:, 1, 1:, 512:] = gates[:, 1, 1:, 512:]
        q = torch.tensor([[[1., .2, .4], [.2, 1., .6], [.4, .6, 1.]]])
        stats = _component_dynamics({"component_spikes": spikes,
                                     "component_gates": gates}, q)
        self.assertEqual(stats["settled_steps"], 2)
        self.assertEqual(stats["per_component"][0]["weighted_spike_mean"], 0)
        self.assertIsNone(stats["per_component"][0]["event_on_fraction_when_gate_positive"])
        self.assertEqual(stats["per_component"][0]["gate_positive_support_count"], 0)
        self.assertAlmostEqual(stats["per_component"][1]["weighted_spike_mean"], 1 / 3)
        self.assertAlmostEqual(stats["per_component"][1]["spike_variance"], 1 / 18)
        self.assertEqual(stats["per_component"][1]["event_on_fraction_when_gate_positive"], 1)
        self.assertEqual(stats["per_component"][1]["gate_positive_support_count"], 4)
        self.assertEqual(stats["per_component"][0]["gate_mean"], 0)
        self.assertEqual(stats["per_component"][0]["gate_active_fraction"], 0)
        self.assertEqual(stats["per_component"][1]["gate_mean"], 1 / 3)
        self.assertEqual(stats["per_component"][1]["gate_active_fraction"], 2 / 3)
        self.assertAlmostEqual(stats["actual_spike_q_offdiagonal"]["mean"], .4)
        self.assertAlmostEqual(stats["actual_spike_q_offdiagonal"]["min"], .2)
        self.assertAlmostEqual(stats["actual_spike_q_offdiagonal"]["max"], .6)

    def test_bad_q_or_component_schema_fails_closed(self):
        trace = {"component_spikes": torch.zeros(1, 4, 2, 514),
                 "component_gates": torch.zeros(1, 4, 2, 514)}
        with self.assertRaises(ValueError):
            _component_dynamics(trace, torch.eye(3).unsqueeze(0))
        trace["component_gates"] = torch.zeros(1, 3, 2, 514)
        with self.assertRaises(ValueError):
            _component_dynamics(trace, torch.eye(2).unsqueeze(0))


if __name__ == "__main__":
    unittest.main()
