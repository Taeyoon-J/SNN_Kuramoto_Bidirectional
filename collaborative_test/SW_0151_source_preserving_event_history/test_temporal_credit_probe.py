from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for import_root in (ROOT, ROOT / "snn_kuramoto_bidirectional"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from collaborative_test.SW_0151_source_preserving_event_history.temporal_credit_probe import (
    DetachedAuxiliaryState, run_temporal_credit_probe,
)
from collaborative_test.SW_0151_source_preserving_event_history.test_adaptive_rollout import _wrapped_pair
from collaborative_test.SW_0151_source_preserving_event_history.event_history import AdaptiveEventHistory


class TemporalCreditProbeTests(unittest.TestCase):
    def test_detached_control_keeps_forward_values_but_cuts_cross_step_credit(self):
        torch.manual_seed(15131)
        live = AdaptiveEventHistory()
        detached = DetachedAuxiliaryState()
        detached.load_state_dict(copy.deepcopy(live.state_dict()), strict=True)
        with torch.no_grad():
            live.kappa_raw.fill_(0.8)
            detached.kappa_raw.fill_(0.8)
        base = torch.full((4,), 0.06)
        scale = 0.5
        gates = torch.ones(1, 4, 1)
        membranes = [
            torch.tensor([[[.063, .064], [.062, .067], [.061, .065], [.062, .063]]]),
            torch.tensor([[[.064, .062], [.066, .061], [.065, .063], [.064, .061]]]),
            torch.tensor([[[.061, .066], [.063, .064], [.062, .067], [.065, .064]]]),
        ]

        def run(helper):
            previous = torch.zeros_like(membranes[0])
            history = torch.zeros_like(membranes[0])
            outputs = []
            for membrane in membranes:
                result = helper(membrane, gates, base, scale, previous, history)
                previous, history = result["previous_event"], result["history"]
                outputs.append(result)
            objective = outputs[-1]["spike"].sum()
            grads = torch.autograd.grad(objective, (helper.lambda_logit, helper.kappa_raw))
            return outputs, grads

        live_out, live_grads = run(live)
        detached_out, detached_grads = run(detached)
        for left, right in zip(live_out, detached_out):
            for key in ("event", "spike", "history", "previous_event", "threshold"):
                self.assertTrue(torch.equal(left[key], right[key]), key)
        self.assertTrue(torch.isfinite(live_grads[0]).all())
        self.assertTrue(torch.isfinite(detached_grads[0]).all())
        self.assertGreater(float(live_grads[0].abs().sum().detach()), 0.0)
        self.assertGreater(float((live_grads[0] - detached_grads[0]).abs().sum().detach()), 0.0)

    def test_actual_native_and_adaptive_q_probe_preserves_state_and_matches_at_zero(self):
        torch.manual_seed(15132)
        source, _ = _wrapped_pair(seed=15133)
        gamma = torch.randn(1, 8, 4, dtype=torch.float32)
        before = copy.deepcopy(source.state_dict())
        rng_before = torch.random.get_rng_state().clone()
        result = run_temporal_credit_probe(
            source, gamma, scale=0.5, positive_kappa=0.5,
            total_steps=1024, live_tail_steps=64, settle_steps=512)
        self.assertEqual(result["status"], "probe_complete")
        self.assertFalse(result["training_admission"])
        self.assertTrue(result["zero_kappa_trace_exact_native"])
        self.assertTrue(result["zero_kappa_actual_spike_q_exact_native"])
        self.assertTrue(result["positive_live_vs_detached_forward_exact"])
        self.assertTrue(result["positive_live_vs_detached_q_exact"])
        self.assertTrue(result["reference_rollouts_no_grad"])
        self.assertTrue(result["source_model_state_unchanged"])
        self.assertTrue(result["gamma_unchanged"])
        self.assertTrue(result["caller_rng_restored"])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for key, value in before.items():
            self.assertTrue(torch.equal(source.state_dict()[key], value), key)
        self.assertIn("positive_kappa_event_flips_vs_zero", result)
        self.assertIn("positive_history_variance", result)
        for branch in ("positive_live_q_gradient", "positive_detached_control_q_gradient"):
            self.assertTrue(result[branch]["lambda_logit"]["finite"])
            self.assertTrue(result[branch]["kappa_raw"]["finite"])

    def test_rejects_gamma_node_shape_that_differs_from_native_core(self):
        source, _ = _wrapped_pair(seed=15134)
        gamma = torch.zeros(1, 8, 5, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "node dimension"):
            run_temporal_credit_probe(source, gamma, scale=0.5, positive_kappa=0.5)


if __name__ == "__main__":
    unittest.main()
