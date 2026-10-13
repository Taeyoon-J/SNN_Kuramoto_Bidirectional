from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0151_source_preserving_event_history.event_history import AdaptiveEventHistory
from snn_kuramoto_bidirectional.membrane_layer import act_fun_adp


class EventHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(151)
        self.membrane = torch.randn(2, 4, 11, dtype=torch.float32)
        self.gate = torch.rand(2, 4, 11)
        self.threshold = torch.tensor([0.04, 0.06, 0.09, 0.12])
        self.zero = torch.zeros_like(self.membrane)

    def test_zero_kappa_matches_native_events_and_spikes_for_existing_threshold(self):
        kernel = AdaptiveEventHistory()
        previous = torch.zeros_like(self.membrane)
        history = torch.zeros_like(self.membrane)
        for step in range(7):
            membrane = self.membrane + step * 0.07
            gate = self.gate.clone()
            if step % 2 == 0:
                gate[:, :, step::4] = 0
            result = kernel(membrane, gate, self.threshold, 0.3, previous, history)
            native_event = act_fun_adp(membrane - self.threshold.view(1, 4, 1))
            self.assertTrue(torch.equal(result["event"], native_event))
            self.assertTrue(torch.equal(result["spike"], native_event * gate))
            self.assertTrue(torch.equal(result["threshold"], self.threshold.view(1, 4, 1).expand_as(membrane)))
            previous, history = result["previous_event"], result["history"]

    def test_zero_gate_holds_both_auxiliary_states_elementwise(self):
        kernel = AdaptiveEventHistory()
        gate = torch.ones_like(self.gate)
        gate[:, :, 2::3] = 0
        prev = torch.full_like(self.membrane, 0.25)
        hist = torch.full_like(self.membrane, 0.75)
        result = kernel(self.membrane, gate, self.threshold, 0.3, prev, hist)
        inactive = gate == 0
        self.assertTrue(torch.equal(result["spike"][inactive], torch.zeros_like(result["spike"][inactive])))
        self.assertTrue(torch.equal(result["history"][inactive], hist[inactive]))
        self.assertTrue(torch.equal(result["previous_event"][inactive], prev[inactive]))
        self.assertTrue(torch.equal(result["history"][~inactive], (0.95 * hist + 0.05 * prev)[~inactive]))

    def test_history_credit_and_detached_state(self):
        kernel = AdaptiveEventHistory()
        with torch.no_grad():
            kernel.kappa_raw.fill_(0.4)
        prev = torch.ones_like(self.membrane)
        hist = torch.zeros_like(self.membrane)
        result = kernel(self.membrane, torch.ones_like(self.gate), self.threshold, 0.3, prev, hist,
                        detach_state=True)
        loss = result["spike"].sum()
        grad_lambda, grad_kappa = torch.autograd.grad(loss, (kernel.lambda_logit, kernel.kappa_raw))
        self.assertTrue(torch.isfinite(grad_lambda).all())
        self.assertGreater(float(grad_lambda.abs().sum().detach()), 0.0)
        self.assertTrue(torch.isfinite(grad_kappa).all())
        self.assertGreater(float(grad_kappa.abs().sum().detach()), 0.0)
        self.assertFalse(result["history"].requires_grad)
        self.assertIsNone(result["history"].grad_fn)
        self.assertFalse(result["previous_event"].requires_grad)
        self.assertNotEqual(result["history"].data_ptr(), hist.data_ptr())

    def test_live_multistep_history_credit_and_explicit_boundary_detach(self):
        def rollout(detach_boundary):
            kernel = AdaptiveEventHistory()
            with torch.no_grad():
                kernel.kappa_raw.fill_(0.4)
            event = torch.full((1, 4, 3), 0.7, requires_grad=True)
            history = torch.zeros_like(event, requires_grad=True)
            initial_event, initial_history = event, history
            membranes = [torch.full((1, 4, 3), 0.12 + 0.01 * i, requires_grad=True)
                         for i in range(4)]
            for i, membrane in enumerate(membranes):
                result = kernel(membrane, torch.ones_like(event), self.threshold, 0.4,
                                event, history, detach_state=(detach_boundary and i == 1))
                event, history = result["previous_event"], result["history"]
            return torch.autograd.grad(
                result["spike"].sum(),
                (kernel.lambda_logit, membranes[0], initial_event, initial_history),
                allow_unused=True,
            )

        live_grad, live_membrane_grad, live_event_grad, live_history_grad = rollout(False)
        boundary_grad, boundary_membrane_grad, boundary_event_grad, boundary_history_grad = rollout(True)
        self.assertTrue(torch.isfinite(live_grad).all())
        self.assertGreater(float(live_grad.abs().sum().detach()), 0.0)
        self.assertTrue(torch.isfinite(boundary_grad).all())
        self.assertGreater(float(boundary_grad.abs().sum().detach()), 0.0)
        self.assertFalse(torch.equal(live_grad, boundary_grad))
        for prior_grad in (live_membrane_grad, live_event_grad, live_history_grad):
            self.assertIsNotNone(prior_grad)
            self.assertTrue(torch.isfinite(prior_grad).all())
            self.assertGreater(float(prior_grad.abs().sum().detach()), 0.0)
        self.assertIsNone(boundary_membrane_grad)
        self.assertIsNone(boundary_event_grad)
        self.assertIsNone(boundary_history_grad)

    def test_component_specific_parameters_use_batch_component_order(self):
        kernel = AdaptiveEventHistory()
        with torch.no_grad():
            kernel.kappa_raw.copy_(torch.tensor([0.0, 0.2, 0.4, 0.6]))
            kernel.lambda_logit.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))
        batch, nodes = 3, 5
        membrane = torch.full((batch, 4, nodes), 0.25)
        previous, history = torch.ones_like(membrane), torch.zeros_like(membrane)
        result = kernel(membrane, torch.ones_like(membrane), self.threshold, 0.5,
                        previous, history)
        lam = torch.sigmoid(kernel.lambda_logit).view(1, 4, 1).expand(batch, 4, nodes)
        kappa = kernel.kappa.view(1, 4, 1).expand(batch, 4, nodes)
        expected_history = (1.0 - lam) * previous
        expected_threshold = self.threshold.view(1, 4, 1) + kappa * 0.5 * expected_history
        self.assertTrue(torch.equal(result["history"], expected_history))
        self.assertTrue(torch.equal(result["threshold"], expected_threshold.expand_as(membrane)))
        self.assertTrue(torch.equal(result["threshold"][0], result["threshold"][2]))

    def test_recurrent_inputs_are_not_mutated(self):
        kernel = AdaptiveEventHistory()
        previous, history = torch.rand_like(self.membrane), torch.rand_like(self.membrane)
        previous_before, history_before = previous.clone(), history.clone()
        kernel(self.membrane, self.gate, self.threshold, 0.3, previous, history)
        self.assertTrue(torch.equal(previous, previous_before))
        self.assertTrue(torch.equal(history, history_before))

    def test_zero_kappa_lambda_gradient_is_expected_at_source_initialization(self):
        kernel = AdaptiveEventHistory()
        result = kernel(
            self.membrane,
            torch.ones_like(self.gate),
            self.threshold,
            0.3,
            torch.ones_like(self.membrane),
            torch.zeros_like(self.membrane),
        )
        grad = torch.autograd.grad(result["spike"].sum(), kernel.lambda_logit)[0]
        self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_projection_and_input_contracts(self):
        kernel = AdaptiveEventHistory()
        with torch.no_grad():
            kernel.kappa_raw[0] = -0.4
        kernel.project_parameters_()
        self.assertTrue(torch.all(kernel.kappa_raw >= 0))
        with self.assertRaises(ValueError):
            kernel(self.membrane[:, :3], self.gate[:, :3], self.threshold[:3], 0.2, self.zero[:, :3], self.zero[:, :3])
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate, self.threshold, 0.0, self.zero, self.zero)
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate, self.threshold, torch.ones(2, 2), self.zero, self.zero)
        scale = torch.tensor(0.2, requires_grad=True)
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate, self.threshold, scale, self.zero, self.zero)
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate, torch.tensor([.1, .2, float("nan"), .3]), .2, self.zero, self.zero)
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate[:, :, :5], self.threshold, .2, self.zero, self.zero)
        with self.assertRaises(ValueError):
            kernel(self.membrane.double(), self.gate.double(), self.threshold, .2,
                   self.zero.double(), self.zero.double())
        with self.assertRaises(ValueError):
            kernel(self.membrane, self.gate.double(), self.threshold, .2, self.zero, self.zero)


if __name__ == "__main__":
    unittest.main()
